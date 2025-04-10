# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import pandas as pd
from pandas._libs import lib
from pandas._typing import (
    DtypeBackend,
    FilePath,
    ReadBuffer,
)
import pyarrow as pa

from .core import NestedFrame
from ..series.dtype import NestedDtype


def read_parquet(
    data: FilePath | ReadBuffer[bytes],
    columns: list[str] | None = None,
    reject_nesting: list[str] | str | None = None,
) -> NestedFrame:
    """
    Load a parquet object from a file path and load a set of other
    parquet objects to pack into the resulting NestedFrame.

    Docstring based on the Pandas equivalent. Pyarrow is automatically
    used as the `engine` and  `dtype_backend` for the read_parquet function. 

    Parameters
    ----------
    data : str, path object or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``read()`` function.
        The string could be a URL. Valid URL schemes include http, ftp, s3,
        gs, and file. For file URLs, a host is expected. A local file could be:
        ``file://localhost/path/to/table.parquet``.
        A file URL can also be a path to a directory that contains multiple
        partitioned parquet files. Both pyarrow and fastparquet support
        paths to directories as well as file URLs. A directory path could be:
        ``file://localhost/path/to/tables`` or ``s3://bucket/partition_dir``.
    columns : list, default=None
        If not None, only these columns will be read from the file.
    reject_nesting: list or str, default=None
        Column(s) to reject from being cast to a nested dtype. By default,
        nested-pandas assumes that any struct column is castable to a nested
        column, but this is not always the case for a given struct. Any columns
        specified here will be read as their original struct type.

    Returns
    -------
    NestedFrame

    Notes
    -----
    pyarrow supports partial loading of nested structures from parquet, for
    example ```pd.read_parquet("data.parquet", columns=["nested.a"])``` will
    load the "a" column of the "nested" column. Standard pandas/pyarrow
    behavior will return "a" as a list-array base column with name "a". In
    Nested-Pandas, this behavior is changed to load the column as a sub-column
    of a nested column called "nested". Be aware that this will prohibit calls
    like ```pd.read_parquet("data.parquet", columns=["nested.a", "nested"])```
    from working, as this implies both full and partial load of "nested".
    """

    # Type convergence for reject_nesting
    if reject_nesting is None:
        reject_nesting = []
    elif isinstance(reject_nesting, str):
        reject_nesting = [reject_nesting]

    # First load through pyarrow
    table = pa.parquet.read_pandas(
        data,
        columns=columns)

    # Resolve partial loading of nested structures
    # Using pyarrow to avoid naming conflicts from partial loading ("flux" vs "lc.flux")
    # Use input column names and the table column names to determine if a column
    # was from a nested column.
    nested_structures = {}
    for col_in, col_pa in zip(columns, table.column_names):
        # if the column name is not the same, it was a partial load
        if col_in != col_pa:
            # get the top-level column name
            nested_col = col_in.split(".")[0]
            if nested_col not in reject_nesting:
                if nested_col not in nested_structures.keys():
                    nested_structures[nested_col] = [table.column_names.index(col_pa)]
                else:
                    nested_structures[nested_col].append(table.column_names.index(col_pa))

    # Check for full and partial load of the same column and error
    # Columns in the reject_nesting will not be checked
    for col in columns:
        if col in nested_structures.keys():
            raise ValueError(
                f"The provided column list contains both a full and partial "
                f"load of the column '{col}'. This is not allowed as the partial "
                "load will be cast to a nested column that already exists. "
                "Please either remove the partial load or the full load."
            )

    # Build structs and replace columns in table
    for col, indices in nested_structures.items():
        # Build a struct column from the columns
        field_names = [table.column_names[i] for i in indices]
        struct = pa.StructArray.from_arrays([table.column(i).chunk(0) for i in indices], field_names)
        # Replace the columns with the struct column
        # sort in reverse order to avoid index shifting
        for i in sorted(indices, reverse=True):
            table = table.remove_column(i)
        table = table.append_column(col, struct)

    # Convert to NestedFrame
    # TODO: How much of a problem is it that this is not zero_copy? True below fails
    df = NestedFrame(table.to_pandas(types_mapper=lambda ty: pd.ArrowDtype(ty), zero_copy_only=False))

    # Attempt to cast struct columns to NestedDTypes
    df = _cast_struct_cols_to_nested(df, reject_nesting)

    return df


def _cast_struct_cols_to_nested(df, reject_nesting):
    """cast struct columns to nested dtype"""
    # Attempt to cast struct columns to NestedDTypes
    for col, dtype in df.dtypes.items():
        if pa.types.is_struct(dtype.pyarrow_dtype) and col not in reject_nesting:
            try:
                # Attempt to cast Struct to NestedDType
                df = df.astype({col: NestedDtype(dtype.pyarrow_dtype)})
            except TypeError:
                # If cast fails, the struct likely does not fit nested-pandas
                # criteria for a valid nested column
                raise ValueError(
                    f"""Column {col} is a Struct, but an attempt to cast it to
                    a NestedDType failed. This is likely due to the struct
                    not meeting the requirements for a nested column (all
                    fields should be equal length). To proceed, you may add the
                    column to the `reject_nesting` argument of the read_parquet
                    function to skip the cast attempt.
                    """
                )
    return df