# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from upath import UPath

from ..series.dtype import NestedDtype
from .core import NestedFrame


def read_parquet(
    data: str | UPath,
    columns: list[str] | None = None,
    reject_nesting: list[str] | str | None = None,
) -> NestedFrame:
    """
    Load a parquet object from a file path into a NestedFrame.

    As a deviation from `pandas`, this function loads via
    `pyarrow.parquet.read_table`, and then converts to a NestedFrame.

    Parameters
    ----------
    data: str or Upath
        Path to the data. If a string passed, can be a single file name or
        directory name. For file-like objects, only read a single file.
        Can be a local file path, HTTP/HTTPS URL, or S3 path.
    columns : list, default=None
        If not None, only these columns will be read from the file.
    reject_nesting: list or str, default=None
        Column(s) to reject from being cast to a nested dtype. By default,
        nested-pandas assumes that any struct column with all fields being lists
        is castable to a nested column. However, this assumption is invalid if
        the lists within the struct have mismatched lengths for any given item.
        Columns specified here will be read using the corresponding pandas.ArrowDtype.

    Returns
    -------
    NestedFrame

    Notes
    -----
    pyarrow supports partial loading of nested structures from parquet, for
    example ```pd.read_parquet("data.parquet", columns=["nested.a"])``` will
    load the "a" column of the "nested" column. Standard pandas/pyarrow
    behavior will return "a" as a list-array base column with name "a". In
    nested-pandas, this behavior is changed to load the column as a sub-column
    of a nested column called "nested". Be aware that this will prohibit calls
    like ```pd.read_parquet("data.parquet", columns=["nested.a", "nested"])```
    from working, as this implies both full and partial load of "nested".

    Furthermore, there are some cases where subcolumns will have the same name
    as a top-level column. For example, if you have a column "nested" with
    subcolumns "nested.a" and "nested.b", and also a top-level column "a". In
    these cases, keep in mind that if "nested" is in the reject_nesting list
    the operation will fail, as is consistent with the default pandas behavior
    (but nesting will still work normally).

    Examples
    --------

    Simple loading example:

    >>> import nested_pandas as npd
    >>> nf = npd.read_parquet("path/to/file.parquet")

    Partial loading:

    >>> #Load only the "flux" sub-column of the "nested" column
    >>> nf = npd.read_parquet("path/to/file.parquet", columns=["a", "nested.flux"])
    """

    # Type convergence for reject_nesting
    if reject_nesting is None:
        reject_nesting = []
    elif isinstance(reject_nesting, str):
        reject_nesting = [reject_nesting]

    # First load through pyarrow
    # Use UPath to handle the file path
    path = UPath(data)

    # TODO: Support file-object loading

    # use upath to support remote filesystems
    with path.open("rb") as f:
        table = pq.read_table(f, columns=columns)

    # Resolve partial loading of nested structures
    # Using pyarrow to avoid naming conflicts from partial loading ("flux" vs "lc.flux")
    # Use input column names and the table column names to determine if a column
    # was from a nested column.
    if columns is not None:
        nested_structures: dict[str, list[int]] = {}
        for i, (col_in, col_pa) in enumerate(zip(columns, table.column_names)):
            # if the column name is not the same, it was a partial load
            if col_in != col_pa:
                # get the top-level column name
                nested_col = col_in.split(".")[0]

                # validate that the partial load columns are list type
                # if any of the columns are not list type, reject the cast
                # and remove the column from the list of nested structures if
                # it was added
                if not pa.types.is_list(table.schema[i].type):
                    reject_nesting.append(nested_col)
                    if nested_col in nested_structures:
                        # remove the column from the list of nested structures
                        nested_structures.pop(nested_col)
                # track nesting for columns not in the reject list
                elif nested_col not in reject_nesting:
                    if nested_col not in nested_structures:
                        nested_structures[nested_col] = [i]
                    else:
                        nested_structures[nested_col].append(i)

        # Check for full and partial load of the same column and error
        # Columns in the reject_nesting will not be checked
        for col in columns:
            if col in nested_structures:
                raise ValueError(
                    f"The provided column list contains both a full and partial "
                    f"load of the column '{col}'. This is not allowed as the partial "
                    "load will be cast to a nested column that already exists. "
                    "Please either remove the partial load or the full load."
                )

        # Build structs and track column indices used
        structs = {}
        indices_to_remove = []
        for col, indices in nested_structures.items():
            # Build a struct column from the columns
            field_names = [table.column_names[i] for i in indices]

            # Use iterchunks to process chunks of each column
            chunked_arrays = [pa.concat_arrays(list(table.column(i).iterchunks())) for i in indices]

            structs[col] = pa.StructArray.from_arrays(
                chunked_arrays,  # Child arrays
                field_names,  # Field names
            )
            indices_to_remove.extend(indices)

        # Remove the original columns in reverse order to avoid index shifting
        for i in sorted(indices_to_remove, reverse=True):
            table = table.remove_column(i)

        # Append the new struct columns
        for col, struct in structs.items():
            table = table.append_column(col, struct)

    # Convert to NestedFrame
    # not zero-copy, but reduce memory pressure via the self_destruct kwarg
    # https://arrow.apache.org/docs/python/pandas.html#reducing-memory-use-in-table-to-pandas
    df = NestedFrame(
        table.to_pandas(types_mapper=lambda ty: pd.ArrowDtype(ty), split_blocks=True, self_destruct=True)
    )
    del table
    # Attempt to cast struct columns to NestedDTypes
    df = _cast_struct_cols_to_nested(df, reject_nesting)

    return df


def _cast_struct_cols_to_nested(df, reject_nesting):
    """cast struct columns to nested dtype"""
    # Attempt to cast struct columns to NestedDTypes
    for col, dtype in df.dtypes.items():
        # First validate the dtype
        # will return valueerror when not a struct-list
        valid_dtype = True
        try:
            NestedDtype._validate_dtype(dtype.pyarrow_dtype)
        except ValueError:
            valid_dtype = False

        if valid_dtype and col not in reject_nesting:
            try:
                # Attempt to cast Struct to NestedDType
                df = df.astype({col: NestedDtype(dtype.pyarrow_dtype)})
            except ValueError as err:
                # If cast fails, the struct likely does not fit nested-pandas
                # criteria for a valid nested column
                raise ValueError(
                    f"Column '{col}' is a Struct, but an attempt to cast it to a NestedDType failed. "
                    "This is likely due to the struct not meeting the requirements for a nested column "
                    "(all fields should be equal length). To proceed, you may add the column to the "
                    "`reject_nesting` argument of the read_parquet function to skip the cast attempt."
                ) from err
    return df
