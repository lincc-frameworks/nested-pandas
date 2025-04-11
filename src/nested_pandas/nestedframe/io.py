# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations
import os
import io

import pandas as pd
from pandas._libs import lib
from pandas._typing import (
    DtypeBackend,
    FilePath,
    ReadBuffer,
)
import pyarrow as pa
import pyarrow.dataset as ds
import fsspec
import pyarrow.parquet as pq

import requests
import fsspec
from urllib.parse import urlparse

from .core import NestedFrame
from ..series.dtype import NestedDtype


def read_parquet(
    data: FilePath | ReadBuffer[bytes],
    columns: list[str] | None = None,
    reject_nesting: list[str] | str | None = None,
) -> NestedFrame:
    """
    Load a parquet object from a file path into a NestedFrame.

    As a deviation from `pandas`, this function loads via 
    `pyarrow.parquet.read_table`, and then converts to a NestedFrame.

    Parameters
    ----------
    data: str, pyarrow.NativeFile, or file-like object
        Path to the data. If a string passed, can be a single file name or
        directory name. For file-like objects, only read a single file.
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

    Furthermore, there are some cases where subcolumns will have the same name
    as a top-level column. For example, if you have a column "nested" with
    subcolumns "nested.a" and "nested.b", and also a top-level column "a". In
    these cases, keep in mind that if "nested" is in the reject_nesting list
    the operation will fail (but nesting will still work normally).
    """

    # Type convergence for reject_nesting
    if reject_nesting is None:
        reject_nesting = []
    elif isinstance(reject_nesting, str):
        reject_nesting = [reject_nesting]

    # TODO: This potentially can't read remote files
    # maybe load into a pyarrow.dataset first
    """
    if isinstance(data, str):
        # Check if the file is a URL
        if data.startswith("http://") or data.startswith("https://"):
            # Use fsspec to open the file
            fs = fsspec.filesystem("http")
            with fs.open(data) as f:
                table = pq.read_table(f, columns=columns)
        else:
            # Use pyarrow to read the file directly
            table = pq.read_table(data, columns=columns)
    """

    # First load through pyarrow
    # This will handle local files, http(s) and s3
    table = _fs_read_table(data, use_fsspec=True, columns=columns)

    # Resolve partial loading of nested structures
    # Using pyarrow to avoid naming conflicts from partial loading ("flux" vs "lc.flux")
    # Use input column names and the table column names to determine if a column
    # was from a nested column.
    if columns is not None:
        nested_structures = {}
        for i, (col_in, col_pa) in enumerate(zip(columns, table.column_names)):
            # if the column name is not the same, it was a partial load
            if col_in != col_pa:
                # get the top-level column name
                nested_col = col_in.split(".")[0]
                if nested_col not in reject_nesting:
                    if nested_col not in nested_structures.keys():
                        nested_structures[nested_col] = [i]
                    else:
                        nested_structures[nested_col].append(i)

        #print(nested_structures)
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

        # Build structs and track column indices used
        structs = {}
        indices_to_remove = []
        for col, indices in nested_structures.items():
            print(f"Processing nested column: {col}, indices: {indices}")

            # Build a struct column from the columns
            field_names = [table.column_names[i] for i in indices]
            structs[col] = pa.StructArray.from_arrays(
                [table.column(i).chunk(0) for i in indices],  # Child arrays
                field_names  # Field names
            )
            indices_to_remove.extend(indices)

        # Remove the original columns in reverse order to avoid index shifting
        for i in sorted(indices_to_remove, reverse=True):
            print(f"Removing column at index {i}: {table.column_names[i]}")
            table = table.remove_column(i)

        # Append the new struct columns
        for col, struct in structs.items():
            print(f"Appending struct column: {col}")
            table = table.append_column(col, struct)

    # Convert to NestedFrame
    # not zero-copy, but reduce memory pressure via the self_destruct kwarg
    # https://arrow.apache.org/docs/python/pandas.html#reducing-memory-use-in-table-to-pandas
    df = NestedFrame(table.to_pandas(types_mapper=lambda ty: pd.ArrowDtype(ty), self_destruct=True))
    del table

    #print(df.dtypes)
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


def _fs_read_table(uri, use_fsspec=True, headers=None, **kwargs):
    """
    A smart wrapper around `pq.read_table` that handles multiple filesystems.

    Parameters
    ----------
    uri (str):
        path or URI to a Parquet file
    use_fsspec (bool):
        whether to use fsspec for URI handling (e.g., for S3)
    headers (dict):
        headers for HTTP requests (optional)
    kwargs:
        other keyword arguments passed to `pq.read_table`

    Returns
    -------
    pyarrow.Table
    """
    parsed = urlparse(uri)

    # --- Local file or file:// URI ---
    if parsed.scheme in ("", "file"):
        return pq.read_table(uri, **kwargs)

    # --- HTTP/HTTPS via requests ---
    elif parsed.scheme in ("http", "https"):
        if use_fsspec:
            fs = fsspec.filesystem("http")
            with fs.open(uri, mode="rb") as f:
                return pq.read_table(f, **kwargs)
        else:
            response = requests.get(uri, headers=headers or {}, stream=True)
            response.raise_for_status()
            buf = pa.BufferReader(response.content)
            return pq.read_table(buf, **kwargs)

    # --- S3/GS/etc via fsspec ---
    elif use_fsspec:
        fs, path = fsspec.core.url_to_fs(uri)
        with fs.open(path, mode="rb") as f:
            return pq.read_table(f, **kwargs)

    else:
        raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")