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
    to_pack: dict | None = None,
    columns: list[str] | None = None,
    pack_columns: dict | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    reject_nesting: list[str] | str | None = None,
    **kwargs,
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
    to_pack: dict, default=None
        A dictionary of parquet data paths (same criteria as `data`), where
        each key reflects the desired column name to pack the data into and
        each value reflects the parquet data to pack. If None, it assumes
        that any data to pack is already packed as a column within `data`.
    columns : list, default=None
        If not None, only these columns will be read from the file.
    pack_columns: dict, default=None
        If not None, selects a set of columns from each keyed nested parquet
        object to read from the nested files.
    reject_nesting: list or str, default=None
        Column(s) to reject from being cast to a nested dtype. By default,
        nested-pandas assumes that any struct column is castable to a nested
        column, but this is not always the case for a given struct. Any columns
        specified here will be read as their original struct type.

    Returns
    -------
    NestedFrame
    """

    df = NestedFrame(pd.read_parquet(data, engine="pyarrow", columns=columns, dtype_backend="pyarrow", **kwargs))

    # Type convergence for reject_nesting
    if reject_nesting is None:
        reject_nesting = []
    elif isinstance(reject_nesting, str):
        reject_nesting = [reject_nesting]

    # Attempt to cast struct columns to NestedDTypes
    df = _cast_struct_cols_to_nested(df, reject_nesting)

    if to_pack is None:
        return df
    for pack_key in to_pack:
        col_subset = pack_columns.get(pack_key, None) if pack_columns is not None else None
        packed = pd.read_parquet(
            to_pack[pack_key], engine="pyarrow", columns=col_subset, dtype_backend="pyarrow"
        )
        df = df.add_nested(packed, pack_key)

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