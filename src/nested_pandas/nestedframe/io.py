# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import pandas as pd
from pandas._libs import lib
from pandas._typing import (
    DtypeBackend,
    FilePath,
    ReadBuffer,
)

from .core import NestedFrame


def read_parquet(
    data: FilePath | ReadBuffer[bytes],
    to_pack: dict | None = None,
    columns: list[str] | None = None,
    pack_columns: dict | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> NestedFrame:
    """
    Load a parquet object from a file path and load a set of other
    parquet objects to pack into the resulting NestedFrame.

    Docstring based on the Pandas equivalent.

    #TODO after MVP: Include full kwarg-set
    #TODO: Switch dtype backend default?

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
    dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

    Returns
    -------
    NestedFrame
    """

    df = NestedFrame(pd.read_parquet(data, engine="pyarrow", columns=columns, dtype_backend=dtype_backend))
    if to_pack is None:
        return df
    for pack_key in to_pack:
        col_subset = pack_columns.get(pack_key, None) if pack_columns is not None else None
        packed = pd.read_parquet(
            to_pack[pack_key], engine="pyarrow", columns=col_subset, dtype_backend=dtype_backend
        )
        df = df.add_nested(packed, pack_key)

    return df
