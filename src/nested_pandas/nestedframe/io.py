# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import pandas as pd

from .core import NestedFrame


def read_parquet(
    data: str,
    to_pack: dict,
    engine: str = "auto",
    columns: list[str] | None = None,
    pack_columns: dict | None = None,
) -> NestedFrame:
    """
    Load a parquet object from a file path and load a set of other
    parquet objects to pack into the resulting NestedFrame.

    Docstring based on the Pandas equivalent.

    #TODO after MVP: Include full kwarg-set

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
    to_pack: dict,
        A dictionary of parquet data paths (same criteria as `data`), where
        each key reflects the desired column name to pack the data into and
        each value reflects the parquet data to pack.
    engine : {{'auto', 'pyarrow', 'fastparquet'}}, default 'auto'
        Parquet library to use. If 'auto', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try 'pyarrow', falling back to 'fastparquet' if
        'pyarrow' is unavailable.

        When using the ``'pyarrow'`` engine and no storage options are provided
        and a filesystem is implemented by both ``pyarrow.fs`` and ``fsspec``
        (e.g. "s3://"), then the ``pyarrow.fs`` filesystem is attempted first.
        Use the filesystem keyword with an instantiated fsspec filesystem
        if you wish to use its implementation.
    columns : list, default=None
        If not None, only these columns will be read from the file.
    pack_columns: dict, default=None
        If not None, selects a set of columns from each keyed nested parquet
        object to read from the nested files.

    Returns
    -------
    NestedFrame
    """

    df = NestedFrame(pd.read_parquet(data, engine, columns))

    for pack_key in to_pack:
        col_subset = pack_columns[pack_key] if pack_columns is not None else None
        packed = pd.read_parquet(to_pack[pack_key], engine=engine, columns=col_subset)
        df = df.add_nested(packed, pack_key)

    return df
