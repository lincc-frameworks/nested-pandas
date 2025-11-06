# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

from pathlib import Path

import fsspec.parquet
import pandas as pd
import pyarrow as pa
import pyarrow.fs
import pyarrow.parquet as pq
from pyarrow.lib import ArrowInvalid
from upath import UPath

from ..series.dtype import NestedDtype
from ..series.packer import pack_lists
from ..series.utils import table_to_struct_array
from .core import NestedFrame

# Use smaller block size for these FSSPEC filesystems.
# It usually helps with parquet read speed.
FSSPEC_FILESYSTEMS = ("http", "https")
FSSPEC_BLOCK_SIZE = 32 * 1024

# Filesystems for which calling .is_dir() may be very slow and/or .iterdir()
# may yield non-parquet paths. See details in _read_parquet_into_table()
# docstring.
NO_ITERDIR_FILESYSTEMS = (
    "http",
    "https",
)


def read_parquet(
    data: str | UPath | bytes,
    columns: list[str] | None = None,
    reject_nesting: list[str] | str | None = None,
    autocast_list: bool = False,
    **kwargs,
) -> NestedFrame:
    """Load a parquet object from a file path into a NestedFrame.

    As a specialization of the ``pandas.read_parquet`` function, this
    function loads the data via existing ``pyarrow`` or
    ``fsspec.parquet`` methods, and then converts the data to a
    NestedFrame.

    Parameters
    ----------
    data: str, list or str, Path, Upath, or file-like object
        Path to the data or a file-like object. If a string is passed,
        it can be a single file name, directory name, or a remote path
        (e.g., HTTP/HTTPS or S3). If a file-like object is passed, it
        must support the ``read`` method. You can also pass a
        ``filesystem`` keyword argument with a ``pyarrow.fs`` object, which
        will be passed along to the underlying file-reading method.
        A file URL can also be a path to a directory that contains multiple
        partitioned parquet files. Both pyarrow and fastparquet support
        paths to directories as well as file URLs. A directory path could be:
        ``file://localhost/path/to/tables/`` or ``s3://bucket/partition_dir/``
        (note trailing slash for web locations, since it may be expensive
        to test a path for being a directory). Directory reading is not
        supported for HTTP(S). If the path is to a single Parquet file, it will
        be loaded using
        ``fsspec.parquet.open_parquet_file``, which has optimized handling
        for remote Parquet files.
    columns : list, default=None
        If not None, only these columns will be read from the file.
    reject_nesting: list or str, default=None
        Column(s) to reject from being cast to a nested dtype. By default,
        nested-pandas assumes that any struct column with all fields being lists
        is castable to a nested column. However, this assumption is invalid if
        the lists within the struct have mismatched lengths for any given item.
        Columns specified here will be read using the corresponding pandas.ArrowDtype.
    autocast_list: bool, default=True
        If True, automatically cast list columns to nested columns with NestedDType.
    kwargs: dict
        Keyword arguments passed to `pyarrow.parquet.read_table`

    Returns
    -------
    NestedFrame

    Notes
    -----
    For paths to single Parquet files, this function uses
    fsspec.parquet.open_parquet_file, which performs intelligent
    precaching.  This can significantly improve performance compared
    to standard PyArrow reading on remote files.

    pyarrow supports partial loading of nested structures from parquet, for
    example ```pd.read_parquet("data.parquet", columns=["nested.a"])``` will
    load the "a" column of the "nested" column. Standard pandas/pyarrow
    behavior will return "a" as a list-array base column with name "a". In
    nested-pandas, this behavior is changed to load the column as a sub-column
    of a nested column called "nested". Be aware that this will prohibit calls
    like ```pd.read_parquet("data.parquet", columns=["nested.a", "nested"])```
    from working, as this implies both full and partial load of "nested".

    Additionally with partial loading, be aware that nested-pandas (and pyarrow)
    only supports partial loading of struct of list columns. Your data may be
    stored as a list of structs, which can be read by nested-pandas, but without
    support for partial loading. We try to throw a helpful error message in these
    cases.

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
    >>> nf = npd.read_parquet("path/to/file.parquet")  # doctest: +SKIP

    Partial loading:

    >>> #Load only the "flux" sub-column of the "nested" column
    >>> nf = npd.read_parquet("path/to/file.parquet", columns=["a", "nested.flux"])  # doctest: +SKIP

    """

    # Type convergence for reject_nesting
    if reject_nesting is None:
        reject_nesting = []
    elif isinstance(reject_nesting, str):
        reject_nesting = [reject_nesting]

    table = _read_parquet_into_table(data, columns, **kwargs)

    # Resolve partial loading of nested structures
    # Using pyarrow to avoid naming conflicts from partial loading ("flux" vs "lc.flux")
    # Use input column names and the table column names to determine if a column
    # was from a nested column.
    if columns is not None:
        nested_structures: dict[str, list[int]] = {}
        for i, (col_in, col_pa) in enumerate(zip(columns, table.column_names, strict=True)):
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
            structs[col] = table_to_struct_array(table.select(indices))
            indices_to_remove.extend(indices)

        # Remove the original columns in reverse order to avoid index shifting
        for i in sorted(indices_to_remove, reverse=True):
            table = table.remove_column(i)

        # Append the new struct columns
        for col, struct in structs.items():
            table = table.append_column(col, struct)

    return from_pyarrow(table, reject_nesting=reject_nesting, autocast_list=autocast_list)


def _read_parquet_into_table(
    data: str | UPath | bytes,
    columns: list[str] | None,
    **kwargs,
) -> pa.Table:
    """Reads parquet file(s) from path and returns a pyarrow table.

    For single Parquet file paths, we want to use
    `fsspec.parquet.open_parquet_file`.  But for any other usage
    (which includes file-like objects, local directories and lists
    thereof), we want to defer to `pq.read_table`.

    NOTE: the test for _is_local_dir is sufficient, because we're
    preserving a path to pq.read_table, which can read local
    directories, but not remote directories. Remote directories
    are supported separately via _read_parquet_directory.
    We don't support HTTP "directories", because 1) calling .is_dir()
    may be very expensive, because it downloads content first,
    2) because .iter_dir() is likely to return a lot of "junk"
    besides of the actual parquet files.
    """
    if isinstance(data, str | Path | UPath) and not _is_local_dir(path_to_data := UPath(data)):
        storage_options = _get_storage_options(path_to_data)
        filesystem = kwargs.get("filesystem")
        if not filesystem:
            _, filesystem = _transform_read_parquet_data_arg(path_to_data)
        # Will not detect HTTP(S) directories.
        if _is_remote_dir(data, path_to_data):
            return _read_remote_parquet_directory(
                path_to_data, filesystem, storage_options, columns, **kwargs
            )

        with fsspec.parquet.open_parquet_file(
            path_to_data.path,
            columns=columns,
            storage_options=storage_options,
            fs=filesystem,
            engine="pyarrow",
        ) as parquet_file:
            return _read_table_with_partial_load_check(parquet_file, columns=columns, **kwargs)

    # All other cases, including file-like objects, directories, and
    # even lists of the foregoing.

    # If `filesystem` is specified - use it, passing it as part of **kwargs
    if kwargs.get("filesystem") is not None:
        return _read_table_with_partial_load_check(data, columns=columns, **kwargs)

    # Otherwise convert with a special function
    data, filesystem = _transform_read_parquet_data_arg(data)
    return _read_table_with_partial_load_check(data, columns=columns, filesystem=filesystem, **kwargs)


def _read_table_with_partial_load_check(data, columns=None, filesystem=None, **kwargs):
    """Read a pyarrow table with partial load check for nested structures"""
    try:
        return pq.read_table(data, columns=columns, filesystem=filesystem, **kwargs)
    except ArrowInvalid as e:
        # if it's not related to partial loading of nested structures, re-raise
        if "No match for" not in str(e):
            raise e
        if columns is not None:
            check_schema = any("." in col for col in columns)  # Check for potential partial loads
            if check_schema:
                try:
                    _validate_structs_from_schema(data, columns=columns, filesystem=filesystem)
                except ValueError as validation_error:
                    raise validation_error from e  # Chain the exceptions for better context
        raise e


def _validate_structs_from_schema(data, columns=None, filesystem=None):
    """Validate that nested columns are structs"""
    if columns is not None:
        schema = pq.read_schema(data, filesystem=filesystem)
        for col in columns:
            # check if column is a partial load of a nested structure
            if "." in col:
                # first check if column exists as a top-level column
                if col in schema.names:
                    continue
                # if not, inspect the base column name type
                else:
                    if col.split(".")[0] in schema.names:
                        # check if the column is a list-struct
                        col_type = schema.field(col.split(".")[0]).type
                        if not pa.types.is_struct(col_type):
                            base_col = col.split(".")[0]
                            raise ValueError(
                                f"The provided column '{col}' signals to partially load a nested structure, "
                                f"but the nested structure '{base_col}' is not a struct. "
                                "Partial loading of nested structures is only supported for struct of list "
                                f"columns. To resolve this, fully load the column '{base_col}' "
                                f"instead of partially loading it and perform column selection afterwards."
                            )


def _is_local_dir(upath: UPath) -> bool:
    """Returns True if the given path refers to a local directory.

    It's necessary to have this function, rather than simply checking
    ``UPath(p).is_dir()``, because ``UPath.is_dir`` can be quite
    expensive in the case of a remote file path that isn't a directory.
    """
    return upath.protocol in ("", "file") and upath.is_dir()


def _is_remote_dir(orig_data: str | Path | UPath, upath: UPath) -> bool:
    # Iterating over HTTP(S) directories is very difficult, let's just not do that.
    # See details in _read_parquet_into_table docstring.
    if upath.protocol in NO_ITERDIR_FILESYSTEMS:
        return False
    if str(orig_data).endswith("/"):
        return True
    return upath.is_dir()


def _read_remote_parquet_directory(
    dir_upath: UPath, filesystem, storage_options, columns: list[str] | None, **kwargs
) -> pa.Table:
    """Read files one-by-one with fsspec.open_parquet_file and concat the result"""
    tables = []
    for upath in dir_upath.iterdir():
        # Go recursively for filesystems which support file/directory identification with fsspec file
        # handlers. This would work for e.g. S3, but not for HTTP(S).
        if upath.is_dir():
            table = _read_remote_parquet_directory(upath, filesystem, storage_options, columns, **kwargs)
        else:
            with fsspec.parquet.open_parquet_file(
                upath.path,
                columns=columns,
                storage_options=storage_options,
                fs=filesystem,
                engine="pyarrow",
            ) as parquet_file:
                table = _read_table_with_partial_load_check(parquet_file, columns=columns, **kwargs)
        tables.append(table)
    return pa.concat_tables(tables)


def _get_storage_options(path_to_data: UPath):
    """Get storage options for fsspec.parquet.open_parquet_file.

    Parameters
    ----------
    path_to_data : UPath
        The data source

    Returns
    -------
    dict
        Storage options (or None)
    """
    if path_to_data.protocol not in ("", "file"):
        # Remote files of all types (s3, http)
        storage_options = path_to_data.storage_options or {}
        # For some cases, use smaller block size
        if path_to_data.protocol in FSSPEC_FILESYSTEMS:
            storage_options = {**storage_options, "block_size": FSSPEC_BLOCK_SIZE}
        return storage_options

    # Local files
    return None


def _transform_read_parquet_data_arg(data):
    """Transform `data` argument of read_parquet to pq.read_parquet's `source` and `filesystem`"""
    # Check if a list, run the function recursively and check that filesystems are all the same
    if isinstance(data, list):
        paths = []
        first_fs = None
        for i, d in enumerate(data):
            path, fs = _transform_read_parquet_data_arg(d)
            paths.append(path)
            if i == 0:
                first_fs = fs
            elif fs != first_fs:
                raise ValueError(
                    f"All filesystems in the list should be the same, first fs: {first_fs}, {i + 1} fs: {fs}"
                )
        return paths, first_fs
    # Check if a file-like object
    if hasattr(data, "read"):
        return data, None
    # Check if `data` is a UPath and use it
    if isinstance(data, UPath):
        return data.path, data.fs
    # Check if `data` is a Path (Path is a superclass for UPath, so this order of checks)
    if isinstance(data, Path):
        return data, None
    # It should be a string now
    if not isinstance(data, str):
        raise TypeError("data must be a file-like object, Path, UPath, list, or str")

    # Try creating pyarrow-native filesystem assuming that `data` is a URI
    try:
        fs, path = pa.fs.FileSystem.from_uri(data)
    # If the convertion failed, continue
    except (TypeError, pa.ArrowInvalid):
        pass
    # If not, use pyarrow filesystem
    else:
        return path, fs

    # Otherwise, treat `data` as a URI or a local path
    upath = UPath(data)
    # If it is a local path, use pyarrow's filesystem
    if upath.protocol == "":
        return upath.path, None
    # Change the default UPath object to use a smaller block size in some cases
    if upath.protocol in FSSPEC_FILESYSTEMS:
        upath = UPath(upath, block_size=FSSPEC_BLOCK_SIZE)
    return upath.path, upath.fs


def from_pyarrow(
    table: pa.Table,
    reject_nesting: list[str] | str | None = None,
    autocast_list: bool = False,
) -> NestedFrame:
    """
    Load a pyarrow Table object into a NestedFrame.

    Parameters
    ----------
    table: pa.Table
        PyArrow Table object to load NestedFrame from
    reject_nesting: list or str, default=None
        Column(s) to reject from being cast to a nested dtype. By default,
        nested-pandas assumes that any struct column with all fields being lists
        is castable to a nested column. However, this assumption is invalid if
        the lists within the struct have mismatched lengths for any given item.
        Columns specified here will be read using the corresponding pandas.ArrowDtype.
    autocast_list: bool, default=False
        If True, automatically cast list columns to nested columns with NestedDType.

    Returns
    -------
    NestedFrame

    """

    if reject_nesting is None:
        reject_nesting = []
    elif isinstance(reject_nesting, str):
        reject_nesting = [reject_nesting]

    # Convert to NestedFrame
    # not zero-copy, but reduce memory pressure via the self_destruct kwarg
    # https://arrow.apache.org/docs/python/pandas.html#reducing-memory-use-in-table-to-pandas
    df = NestedFrame(table.to_pandas(types_mapper=pd.ArrowDtype, split_blocks=True, self_destruct=True))
    # Attempt to cast struct columns to NestedDTypes
    df = _cast_struct_cols_to_nested(df, reject_nesting)

    # If autocast_list is True, cast list columns to NestedDTypes
    if autocast_list:
        df = _cast_list_cols_to_nested(df)

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
                    "`reject_nesting` argument of the read_parquet function to skip the cast attempt:"
                    f" read_parquet(..., reject_nesting=['{col}'])"
                ) from err
    return df


def _cast_list_cols_to_nested(df):
    """cast list columns to nested dtype"""
    for col, dtype in df.dtypes.items():
        if pa.types.is_list(dtype.pyarrow_dtype):
            df[col] = pack_lists(df[[col]])
    return df
