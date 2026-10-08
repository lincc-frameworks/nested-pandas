# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import warnings
from pathlib import Path
from typing import cast

import fsspec.parquet
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs
import pyarrow.parquet as pq
from pyarrow.lib import ArrowInvalid
from upath import UPath

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


def _pyarrow_read_table(data, *, columns=None, filesystem=None, **kwargs):
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


def _is_local_path(upath: UPath) -> bool:
    """Returns True if the given path refers to a local file or directory."""
    return upath.protocol in ("", "file")


def _is_remote_dir(orig_data: str | Path | UPath, upath: UPath, is_dir: bool | None) -> bool:
    # Iterating over HTTP(S) directories is very difficult, let's just not do that.
    # See details in _read_parquet_into_table docstring.
    if upath.protocol in NO_ITERDIR_FILESYSTEMS:
        return False
    if is_dir is not None:
        return is_dir
    if str(orig_data).endswith("/"):
        return True
    if is_dir is None:
        return upath.is_dir()


def _columns_to_load(
    columns: list[str] | None,
    filters: pc.Expression | list[tuple[str, str, object]] | list[list[tuple[str, str, object]]] | None,
) -> list[str] | None:
    """Get a list of the columns to pass to fsspec.parquet.open_parquet_file

    It should be more than just "columns", because it may also include columns needed for filters.

    It returns either a list of columns to load, or None if original columns is None, or if filters are
    given as a PyArrow Expression object.
    """
    if columns is None or filters is None:
        return columns

    # There is no simple way to introspect PyArrow expression objects, so we just load all the columns
    if isinstance(filters, pc.Expression):
        warnings.warn(
            "All columns will be loaded when 'filters' is a PyArrow Expression and 'columns' is set. "
            "Use list-of-tuples filters to avoid this: [(col, op, value), ...]. "
            "See pyarrow.parquet.read_table docs for the format.",
            UserWarning,
            stacklevel=2,
        )
        return None

    if not isinstance(filters, list):
        raise ValueError(
            "filters must be an PyArrow Expression, list of tuples, or list of lists of tuples, or None; "
            f"got '{type(filters)}'"
        )

    # Convert list[tuple] to list[list[tuple]] if needed
    try:
        element = filters[0][0]
    except IndexError as e:
        raise ValueError(
            "filters format must be [(col, op, value), ...], or [[(col, op, value), ...], ...]"
        ) from e
    is_nested_list = isinstance(element, tuple)
    if not is_nested_list:
        filters = [cast(list[tuple[str, str, object]], filters)]

    columns_from_filters: list[str] = []
    for filter_ in filters:
        if not isinstance(filter_, list):
            raise ValueError(
                "filters format must be [(col, op, value), ...], or [[(col, op, value), ...], ...]"
            )
        try:
            columns_from_filters.extend(col for col, _, _ in filter_)
        except ValueError as e:
            raise ValueError(
                "filters format must be [(col, op, value), ...], or [[(col, op, value), ...], ...]"
            ) from e

    return sorted(set(columns_from_filters + columns))


def _read_remote_parquet_directory(
    dir_upath: UPath, filesystem, storage_options, columns: list[str] | None, **kwargs
) -> pa.Table:
    """List files recursively and read them with fsspec.parquet.open_parquet_files."""
    file_paths = filesystem.find(dir_upath.path, withdirs=False, detail=False)
    parquet_files = fsspec.parquet.open_parquet_files(
        file_paths,
        columns=_columns_to_load(columns, kwargs.get("filters")),
        storage_options=storage_options,
        fs=filesystem,
        engine="pyarrow",
    )
    tables = []
    for parquet_file in parquet_files:
        with parquet_file:
            tables.append(_pyarrow_read_table(parquet_file, columns=columns, **kwargs))
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
        # Local filesystem: never hand off to fsspec, just use the plain path.
        if data.protocol in ("", "file"):
            return data.path, None
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
