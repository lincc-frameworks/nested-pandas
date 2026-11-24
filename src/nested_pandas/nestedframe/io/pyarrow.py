# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import contextlib
import io
import math
import warnings
from pathlib import Path
from typing import cast

import fsspec.parquet
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs
import pyarrow.parquet as pq
from fsspec.utils import merge_offset_ranges
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


class _StructSubcolumnsUnsupportedError(Exception):
    """`_read_struct_subcolumns` can't serve this read; use the regular reader."""


_FOOTER_SAMPLE_BYTES = 256_000
_PREFETCH_MAX_GAP = 64_000
_PREFETCH_MAX_BLOCK = 32_000_000


def _normalize_dnf_filters(
    filters: list[tuple[str, str, object]] | list[list[tuple[str, str, object]]],
) -> list[list[tuple[str, str, object]]]:
    """Normalize pyarrow list-of-tuples filters to DNF (list of conjunctions)."""
    if not isinstance(filters, list):
        raise ValueError(
            "filters must be an PyArrow Expression, list of tuples, or list of lists of tuples, or None; "
            f"got '{type(filters)}'"
        )
    if not filters:
        return []
    first = filters[0]
    if isinstance(first, tuple):
        conjunctions = [cast(list[tuple[str, str, object]], filters)]
    elif isinstance(first, list):
        conjunctions = cast(list[list[tuple[str, str, object]]], filters)
    else:
        raise ValueError("filters format must be [(col, op, value), ...], or [[(col, op, value), ...], ...]")
    for conj in conjunctions:
        if not isinstance(conj, list):
            raise ValueError(
                "filters format must be [(col, op, value), ...], or [[(col, op, value), ...], ...]"
            )
        for term in conj:
            if not isinstance(term, tuple) or len(term) != 3:
                raise ValueError(
                    "filters format must be [(col, op, value), ...], or [[(col, op, value), ...], ...]"
                )
    return conjunctions


def _predicate_matches_row_group(
    rg_meta: pq.RowGroupMetaData, col_indices: dict[str, int], col: str, op: str, val: object
) -> bool:
    """Conservatively check whether any row in ``rg_meta`` can satisfy ``(col, op, val)``."""
    col_idx = col_indices.get(col)
    if col_idx is None:
        return True
    stats = rg_meta.column(col_idx).statistics
    if stats is None or val is None or (isinstance(val, float) and math.isnan(val)):
        return True
    if not stats.has_min_max:
        return True
    if stats.has_null_count and stats.null_count == rg_meta.num_rows:
        return False
    smin, smax = stats.min, stats.max
    try:
        if op in ("==", "="):
            return bool(smin <= val <= smax)
        if op == "!=":
            return not (smin == val and smax == val)
        if op == "<":
            return bool(smin < val)
        if op == "<=":
            return bool(smin <= val)
        if op == ">":
            return bool(smax > val)
        if op == ">=":
            return bool(smax >= val)
        if op == "in":
            if not isinstance(val, (list, tuple, set, frozenset)):
                return True
            return any(
                v is not None and not (isinstance(v, float) and math.isnan(v)) and smin <= v <= smax
                for v in val
            )
    except TypeError:
        return True
    return True


def _select_row_groups(metadata: pq.FileMetaData, dnf: list[list[tuple[str, str, object]]]) -> list[int]:
    """Return row-group indices that may satisfy ``dnf`` based on Parquet min/max statistics."""
    if not dnf or metadata.num_row_groups == 0:
        return list(range(metadata.num_row_groups))
    rg0 = metadata.row_group(0)
    col_indices = {rg0.column(i).path_in_schema: i for i in range(rg0.num_columns)}
    selected: list[int] = []
    for rg_idx in range(metadata.num_row_groups):
        rg = metadata.row_group(rg_idx)
        if any(
            all(_predicate_matches_row_group(rg, col_indices, col, op, val) for col, op, val in conj)
            for conj in dnf
        ):
            selected.append(rg_idx)
    return selected


def _is_remote_fsspec(filesystem) -> bool:
    """Return True when ``filesystem`` is a remote fsspec filesystem supporting ``cat_ranges``."""
    if not isinstance(filesystem, fsspec.AbstractFileSystem) or not hasattr(filesystem, "cat_ranges"):
        return False
    proto = getattr(filesystem, "protocol", None)
    if isinstance(proto, str):
        return proto not in ("", "file", "local")
    if isinstance(proto, (tuple, list)):
        return not any(p in ("", "file", "local") for p in proto)
    return True


def _validate_struct_subcolumns_schema(
    file_schema: pa.Schema,
    columns: list[str],
    filter_columns: list[str],
    schema: pa.Schema | None,
) -> None:
    """Validate that ``columns``, ``filter_columns``, and ``schema`` fit the leaf-only reader."""
    if schema is not None and not schema.equals(file_schema, check_metadata=False):
        raise _StructSubcolumnsUnsupportedError("schema differs from the file's schema")
    for col in columns:
        if col in file_schema.names:
            continue
        name, _, field = col.partition(".")
        if (
            "." in field
            or name not in file_schema.names
            or not pa.types.is_struct(struct := file_schema.field(name).type)
            or struct.get_field_index(field) < 0
        ):
            raise _StructSubcolumnsUnsupportedError(f"{col!r} is not a direct field of a struct column")
    for col in filter_columns:
        if col not in file_schema.names:
            raise _StructSubcolumnsUnsupportedError(f"filter column {col!r} is not a top-level column")


def _read_remote_coalesced_parquet(
    filesystem: fsspec.AbstractFileSystem,
    path: str,
    columns: list[str],
    read_columns: list[str],
    schema: pa.Schema | None,
    dnf: list[list[tuple[str, str, object]]] | None,
    filter_columns: list[str],
) -> tuple[pa.Table, pa.Schema]:
    """Fetch footer + exact leaf/row-group byte ranges via coalesced ``cat_ranges``."""
    file_size = int(filesystem.size(path))
    if file_size < 12:
        raise _StructSubcolumnsUnsupportedError(f"parquet file too small ({file_size} bytes)")
    footer_start = max(0, file_size - _FOOTER_SAMPLE_BYTES)
    footer = filesystem.cat_file(path, start=footer_start, end=file_size)
    footer_len = int.from_bytes(footer[-8:-4], "little")
    real_footer_start = file_size - (footer_len + 8)
    if real_footer_start < 0:
        raise _StructSubcolumnsUnsupportedError("invalid parquet footer length")
    if real_footer_start < footer_start:
        prefix = filesystem.cat_file(path, start=real_footer_start, end=footer_start)
        footer = prefix + footer
        footer_start = real_footer_start

    meta_pf = pq.ParquetFile(io.BytesIO(footer))
    file_schema = meta_pf.schema_arrow
    _validate_struct_subcolumns_schema(file_schema, columns, filter_columns, schema)

    md = meta_pf.metadata
    row_groups = _select_row_groups(md, dnf) if dnf else list(range(md.num_row_groups))
    blocks: dict[tuple[int, int], bytes] = {(footer_start, file_size): footer}
    if not row_groups:
        if footer_start > 0:
            blocks[(0, 4)] = b"PAR1"
        buffered = fsspec.parquet.AlreadyBufferedFile(
            fs=None, path=path, mode="rb", cache_type="parts", cache_options={"data": blocks}, size=file_size
        )
        return pq.ParquetFile(buffered).read_row_groups([], columns=read_columns), file_schema

    starts: list[int] = []
    ends: list[int] = []
    for r in row_groups:
        rg = md.row_group(r)
        for c in range(rg.num_columns):
            col_meta = rg.column(c)
            leaf_path = col_meta.path_in_schema
            if any(leaf_path == req or leaf_path.startswith(req + ".") for req in read_columns):
                d = col_meta.to_dict()
                offset0 = min(
                    v
                    for v in (
                        d.get("dictionary_page_offset"),
                        d.get("data_page_offset"),
                        d.get("index_page_offset"),
                    )
                    if v is not None
                )
                if offset0 < footer_start:
                    starts.append(offset0)
                    ends.append(min(offset0 + d["total_compressed_size"], footer_start))

    if starts:
        m_paths, m_starts, m_ends = merge_offset_ranges(
            [path] * len(starts),
            starts,
            ends,
            max_gap=_PREFETCH_MAX_GAP,
            max_block=_PREFETCH_MAX_BLOCK,
            sort=True,
        )
        buffers = filesystem.cat_ranges(m_paths, m_starts, m_ends, on_error="raise")
        for s, e, data in zip(m_starts, m_ends, buffers, strict=True):
            blocks[(s, e)] = data

    if not any(s == 0 and e >= 4 for s, e in blocks):
        blocks[(0, 4)] = b"PAR1"

    buffered = fsspec.parquet.AlreadyBufferedFile(
        fs=None, path=path, mode="rb", cache_type="parts", cache_options={"data": blocks}, size=file_size
    )
    return pq.ParquetFile(buffered).read_row_groups(row_groups, columns=read_columns), file_schema


def _read_struct_subcolumns(
    path: str, columns: list[str], *, filesystem=None, schema: pa.Schema | None = None, **kwargs
) -> pa.Table:
    """Read struct sub-columns (or filtered remote files), fetching only needed leaves/row groups.

    `pq.read_table` (and so `fsspec.parquet`'s prefetch) downloads every field of a struct
    when a single sub-column is requested, while `ParquetFile.read` fetches just the
    requested leaves. For remote fsspec filesystems, byte ranges for the requested leaves
    and matching row groups are coalesced and prefetched in one ``cat_ranges`` call.

    Raises
    ------
    _StructSubcolumnsUnsupportedError
        If the read needs anything `ParquetFile.read` can't do here, before reading any data.
    """
    filters = kwargs.pop("filters", None)
    # Partitioning only applies to directories; any other read_table option needs the regular reader.
    if unsupported := [key for key, value in kwargs.items() if key != "partitioning" and value is not None]:
        raise _StructSubcolumnsUnsupportedError(f"unsupported read_table options: {unsupported}")
    if isinstance(filters, pc.Expression):
        raise _StructSubcolumnsUnsupportedError("PyArrow Expression filters are not supported")
    dnf = _normalize_dnf_filters(filters) if filters is not None else None
    filter_expr = pq.filters_to_expression(filters) if dnf else None
    filter_columns = list(dict.fromkeys(col for conj in dnf for col, _, _ in conj)) if dnf else []
    read_columns = list(dict.fromkeys([*columns, *filter_columns]))

    if _is_remote_fsspec(filesystem):
        table, file_schema = _read_remote_coalesced_parquet(
            filesystem, path, columns, read_columns, schema, dnf, filter_columns
        )
    else:
        with contextlib.ExitStack() as stack:
            # fsspec: skip the block cache. pyarrow requests just the needed ranges, and readahead
            # blocks re-download them (5x slower on S3). pyarrow opens other sources itself.
            if isinstance(filesystem, fsspec.AbstractFileSystem):
                path, filesystem = stack.enter_context(filesystem.open(path, "rb", cache_type="none")), None
            parquet_file = stack.enter_context(pq.ParquetFile(path, filesystem=filesystem))
            file_schema = parquet_file.schema_arrow
            _validate_struct_subcolumns_schema(file_schema, columns, filter_columns, schema)
            if dnf:
                row_groups = _select_row_groups(parquet_file.metadata, dnf)
                table = parquet_file.read_row_groups(row_groups, columns=read_columns)
            else:
                table = parquet_file.read(columns=read_columns)

    if filter_expr is not None and table.num_rows > 0:
        table = table.filter(filter_expr)
    # Match pq.read_table, which returns a sub-column as a top-level column named after its leaf.
    arrays, names = [], []
    for col in columns:
        if col in table.column_names:
            arrays.append(table[col])
            names.append(col)
        else:
            name, field = col.split(".")
            arrays.append(pc.struct_field(table[name], field))
            names.append(field)
    metadata = schema.metadata if schema is not None else file_schema.metadata
    return pa.Table.from_arrays(arrays, names=names, metadata=metadata)


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
