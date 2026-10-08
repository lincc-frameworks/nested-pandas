# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

from pathlib import Path
from typing import Literal

import fsspec.parquet
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from upath import UPath

from ...series.ext_array import NestedExtensionArray
from ...series.packer import pack_lists
from ...series.utils import is_pa_type_a_list, table_to_struct_array
from ...tensors.ext_array import TensorExtensionArray
from ..core import NestedFrame
from .datafusion import _datafusion_read_table
from .pyarrow import (
    _columns_to_load,
    _get_storage_options,
    _is_local_path,
    _is_remote_dir,
    _pyarrow_read_table,
    _read_remote_parquet_directory,
    _transform_read_parquet_data_arg,
)


def read_parquet(
    data: str | UPath | bytes,
    columns: list[str] | None = None,
    reject_nesting: list[str] | str | None = None,
    autocast_list: bool = False,
    is_dir: bool | None = None,
    use_pandas_metadata: bool = True,
    engine: Literal["pyarrow", "datafusion"] = "pyarrow",
    **kwargs,
) -> NestedFrame:
    """Load a parquet object from a file path into a NestedFrame.

    As a specialization of the ``pandas.read_parquet`` function, this
    function loads the data via existing ``pyarrow`` or
    ``fsspec.parquet`` methods, and then converts the data to a
    NestedFrame. Struct-of-list columns become nested columns and
    ``fixed_shape_tensor`` columns become tensor columns (``TensorDtype``).

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
    is_dir: bool, None, default=None
        If True, the pointer represents a pixel directory; if False, the pointer
        represents a file. In both cases there is no need to check the pointer's
        content type. If `is_dir` is None (default), this method will resort to
        `upath.is_dir()` to identify the type of pointer. This argument is ignored
        for HTTP, as inferring the type for HTTP is particularly expensive because
        it requires downloading the contents of the pointer in its entirety.
    use_pandas_metadata: bool, default=True
        If True (default), apply the pandas metadata stored in the Parquet
        file's schema when constructing the NestedFrame (e.g. restoring the
        index and column dtypes). This matches the default behavior of
        pd.read_parquet. Set to False to ignore the metadata.
    engine: "pyarrow" or "datafusion", default="pyarrow"
        Which library to use for the parquet read.

        - "pyarrow" (default) uses `fsspec.parquet` (remote paths only) and
          `pyarrow.parquet.read_table`. It reads a nested column whole even
          when a single sub-column is asked for, and it applies `filters`
          after reading the row groups they select, so it is not the best
          choice for partial loads or selective filters.
        - "datafusion" may be very fast when a few rows and/or a few nested
          subcolumns are selected. It reads a single local, HTTPS, or
          S3 path, and raises for anything else. Needs the optional
          `datafusion` package.
    kwargs: dict
        Keyword arguments passed to `pyarrow.parquet.read_table`, of which
        the "datafusion" engine accepts `filters` and `filesystem` (a local
        one), plus a `schema` of its own. `filters` must be in the
        list-of-tuples format there, a `pyarrow.compute.Expression` is not
        supported by the "datafusion" engine.

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

    if engine not in ("pyarrow", "datafusion"):
        raise ValueError(f"Invalid engine: '{engine}', must be 'pyarrow' or 'datafusion'")

    _validate_filter_columns(kwargs.get("filters"))

    # Type convergence for reject_nesting
    if reject_nesting is None:
        reject_nesting = []
    elif isinstance(reject_nesting, str):
        reject_nesting = [reject_nesting]

    table = _read_parquet_into_table(data, columns=columns, is_dir=is_dir, engine=engine, **kwargs)

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
                if not is_pa_type_a_list(table.schema[i].type):
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

    return from_pyarrow(
        table,
        reject_nesting=reject_nesting,
        autocast_list=autocast_list,
        use_pandas_metadata=use_pandas_metadata,
    )


def _read_parquet_into_table(
    data: str | UPath | bytes,
    *,
    columns: list[str] | None,
    is_dir: bool | None = None,
    engine: str,
    **kwargs,
) -> pa.Table:
    """Reads parquet file(s) from path and returns a pyarrow table.

    For single remote Parquet file paths, we want to use
    `fsspec.parquet.open_parquet_file`. Remote directories are handled
    separately via `_read_remote_parquet_directory`. For everything local
    (files and directories alike), as well as file-like objects and lists
    thereof, we want to defer to `pq.read_table`, which can read local
    directories directly.

    NOTE: local paths are routed straight to `pq.read_table` via `_is_local_path`,
    without ever checking whether they're a file or a directory, since
    `pq.read_table` handles both natively. We don't support HTTP
    "directories", because 1) calling .is_dir() may be very expensive, because
    it downloads content first, 2) because .iter_dir() is likely to return a
    lot of "junk" besides of the actual parquet files.
    """
    if isinstance(data, str | Path | UPath) and not _is_local_path(path_to_data := UPath(data)):
        if engine == "datafusion":
            return _datafusion_read_table(path_to_data, columns=columns, **kwargs)

        storage_options = _get_storage_options(path_to_data)
        filesystem = kwargs.get("filesystem")
        if not filesystem:
            _, filesystem = _transform_read_parquet_data_arg(path_to_data)
        # Will not detect HTTP(S) directories.
        if _is_remote_dir(data, path_to_data, is_dir=is_dir):
            return _read_remote_parquet_directory(
                path_to_data, filesystem, storage_options, columns, **kwargs
            )

        with fsspec.parquet.open_parquet_file(
            path_to_data.path,
            columns=_columns_to_load(columns, kwargs.get("filters")),
            storage_options=storage_options,
            fs=filesystem,
            engine="pyarrow",
        ) as parquet_file:
            return _read_table_with_partial_load_check(
                parquet_file, columns=columns, engine="pyarrow", **kwargs
            )

    # All other cases, including file-like objects, directories, and
    # even lists of the foregoing.

    # If `filesystem` is specified - use it, passing it as part of **kwargs
    if kwargs.get("filesystem") is not None:
        return _read_table_with_partial_load_check(data, columns=columns, engine=engine, **kwargs)

    # Otherwise convert with a special function
    data, filesystem = _transform_read_parquet_data_arg(data)
    return _read_table_with_partial_load_check(
        data, columns=columns, filesystem=filesystem, engine=engine, **kwargs
    )


def _validate_filter_columns(filters) -> None:
    if filters is None or isinstance(filters, pc.Expression) or not isinstance(filters, list):
        return
    # Both [(column, op, value), ...] and [[(column, op, value), ...], ...]
    for filter_ in filters:
        conjunct = [filter_] if isinstance(filter_, tuple) else filter_
        if not isinstance(conjunct, list | tuple):
            continue
        for term in conjunct:
            if not isinstance(term, tuple) or len(term) != 3:
                continue
            column = term[0]
            if isinstance(column, str) and "." in column:
                raise ValueError(
                    f"Cannot filter on '{column}': filtering on a sub-column of a nested "
                    "column is not supported. Filter on a top-level column instead, or "
                    "load the data and filter the NestedFrame afterwards."
                )


def _read_table_with_partial_load_check(data, *, columns=None, filesystem=None, engine: str, **kwargs):
    """Read a pyarrow table with partial load check for nested structures"""
    if engine == "pyarrow":
        return _pyarrow_read_table(data, columns=columns, filesystem=filesystem, **kwargs)
    elif engine == "datafusion":
        return _datafusion_read_table(data, columns=columns, filesystem=filesystem, **kwargs)
    else:
        raise ValueError(f"Invalid engine: {engine}")


def from_pyarrow(
    table: pa.Table,
    reject_nesting: list[str] | str | None = None,
    autocast_list: bool = False,
    use_pandas_metadata: bool = True,
) -> NestedFrame:
    """
    Load a pyarrow Table object into a NestedFrame.

    Struct-of-list columns become nested columns and ``fixed_shape_tensor``
    columns become tensor columns (``TensorDtype``); everything else gets
    the corresponding ``pandas.ArrowDtype``.

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
    use_pandas_metadata: bool, default=True
        If True (default), apply the pandas metadata stored in the Parquet
        file's schema when constructing the NestedFrame (e.g. restoring the
        index and column dtypes). This matches the default behavior of
        pd.read_parquet. Set to False to ignore the metadata.

    Returns
    -------
    NestedFrame

    Examples
    --------
    >>> import nested_pandas as npd
    >>> import pyarrow as pa
    >>> table = pa.table({
    ...     "obj_id": [1, 2, 3],
    ...     "nested": pa.array([
    ...         [{"flux": 0.5, "time": 1}],
    ...         [{"flux": 1.2, "time": 2}, {"flux": 0.8, "time": 3}],
    ...         [{"flux": 2.0, "time": 4}],
    ...     ])
    ... })
    >>> npd.from_pyarrow(table)
       obj_id                              nested
    0       1              [{flux: 0.5, time: 1}]
    1       2  [{flux: 1.2, time: 2}; …] (2 rows)
    2       3              [{flux: 2.0, time: 4}]

    """

    if reject_nesting is None:
        reject_nesting = []
    elif isinstance(reject_nesting, str):
        reject_nesting = [reject_nesting]

    # Convert to a NestedFrame. With types_mapper=pd.ArrowDtype every column is
    # backed by the table's Arrow buffers, so this is zero-copy and there is no
    # need for the self_destruct memory optimization (which only helps the
    # NumPy-conversion path).
    df = NestedFrame(
        table.to_pandas(
            types_mapper=pd.ArrowDtype,
            split_blocks=True,
            ignore_metadata=not use_pandas_metadata,
        )
    )
    # Replace struct columns with NestedExtensionArrays and fixed-shape tensor
    # columns with TensorExtensionArrays, both built from the table.
    df = _cast_cols_to_extension_arrays(df, reject_nesting, table)

    # If autocast_list is True, cast list columns to NestedDTypes
    if autocast_list:
        df = _cast_list_cols_to_nested(df)

    return df


def _cast_cols_to_extension_arrays(
    df: NestedFrame, reject_nesting: list[str], table: pa.Table
) -> NestedFrame:
    """Replace struct and fixed-shape tensor columns of ``df`` with nested-pandas' extension arrays.

    Struct columns become nested columns and ``pa.FixedShapeTensorType``
    columns become tensor columns, both built straight from the pyarrow
    ``table`` rather than from the ``df`` columns produced by
    ``Table.to_pandas``. Converting a struct column that holds a
    ``null``-typed (all-null) field via ``types_mapper=pd.ArrowDtype``
    corrupts it (https://github.com/apache/arrow/issues/44881).
    """
    for field in table.schema:
        if isinstance(field.type, pa.FixedShapeTensorType):
            df[field.name] = _tensor_column(field.name, table.column(field.name))
        elif field.name not in reject_nesting and NestedExtensionArray.is_input_pa_type_supported(field.type):
            df[field.name] = _nested_column(field.name, table.column(field.name))
    return df


def _tensor_column(name: str, column: pa.ChunkedArray) -> TensorExtensionArray:
    """Build a tensor column from a ``fixed_shape_tensor`` chunked array, explaining a rejected permutation.

    Permutations other than the identity are not supported by TensorDtype; the error says how to rebuild
    the column without one.
    """
    try:
        return TensorExtensionArray(column)
    except NotImplementedError as err:
        # TensorDtype rejects non-trivial permutations, add the column and a way out
        raise NotImplementedError(
            f"Column '{name}' has a fixed_shape_tensor type with a non-trivial permutation "
            f"{list(column.type.permutation)}, which nested-pandas does not support: {column.type}. "
            "Please open an issue on the nested-pandas github if you need this feature. As a workaround, "
            "read the data with pyarrow, convert the column to a C-ordered numpy array, which applies "
            "the permutation, rebuild the column without one and pass the table to from_pyarrow():\n"
            f"    ndarray = table.column('{name}').combine_chunks().to_numpy_ndarray()\n"
            "    tensors = pa.FixedShapeTensorArray.from_numpy_ndarray("
            "np.ascontiguousarray(ndarray))\n"
            f"    table = table.set_column(table.schema.get_field_index('{name}'), '{name}', tensors)"
        ) from err


def _nested_column(name: str, column: pa.ChunkedArray) -> NestedExtensionArray:
    """Build a nested column from a struct-of-lists chunked array, explaining a failed cast."""
    try:
        return NestedExtensionArray(column)
    except ValueError as err:
        # If cast fails, the struct likely does not fit nested-pandas criteria for a valid nested column
        raise ValueError(
            f"Column '{name}' is a Struct, but an attempt to cast it to a NestedDType failed. "
            "This is likely due to the struct not meeting the requirements for a nested column "
            "(all fields should be equal length). To proceed, you may add the column to the "
            "`reject_nesting` argument of the read_parquet function to skip the cast attempt:"
            f" read_parquet(..., reject_nesting=['{name}'])"
        ) from err


def _cast_list_cols_to_nested(df):
    """cast list columns to nested dtype"""
    for col, dtype in df.dtypes.items():
        if is_pa_type_a_list(dtype.pyarrow_dtype):
            df[col] = pack_lists(df[[col]])
    return df
