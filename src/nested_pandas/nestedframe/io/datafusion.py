# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs
from upath import UPath

from .pyarrow import _is_local_path, _validate_structs_from_schema


@lru_cache(maxsize=128)
def _build_http_store(base_url: str):
    from datafusion.object_store import Http

    return Http(base_url)


@lru_cache(maxsize=128)
def _build_s3_store(bucket: str, region: str | None, key: str, secret: str, endpoint: str | None):
    from datafusion.object_store import AmazonS3

    return AmazonS3(
        bucket_name=bucket,
        region=region,
        access_key_id=key,
        secret_access_key=secret,
        endpoint=endpoint,
    )


def _datafusion_object_store(path: UPath):
    options = dict(path.storage_options)

    # Only https: datafusion-python doesn't support http yet
    if path.protocol == "https":
        if options:
            raise ValueError(
                f"The 'datafusion' engine cannot pass these HTTPS storage options on to its "
                f"object store: {sorted(options)}; use engine='pyarrow' to read with them."
            )
        base_url = f"{path.drive}/"
        return base_url, _build_http_store(base_url)

    if path.protocol == "s3":
        key = options.pop("key", None)
        secret = options.pop("secret", None)
        endpoint = options.pop("endpoint_url", None)
        client_kwargs = options.pop("client_kwargs", None) or {}
        region = client_kwargs.get("region_name")
        if key is None or secret is None:
            raise ValueError(
                "The 'datafusion' engine needs both 'key' and 'secret' in the S3 path's "
                "storage_options: it cannot read credentials from the environment, nor "
                "make an anonymous request; use engine='pyarrow' for either."
            )
        if leftover := sorted(options) + sorted(set(client_kwargs) - {"region_name"}):
            raise ValueError(
                f"The 'datafusion' engine cannot pass these S3 storage options on to its "
                f"object store: {leftover}; use engine='pyarrow' to read with them."
            )
        bucket = path.drive
        return f"s3://{bucket}/", _build_s3_store(bucket, region, key, secret, endpoint)

    raise ValueError(
        f"The 'datafusion' engine cannot build an object store for this '{path.protocol}' "
        "path, it reads local paths, HTTPS, and S3 with an explicit key and secret; "
        "use engine='pyarrow' for anything else."
    )


def _check_datafusion_support(
    data, *, columns=None, filesystem=None, filters=None, schema=None, use_threads=True, **kwargs
) -> None:
    try:
        import datafusion  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The 'datafusion' engine requires the 'datafusion' package, install it with "
            "`pip install 'datafusion>=54.1'`."
        ) from e

    if not isinstance(data, str | Path | UPath):
        raise ValueError(
            f"The 'datafusion' engine supports a single path only, got '{type(data).__name__}'; "
            "use engine='pyarrow' for file-like objects and lists of paths."
        )
    if not _is_local_path(path := UPath(data)):
        _datafusion_object_store(path)
    if filesystem is not None and not isinstance(filesystem, pa.fs.LocalFileSystem):
        raise ValueError(
            f"The '{type(filesystem).__name__}' filesystem is not supported by the 'datafusion' engine."
        )
    if isinstance(filters, pc.Expression):
        raise ValueError(
            "PyArrow Expression 'filters' are not supported by the 'datafusion' engine, "
            "use the list-of-tuples format instead: [(column, op, value), ...]."
        )
    if not use_threads:
        raise ValueError("'use_threads=False' is not supported by the 'datafusion' engine.")
    if kwargs:
        raise ValueError(f"These arguments are not supported by the 'datafusion' engine: {sorted(kwargs)}.")


def _datafusion_read_table(
    data, *, columns=None, filesystem=None, filters=None, schema=None, use_threads=True, **kwargs
):
    """Read local parquet file(s) into a pyarrow table with DataFusion.

    Parameters
    ----------
    data : str or Path or UPath
        Path to a local parquet file or to a directory of parquet files.
    columns : list[str], optional
        Columns to load, sub-columns of nested columns are given with a dot,
        e.g. "nested.flux". The output table has the columns in the given
        order, and a sub-column is named after the last component of its path,
        which is what `pyarrow.parquet.read_table` does.
    filesystem : pyarrow.fs.LocalFileSystem, optional
        Only a local filesystem is accepted, which is what datafusion would use
        anyway. Anything else raises.
    filters : list[tuple] or list[list[tuple]], optional
        Filters in pyarrow's DNF format, a list of tuples is a conjunction, a
        list of lists of tuples is a disjunction of conjunctions.
        `pyarrow.compute.Expression` is not supported.
    schema : pyarrow.Schema, optional
        Schema of the parquet file(s), inferred from the data if not given.
    use_threads : bool, default True
        Must be True, datafusion always runs on its own thread pool.
    **kwargs
        Not supported, must be empty.

    Returns
    -------
    pyarrow.Table
    """
    _check_datafusion_support(
        data,
        columns=columns,
        filesystem=filesystem,
        filters=filters,
        schema=schema,
        use_threads=use_threads,
        **kwargs,
    )

    context = _datafusion_session_context()
    path = UPath(data)
    if _is_local_path(path):
        # UPath.path drops the "file://" prefix, which DataFusion doesn't understand
        source = path.path
    else:
        # The check above already established that we can build one
        base_url, store = _datafusion_object_store(path)
        context.register_object_store(base_url, store)
        source = str(path)

    df = context.read_parquet(
        source,
        skip_metadata=False,
        schema=schema,
    )

    # Filter first: the filters may use columns we are not going to load.
    if filters is not None:
        df = df.filter(_datafusion_filters_to_expr(filters))

    if columns is not None:
        # DataFusion requires the projected names to be unique. Project
        # to positional names and put the pyarrow ones back afterward.
        try:
            df = df.select(
                *(_datafusion_column_expr(column).alias(f"__npd_{i}") for i, column in enumerate(columns))
            )
        except Exception as e:
            if any("." in column for column in columns):
                try:
                    _validate_structs_from_schema(data, columns=columns)
                except ValueError as validation_error:
                    raise validation_error from e
            raise

    # Partitions hold contiguous file ranges, so concatenating them keeps file order
    batches = chain.from_iterable(df.collect_partitioned())
    table = pa.Table.from_batches(batches, schema=df.schema())

    if columns is not None:
        table = table.rename_columns([column.split(".")[-1] for column in columns])

    return table


# DataFusion session settings for every read.
DATAFUSION_SESSION_SETTINGS = {
    # Pruning and filter pushdown, `pushdown_filters` is off by default
    "datafusion.execution.parquet.pushdown_filters": "true",
    "datafusion.execution.parquet.reorder_filters": "true",
    "datafusion.execution.parquet.enable_page_index": "true",
    "datafusion.execution.parquet.bloom_filter_on_read": "true",
    # Give `string`/`binary` like pyarrow, not the view types DataFusion prefers
    "datafusion.execution.parquet.schema_force_view_types": "false",
    # Default is 8192, which chunks the output table 15x more finely than pyarrow does
    "datafusion.execution.batch_size": "131072",
    # Keep the plan-time assignment of byte ranges to partitions, so the row order is stable
    "datafusion.execution.enable_file_stream_work_stealing": "false",
}


@lru_cache(maxsize=1)
def _datafusion_session_context():
    from datafusion import SessionConfig, SessionContext

    config = SessionConfig()
    for key, value in DATAFUSION_SESSION_SETTINGS.items():
        config = config.set(key, value)
    return SessionContext(config)


def _datafusion_column_expr(column: str):
    from datafusion import col as df_col

    name, *sub_fields = column.split(".")
    # Quote the name: DataFusion lower-cases unquoted identifiers and treats a dot
    # in them as a qualifier separator.
    expr = df_col(f'"{name}"')
    for sub_field in sub_fields:
        expr = expr[sub_field]
    return expr


def _datafusion_filters_to_expr(filters):
    from datafusion import functions as df_functions
    from datafusion import literal as df_literal

    format_error = ValueError(
        "filters format must be [(column, op, value), ...], or [[(column, op, value), ...], ...]"
    )

    if not isinstance(filters, list):
        raise ValueError(
            "filters must be a PyArrow Expression, list of tuples, or list of lists of tuples, or None; "
            f"got '{type(filters)}'"
        )
    try:
        element = filters[0][0]
    except (IndexError, KeyError, TypeError) as e:
        raise format_error from e
    # Convert a single conjunction, list[tuple], into list[list[tuple]]
    if not isinstance(element, tuple):
        filters = [cast(list[tuple[str, str, object]], filters)]

    def literal(value):
        # DataFusion's literal() doesn't know numpy scalars, dates, and the like
        try:
            return df_literal(value)
        except Exception:
            return df_literal(pa.scalar(value))

    disjunction = None
    for conjunct in filters:
        if not isinstance(conjunct, list) or len(conjunct) == 0:
            raise format_error
        conjunction = None
        for filter_ in conjunct:
            try:
                column, op, value = filter_
            except (TypeError, ValueError) as e:
                raise format_error from e
            expr = _datafusion_column_expr(column)
            op = op.lower()
            if op in ("in", "not in"):
                expr = df_functions.in_list(expr, [literal(v) for v in value], negated=op == "not in")
            else:
                try:
                    build_expr = _DATAFUSION_FILTER_OPS[op]
                except KeyError as e:
                    raise ValueError(f"Unsupported filter operator: '{op}'") from e
                expr = build_expr(expr, literal(value))
            conjunction = expr if conjunction is None else conjunction & expr
        disjunction = conjunction if disjunction is None else disjunction | conjunction
    return disjunction


# Comparison operators of pyarrow's DNF filter format, "in" and "not in" are special-cased
_DATAFUSION_FILTER_OPS = {
    "=": lambda expr, value: expr == value,
    "==": lambda expr, value: expr == value,
    "!=": lambda expr, value: expr != value,
    "<": lambda expr, value: expr < value,
    "<=": lambda expr, value: expr <= value,
    ">": lambda expr, value: expr > value,
    ">=": lambda expr, value: expr >= value,
}
