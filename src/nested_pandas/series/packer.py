"""Module for converting between "flat" and "list" and "nested" representations

TODO: mask support
TODO: multi-index support
"""

# "|" for python 3.9
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from nested_pandas.series.dtype import NestedDtype
from nested_pandas.series.ext_array import DEFAULT_CHUNK_SIZE, DEFAULT_MIN_CHUNK_SIZE, NestedExtensionArray
from nested_pandas.series.nestedseries import NestedSeries
from nested_pandas.series.utils import chunk_sizes_are_fragmented, compute_chunk_boundaries
from nested_pandas.series.utils import rechunk as rechunk_array

__all__ = ["pack", "pack_flat", "pack_lists", "pack_seq"]


N_ROWS_INFER_DTYPE = 1000


def pack(
    obj,
    name: str | None = None,
    *,
    index=None,
    on: None | str | list[str] = None,
    dtype: NestedDtype | pd.ArrowDtype | pa.DataType | None = None,
) -> NestedSeries:
    """Pack a "flat" dataframe or a sequence of dataframes into a "nested" series.

    Parameters
    ----------
    obj : pd.DataFrame or Sequence of
        Input dataframe, with repeated indexes, or a sequence of dataframes or missed values.
    name : str, optional
        Name of the output series.
    index : convertable to pd.Index, optional
        Index of the output series. If obj is a pd.DataFrame, it is always nested by the original index,
        and this value is used to override the index after the nesting.
    on: str or list of str, optional
        Column name(s) to join on. If None, the index is used.
    dtype : dtype or None
        NestedDtype of the output series, or other type to derive from. If None,
        the dtype is inferred from the first non-missing dataframe.

    Returns
    -------
    NestedSeries
        Output series.
    """
    if isinstance(obj, pd.DataFrame):
        nested = pack_flat(obj, name=name, on=on)
        if index is not None:
            nested.index = index
        return nested
    return pack_seq(obj, name=name, index=index, dtype=dtype)


def pack_flat(
    df: pd.DataFrame, name: str | None = None, *, on: None | str | list[str] = None
) -> NestedSeries:
    """Make a structure of lists representation of a "flat" dataframe.

    For the input dataframe with repeated indexes, make a pandas.Series,
    where each original column is replaced by a structure of lists.
    The dtype of the column is `nested_pandas.NestedDtype` with
    the corresponding pyarrow type. The index of the output series is
    the unique index of the input dataframe. The Series has `.nest` accessor,
    see `nested_pandas.series.accessor.NestSeriesAccessor` for details.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, with repeated indexes.
    name : str, optional
        Name of the NestedSeries.
    on : str or list of str, optional
        Column name(s) to join on. If None, the df's index is used.

    Returns
    -------
    NestedSeries
        Output series, with unique indexes.

    See Also
    --------
    nested_pandas.series.accessor.NestedSeriesAccessor : .nest accessor for the output series.
    nested_pandas.series.dtype.NestedDtype : The dtype of the output series.
    nested_pandas.series.packer.pack_lists : Pack a dataframe of nested arrays.
    """
    if on is not None:
        df = df.set_index(on)
    # pandas knows when index is pre-sorted, so it would do nothing if it is already sorted
    sorted_flat = df.sort_index(kind="stable")
    try:
        return pack_sorted_df_into_struct(sorted_flat, name=name)
    except ValueError:
        # Check if the error is due to NaN values and raise a more informative message
        if any(sorted_flat.index.get_level_values(i).hasnans for i in range(sorted_flat.index.nlevels)):
            if on is None:
                raise ValueError(
                    "The index contains NaN values. "
                    "NaN values are not supported because they cannot be used for grouping rows. "
                    "Please remove or fill NaN values before packing."
                ) from None
            cols = [on] if isinstance(on, str) else list(on)
            raise ValueError(
                f"Column(s) {cols} contain NaN values. "
                "NaN values are not supported because they cannot be used for grouping rows. "
                "Please remove or fill NaN values before packing."
            ) from None
        raise


def pack_seq(
    sequence: Sequence,
    name: str | None = None,
    *,
    index: object = None,
    dtype: NestedDtype | pd.ArrowDtype | pa.DataType | None = None,
) -> NestedSeries:
    """Pack a sequence of "flat" dataframes into a "nested" series.

    Parameters
    ----------
    sequence : Sequence of pd.DataFrame or None or pd.NA or convertible to pa.StructScalar
        Input sequence of dataframes or missed values.
    name : str, optional
        Name of the output series.
    index : pd.Index, optional
        Index of the output series.
    dtype : dtype or None
        NestedDtype of the output series, or other type to derive from. If None,
        the dtype is inferred from the first non-missing dataframe.

    Returns
    -------
    NestedSeries
        Output series.
    """
    if isinstance(sequence, pd.Series):  # generalized check for pandas series
        if index is None:
            index = sequence.index
        if name is None:
            name = sequence.name

    ext_array = NestedExtensionArray.from_sequence(sequence, dtype=dtype)
    series = NestedSeries(ext_array, index=index, name=name, copy=False)
    return series


def pack_sorted_df_into_struct(df: pd.DataFrame, name: str | None = None) -> NestedSeries:
    """Make a structure of lists representation of a "flat" dataframe.

    Input dataframe must be sorted and all the columns must have pyarrow dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, with repeated indexes. It must be sorted and
        all the columns must have pyarrow dtypes.

    name : str, optional
        Name of the NestedSeries.

    Returns
    -------
    NestedSeries
        Output series, with unique indexes.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("The index of the input dataframe must be sorted")

    packed_df = view_sorted_df_as_list_arrays(df)
    # No need to validate the dataframe, the length of the nested arrays is forced to be the same by
    # the view_sorted_df_as_list_arrays function.
    return pack_lists(packed_df, name=name, validate=False)


def pack_lists(df: pd.DataFrame, name: str | None = None, *, validate: bool = True) -> NestedSeries:
    """Make a series of arrow structures from a dataframe with nested arrays.

    For the input dataframe with repeated indexes, make a pandas.Series,
    where each original column is replaced by a structure of lists.
    The dtype of the column is `nested_pandas.NestedDtype` with the corresponding
    pyarrow type. The index of the output series is the unique index of the
    input dataframe. The Series has `.nest` accessor, see
    `nested_pandas.series.accessor.NestSeriesAccessor` for details.

    For every row, all the nested array (aka pyarrow list) lengths must be
    the same.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, with pyarrow list-arrays.
    name : str, optional
        Name of the NestedSeries.
    validate : bool, default True
        Whether to validate the input dataframe.

    Returns
    -------
    NestedSeries
        Output series, with unique indexes.

    See Also
    --------
    nested_pandas.series.accessor.NestSeriesAccessor : The accessor for the output series.
    nested_pandas.series.dtype.NestedDtype : The dtype of the output series.
    nested_pandas.series.packer.pack_flat : Pack a "flat" dataframe with repeated indexes.
    """
    # When series is converted to pa.array it may be both Array and ChunkedArray
    # We convert it to chunked for the sake of consistency
    pa_arrays_maybe_chunked = {column: pa.array(df[column]) for column in df.columns}
    pa_chunked_arrays = {
        column: arr if isinstance(arr, pa.ChunkedArray) else pa.chunked_array([arr])
        for column, arr in pa_arrays_maybe_chunked.items()
    }

    first_col = next(iter(pa_chunked_arrays.values()))
    first_sizes = [len(c) for c in first_col.chunks]
    all_aligned = all([len(c) for c in arr.chunks] == first_sizes for arr in pa_chunked_arrays.values())
    struct_type = pa.struct([pa.field(col, arr.type) for col, arr in pa_chunked_arrays.items()])

    if all_aligned and not chunk_sizes_are_fragmented(first_sizes, DEFAULT_MIN_CHUNK_SIZE):
        # All columns share the same non-fragmented chunk layout — build struct with no data copying.
        struct_chunks = [
            pa.StructArray.from_arrays(
                [arr.chunk(i) for arr in pa_chunked_arrays.values()],
                names=list(pa_chunked_arrays.keys()),
            )
            for i in range(first_col.num_chunks)
        ]
        struct_array = pa.chunked_array(struct_chunks, type=struct_type)
        ext_array = NestedExtensionArray(struct_array, validate=validate)
    elif all_aligned:
        # Aligned but fragmented — building struct from aligned columns is cheap (no copying),
        # then rechunk the struct to merge small chunks.
        struct_chunks = [
            pa.StructArray.from_arrays(
                [arr.chunk(i) for arr in pa_chunked_arrays.values()],
                names=list(pa_chunked_arrays.keys()),
            )
            for i in range(first_col.num_chunks)
        ]
        struct_array = pa.chunked_array(struct_chunks, type=struct_type)
        ext_array = NestedExtensionArray(struct_array, validate=validate).rechunk()
    else:
        # Misaligned chunks — rechunk all columns to DEFAULT_CHUNK_SIZE boundaries first,
        # then build struct. This avoids building a misaligned struct and a second rechunk.
        n = len(df)
        full, rem = divmod(n, DEFAULT_CHUNK_SIZE)
        target_sizes = [DEFAULT_CHUNK_SIZE] * full + ([rem] if rem else [])
        aligned = {col: rechunk_array(arr, target_sizes) for col, arr in pa_chunked_arrays.items()}
        struct_chunks = [
            pa.StructArray.from_arrays(
                [aligned[col].chunk(i) for col in aligned],
                names=list(aligned.keys()),
            )
            for i in range(len(target_sizes))
        ]
        struct_array = pa.chunked_array(struct_chunks)
        ext_array = NestedExtensionArray(struct_array, validate=validate)
    return NestedSeries(
        ext_array,
        index=df.index,
        copy=False,
        name=name,
    )


def view_sorted_df_as_list_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """Make a nested array representation of a "flat" dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, with repeated indexes. It must be sorted by its index.

    Returns
    -------
    pd.DataFrame
        Output dataframe, with unique indexes. It is a view over the input
        dataframe, so it would mute the input dataframe if modified.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("The index of the input dataframe must be sorted")

    offset_array = calculate_sorted_index_offsets(df.index)
    unique_index = df.index[offset_array[:-1]]
    boundaries = compute_chunk_boundaries(np.diff(offset_array), DEFAULT_CHUNK_SIZE)

    series_ = {
        column: view_sorted_series_as_list_array(
            df[column], offset=offset_array, boundaries=boundaries, unique_index=unique_index
        )
        for column in df.columns
    }

    df = pd.DataFrame(series_)

    return df


def view_sorted_series_as_list_array(
    series: NestedSeries,
    *,
    offset: np.ndarray | None = None,
    boundaries: list[int] | None = None,
    unique_index: np.ndarray | None = None,
) -> NestedSeries:
    """Make a nested array representation of a "flat" series.

    Parameters
    ----------
    series : NestedSeries
        Input series, with repeated indexes. It must be sorted by its index.

    offset: np.ndarray or None, optional
        Pre-calculated int64 offsets of the input series index.
    boundaries: list[int] or None, optional
        Pre-calculated chunk boundaries from ``compute_chunk_boundaries``.
        If given, ``offset`` must also be provided. Passing this avoids
        recomputing boundaries when processing multiple columns that share
        the same offset array.
    unique_index: np.ndarray or None, optional
        Pre-calculated unique index of the input series. If given it must be
        equal to `series.index.unique()` and `series.index.values[offset[:-1]]`.

    Returns
    -------
    NestedSeries
        Output series, with unique indexes. It is a view over the input series,
        so it would mute the input series if modified.
    """
    if not series.index.is_monotonic_increasing:
        raise ValueError("The index of the input series must be sorted")

    if offset is None:
        offset = calculate_sorted_index_offsets(series.index)
    if unique_index is None:
        unique_index = series.index[offset[:-1]]
    if boundaries is None:
        boundaries = compute_chunk_boundaries(np.diff(offset), DEFAULT_CHUNK_SIZE)

    # Input series may be represented by pyarrow.ChunkedArray, in this case pa.array(series) would fail
    # with "TypeError: Cannot convert a 'ChunkedArray' to a 'ListArray'".
    # https://github.com/lincc-frameworks/nested-pandas/issues/189
    flat_array = pa.array(series, from_pandas=True)
    if isinstance(flat_array, pa.ChunkedArray):
        flat_array = flat_array.combine_chunks()

    # Split into chunks so each ListArray fits within int32 offset limits and
    # aligns with DEFAULT_CHUNK_SIZE. Since all columns in a DataFrame share
    # the same offset array, they produce identical boundaries and are aligned,
    # which lets pack_lists take the zero-copy aligned path.
    list_chunks = [
        pa.ListArray.from_arrays(
            offsets=(offset[b_start : b_end + 1] - offset[b_start]).astype(np.int32),
            values=flat_array[int(offset[b_start]) : int(offset[b_end])],
        )
        for b_start, b_end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    chunked_list_array = pa.chunked_array(list_chunks, type=pa.list_(flat_array.type))

    return NestedSeries(
        chunked_list_array,
        dtype=pd.ArrowDtype(chunked_list_array.type),
        index=unique_index,
        copy=False,
        name=series.name,
    )


def calculate_sorted_index_offsets(index: pd.Index) -> np.ndarray:
    """Calculate the offsets of the pre-sorted index values.

    Parameters
    ----------
    index : pd.Index
        Input index, must be sorted.

    Returns
    -------
    np.ndarray
        Output array of int64 offsets, one element more than the number of
        unique index values. int64 is used to avoid overflow for large flat
        arrays; callers that build ``pa.ListArray`` chunks must cast each
        per-chunk slice to int32 after normalising to zero.
    """
    if not index.is_monotonic_increasing:
        raise ValueError("The index must be sorted")

    # pd.Index.duplicated returns False for the first occurance and True for all others.
    # So our offsets would be indexes of these False values with the array length in the end.
    offset_but_last = np.nonzero(~index.duplicated(keep="first"))[0]
    offset = np.append(offset_but_last, len(index))

    return offset.astype(np.int64)
