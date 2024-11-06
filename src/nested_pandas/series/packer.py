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
from nested_pandas.series.ext_array import NestedExtensionArray

__all__ = ["pack", "pack_flat", "pack_lists", "pack_seq"]


N_ROWS_INFER_DTYPE = 1000


def pack(
    obj,
    name: str | None = None,
    *,
    index=None,
    on: None | str | list[str] = None,
    dtype: NestedDtype | pd.ArrowDtype | pa.DataType | None = None,
) -> pd.Series:
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
    pd.Series
        Output series.
    """
    if isinstance(obj, pd.DataFrame):
        nested = pack_flat(obj, name=name, on=on)
        if index is not None:
            nested.index = index
        return nested
    return pack_seq(obj, name=name, index=index, dtype=dtype)


def pack_flat(df: pd.DataFrame, name: str | None = None, *, on: None | str | list[str] = None) -> pd.Series:
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
        Name of the pd.Series.
    on : str or list of str, optional
        Column name(s) to join on. If None, the df's index is used.

    Returns
    -------
    pd.Series
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
    return pack_sorted_df_into_struct(sorted_flat, name=name)


def pack_seq(
    sequence: Sequence,
    name: str | None = None,
    *,
    index: object = None,
    dtype: NestedDtype | pd.ArrowDtype | pa.DataType | None = None,
) -> pd.Series:
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
    pd.Series
        Output series.
    """
    if isinstance(sequence, pd.Series):
        if index is None:
            index = sequence.index
        if name is None:
            name = sequence.name

    ext_array = NestedExtensionArray.from_sequence(sequence, dtype=dtype)
    series = pd.Series(ext_array, index=index, name=name, copy=False)
    return series


def pack_sorted_df_into_struct(df: pd.DataFrame, name: str | None = None) -> pd.Series:
    """Make a structure of lists representation of a "flat" dataframe.

    Input dataframe must be sorted and all the columns must have pyarrow dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, with repeated indexes. It must be sorted and
        all the columns must have pyarrow dtypes.

    name : str, optional
        Name of the pd.Series.

    Returns
    -------
    pd.Series
        Output series, with unique indexes.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("The index of the input dataframe must be sorted")

    packed_df = view_sorted_df_as_list_arrays(df)
    # No need to validate the dataframe, the length of the nested arrays is forced to be the same by
    # the view_sorted_df_as_list_arrays function.
    return pack_lists(packed_df, name=name, validate=False)


def pack_lists(df: pd.DataFrame, name: str | None = None, *, validate: bool = True) -> pd.Series:
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
        Name of the pd.Series.
    validate : bool, default True
        Whether to validate the input dataframe.

    Returns
    -------
    pd.Series
        Output series, with unique indexes.

    See Also
    --------
    nested_pandas.series.accessor.NestSeriesAccessor : The accessor for the output series.
    nested_pandas.series.dtype.NestedDtype : The dtype of the output series.
    nested_pandas.series.packer.pack_flat : Pack a "flat" dataframe with repeated indexes.
    """
    struct_array = pa.StructArray.from_arrays(
        [df[column] for column in df.columns],
        names=df.columns,
    )
    ext_array = NestedExtensionArray(struct_array, validate=validate)
    return pd.Series(
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

    series_ = {
        column: view_sorted_series_as_list_array(df[column], offset_array, unique_index)
        for column in df.columns
    }

    df = pd.DataFrame(series_)

    return df


def view_sorted_series_as_list_array(
    series: pd.Series, offset: np.ndarray | None = None, unique_index: np.ndarray | None = None
) -> pd.Series:
    """Make a nested array representation of a "flat" series.

    Parameters
    ----------
    series : pd.Series
        Input series, with repeated indexes. It must be sorted by its index.

    offset: np.ndarray or None, optional
        Pre-calculated offsets of the input series index.
    unique_index: np.ndarray or None, optional
        Pre-calculated unique index of the input series. If given it must be
        equal to `series.index.unique()` and `series.index.values[offset[:-1]]`.

    Returns
    -------
    pd.Series
        Output series, with unique indexes. It is a view over the input series,
        so it would mute the input series if modified.
    """
    if not series.index.is_monotonic_increasing:
        raise ValueError("The index of the input series must be sorted")

    if offset is None:
        offset = calculate_sorted_index_offsets(series.index)
    if unique_index is None:
        unique_index = series.index[offset[:-1]]

    list_array = pa.ListArray.from_arrays(
        offset,
        pa.array(series, from_pandas=True),
    )
    return pd.Series(
        list_array,
        dtype=pd.ArrowDtype(list_array.type),
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
        Output array of offsets, one element more than the number of unique
        index values.
    """
    if not index.is_monotonic_increasing:
        raise ValueError("The index must be sorted")

    # pd.Index.duplicated returns False for the first occurance and True for all others.
    # So our offsets would be indexes of these False values with the array length in the end.
    offset_but_last = np.nonzero(~index.duplicated(keep="first"))[0]
    offset = np.append(offset_but_last, len(index))

    # Arrow uses int32 for offsets
    offset = offset.astype(np.int32)

    return offset
