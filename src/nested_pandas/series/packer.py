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

__all__ = ["pack_flat", "pack_lists", "pack_dfs"]


N_ROWS_INFER_DTYPE = 1000


def pack_flat_into_df(df: pd.DataFrame, name=None) -> pd.DataFrame:
    """Pack a "flat" dataframe into a "nested" dataframe.

    For the input dataframe with repeated indexes, make a pandas.DataFrame,
    where each original column is replaced by a column of lists, and,
    optionally, a "structure" column is added, containing a structure of
    lists with the original columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, with repeated indexes.

    name : str, optional
        Name of the structure column. The default is None, which means no
        structure column is added.

    Returns
    -------
    pd.DataFrame
        Output dataframe.
    """
    # TODO: we can optimize name=None case a bit
    struct_series = pack_flat(df, name=name)
    packed_df = struct_series.struct.explode()
    if name is not None:
        packed_df[name] = struct_series
    return packed_df


def pack_flat(df: pd.DataFrame, name: str | None = None) -> pd.Series:
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

    # TODO: think about the case when the data is pre-sorted and we don't need a data copy.
    flat = df.sort_index(kind="stable")
    return pack_sorted_df_into_struct(flat, name=name)


def pack_dfs(dfs: Sequence[pd.DataFrame], index: object = None, name: str | None = None) -> pd.Series:
    """Pack a sequence of "flat" dataframes into a "nested" series.

    Parameters
    ----------
    dfs : Sequence[pd.DataFrame]
        Input sequence of dataframes.
    index : pd.Index, optional
        Index of the output series.
    name : str, optional
        Name of the output series.

    Returns
    -------
    pd.Series
        Output series.
    """
    if isinstance(dfs, pd.Series) and index is None:
        index = dfs.index

    first_df = dfs.iloc[0] if hasattr(dfs, "iloc") else dfs[0]

    field_types = {
        column: pa.array(first_df[column].iloc[:N_ROWS_INFER_DTYPE]).type for column in first_df.columns
    }
    dtype = NestedDtype.from_fields(field_types)
    dummy_value: dict[str, list] = {column: [] for column in first_df.columns}
    series = pd.Series([dummy_value] * len(dfs), dtype=dtype, index=index, name=name)
    series[:] = dfs
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
    offset_array = calculate_sorted_index_offsets(df.index)
    unique_index = df.index.values[offset_array[:-1]]

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
    if offset is None:
        offset = calculate_sorted_index_offsets(series.index)
    if unique_index is None:
        unique_index = series.index.values[offset[:-1]]

    list_array = pa.ListArray.from_arrays(
        offset,
        pa.array(series),
    )
    return pd.Series(
        list_array,
        dtype=pd.ArrowDtype(list_array.type),
        index=unique_index,
        copy=False,
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
    # TODO: implement multi-index support
    index_diff = np.diff(index.values, prepend=index.values[0] - 1, append=index.values[-1] + 1)

    if np.any(index_diff < 0):
        raise ValueError("Table index must be strictly sorted.")

    offset = np.nonzero(index_diff)[0]

    return offset
