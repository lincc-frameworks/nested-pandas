import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas import NestedDtype
from nested_pandas.series import packer
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal


def offsets_reused(nested_series):
    """Check if the offset buffers are reused for all columns of the nested series"""
    lists_df = nested_series.nest.to_lists()
    first_offset_buffers = None
    for column in lists_df.columns:
        offset_buffers = pa.array(lists_df[column]).offsets.buffers()
        if first_offset_buffers is None:
            first_offset_buffers = offset_buffers
        assert offset_buffers == first_offset_buffers


def test_pack_with_flat_df():
    """Test pack(pd.DataFrame)."""
    df = pd.DataFrame(
        data={
            "a": [1, 2, 3, 4],
            "b": [0, 1, 0, 1],
        },
        index=pd.MultiIndex.from_arrays(([1, 1, 1, 1], [1, 2, 1, 2])),
    )
    series = packer.pack(df, name="series")

    desired = pd.Series(
        data=[
            (np.array([1, 3]), np.array([0, 0])),
            (np.array([2, 4]), np.array([1, 1])),
        ],
        index=pd.MultiIndex.from_arrays(([1, 1], [1, 2])),
        dtype=NestedDtype.from_fields(dict(a=pa.int64(), b=pa.int64())),
        name="series",
    )
    offsets_reused(series)
    assert_series_equal(series, desired)


def test_pack_with_flat_df_and_index():
    """Test pack(pd.DataFrame)."""
    df = pd.DataFrame(
        data={
            "a": [1, 2, 3, 4],
            "b": [0, 1, 0, 1],
        },
        index=[1, 2, 1, 2],
    )
    series = packer.pack(df, name="series", index=[101, 102])

    desired = pd.Series(
        data=[
            (np.array([1, 3]), np.array([0, 0])),
            (np.array([2, 4]), np.array([1, 1])),
        ],
        index=[101, 102],
        dtype=NestedDtype.from_fields(dict(a=pa.int64(), b=pa.int64())),
        name="series",
    )
    offsets_reused(series)
    assert_series_equal(series, desired)


def test_pack_with_series_of_dfs():
    """Test pack(pd.Series([pd.DataFrame(), ...]))."""
    input_series = pd.Series(
        [
            pd.DataFrame(
                {
                    "a": [1, 2],
                    "b": [0, 1],
                },
            ),
            pd.DataFrame(
                {
                    "a": [3, 4],
                    "b": [0, 1],
                },
            ),
        ],
        index=[1, 2],
        name="series",
    )
    series = packer.pack(input_series, name="nested")

    desired = pd.Series(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
        ],
        index=[1, 2],
        name="nested",
        dtype=NestedDtype.from_fields(dict(a=pa.int64(), b=pa.int64())),
    )
    offsets_reused(series)
    assert_series_equal(series, desired)


def test_pack_flat():
    """Test pack_flat()."""
    df = pd.DataFrame(
        data={
            "a": [7, 8, 9, 1, 2, 3, 4, 5, 6],
            "b": [0, 1, 0, 0, 1, 0, 1, 0, 1],
        },
        index=[4, 4, 4, 1, 1, 2, 2, 3, 3],
    )
    actual = packer.pack_flat(df)

    desired = pd.Series(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
            (np.array([5, 6]), np.array([0, 1])),
            (np.array([7, 8, 9]), np.array([0, 1, 0])),
        ],
        index=[1, 2, 3, 4],
        dtype=NestedDtype.from_fields(dict(a=pa.int64(), b=pa.int64())),
    )
    offsets_reused(actual)
    assert_series_equal(actual, desired)


def test_pack_sorted_df_into_struct():
    """Test pack_sorted_df_into_struct()."""
    df = pd.DataFrame(
        data={
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "b": [0, 1, 0, 1, 0, 1, 0, 1, 0],
        },
        index=[1, 1, 2, 2, 3, 3, 4, 4, 4],
    )
    actual = packer.pack_sorted_df_into_struct(df)

    desired = pd.Series(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
            (np.array([5, 6]), np.array([0, 1])),
            (np.array([7, 8, 9]), np.array([0, 1, 0])),
        ],
        index=[1, 2, 3, 4],
        dtype=NestedDtype.from_fields(dict(a=pa.int64(), b=pa.int64())),
    )
    offsets_reused(actual)
    assert_series_equal(actual, desired)


def test_pack_sorted_df_into_struct_raises_when_not_sorted():
    """Test pack_sorted_df_into_struct() raises when not sorted."""
    df = pd.DataFrame(
        data={
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "b": [0, 1, 0, 1, 0, 1, 0, 1, 0],
        },
        index=[1, 2, 1, 2, 3, 3, 4, 4, 4],
    )
    with pytest.raises(ValueError):
        packer.pack_sorted_df_into_struct(df)


def test_pack_lists():
    """Test pack_lists()."""
    packed_df = pd.DataFrame(
        data={
            "a": [
                np.array([1, 2]),
                np.array([3, 4]),
                np.array([5, 6]),
                np.array([7, 8, 9]),
            ],
            "b": [
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([0, 1, 0]),
            ],
        },
        index=[1, 2, 3, 4],
        dtype=pd.ArrowDtype(pa.list_(pa.int64())),
    )
    series = packer.pack_lists(packed_df)
    offsets_reused(series)

    for field_name in packed_df.columns:
        assert_series_equal(series.nest.get_list_series(field_name), packed_df[field_name])


def test_pack_seq_with_dfs_and_index():
    """Test pack_seq()."""
    dfs = [
        pd.DataFrame(
            data={
                "a": [1, 2],
                "b": [0, 1],
            },
            index=[100, 100],
        ),
        pd.DataFrame(
            data={
                "a": [3, 4],
                "b": [0, 1],
            },
            index=[101, 101],
        ),
        pd.DataFrame(
            data={
                "a": [5, 6],
                "b": [0, 1],
            },
            index=[102, 102],
        ),
        pd.DataFrame(
            data={
                "a": [7, 8, 9],
                "b": [0, 1, 0],
            },
            index=[103, 103, 103],
        ),
    ]
    series = packer.pack_seq(dfs, index=[100, 101, 102, 103])

    desired = pd.Series(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
            (np.array([5, 6]), np.array([0, 1])),
            (np.array([7, 8, 9]), np.array([0, 1, 0])),
        ],
        index=[100, 101, 102, 103],
        dtype=NestedDtype.from_fields(dict(a=pa.int64(), b=pa.int64())),
    )
    offsets_reused(series)
    assert_series_equal(series, desired)


def test_pack_seq_with_different_elements_and_index():
    """Test pack_seq() with different elements and index"""
    seq = [
        pd.DataFrame(
            data={
                "a": [1, 2],
                "b": [0, 1],
            },
        ),
        None,
        {"a": [3, 4], "b": [-1, 0]},
        pd.NA,
    ]
    series = packer.pack_seq(seq, index=[100, 101, 102, 103])

    desired = pd.Series(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            None,
            (np.array([3, 4]), np.array([-1, 0])),
            pd.NA,
        ],
        index=[100, 101, 102, 103],
        dtype=NestedDtype.from_fields(dict(a=pa.int64(), b=pa.int64())),
    )
    offsets_reused(series)
    assert_series_equal(series, desired)


def test_pack_seq_with_series_of_dfs():
    """Test pack_seq(pd.Series([pd.DataFrame(), ...]))."""
    input_series = pd.Series(
        [
            pd.DataFrame(
                {
                    "a": [1, 2],
                    "b": [0, 1],
                },
            ),
            pd.DataFrame(
                {
                    "a": [3, 4],
                    "b": [0, 1],
                },
            ),
            pd.DataFrame(
                {
                    "a": [5, 6],
                    "b": [0, 1],
                },
            ),
        ],
        index=[100, 101, 102],
        name="series",
    )
    series = packer.pack_seq(input_series)

    desired = pd.Series(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
            (np.array([5, 6]), np.array([0, 1])),
        ],
        index=[100, 101, 102],
        dtype=NestedDtype.from_fields(dict(a=pa.int64(), b=pa.int64())),
        name="series",
    )
    offsets_reused(series)
    assert_series_equal(series, desired)


def test_view_sorted_df_as_list_arrays():
    """Test view_sorted_df_as_list_arrays()."""
    flat_df = pd.DataFrame(
        data={
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "b": [0, 1, 0, 1, 0, 1, 0, 1, 0],
        },
        index=[1, 1, 2, 2, 3, 3, 4, 4, 4],
    )
    nested_df = packer.view_sorted_df_as_list_arrays(flat_df)

    assert_array_equal(nested_df.index, [1, 2, 3, 4])

    desired_nested = pd.DataFrame(
        data={
            "a": [
                np.array([1, 2]),
                np.array([3, 4]),
                np.array([5, 6]),
                np.array([7, 8, 9]),
            ],
            "b": [
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([0, 1, 0]),
            ],
        },
        index=[1, 2, 3, 4],
        dtype=pd.ArrowDtype(pa.list_(pa.int64())),
    )
    assert_frame_equal(nested_df, desired_nested)


def test_view_sorted_df_as_list_arrays_raises_when_not_sorted():
    """Test view_sorted_df_as_list_arrays() raises when not sorted."""
    flat_df = pd.DataFrame(
        data={
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "b": [0, 1, 0, 1, 0, 1, 0, 1, 0],
        },
        index=[1, 2, 1, 2, 3, 3, 4, 4, 4],
    )
    with pytest.raises(ValueError):
        packer.view_sorted_df_as_list_arrays(flat_df)


def test_view_sorted_series_as_list_array():
    """Test view_sorted_series_as_list_array()."""
    series = pd.Series(
        data=[1, 2, 3, 4, 5, None, 7, 8, 9],
        dtype=pd.Int64Dtype(),
        index=[1, 1, 2, 2, 3, 3, 4, 4, 4],
        name="my_series",
    )
    nested = packer.view_sorted_series_as_list_array(series)

    assert_array_equal(nested.index, [1, 2, 3, 4])

    desired_nested = pd.Series(
        data=[
            np.array([1, 2]),
            np.array([3, 4]),
            np.array([5, None]),
            np.array([7, 8, 9]),
        ],
        index=[1, 2, 3, 4],
        dtype=pd.ArrowDtype(pa.list_(pa.int64())),
        name="my_series",
    )
    assert_series_equal(nested, desired_nested)


def test_view_sorted_series_as_list_array_raises_when_not_sorted():
    """Test view_sorted_series_as_list_array() raises when not sorted."""
    series = pd.Series(
        data=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        index=[1, 2, 1, 2, 3, 3, 4, 4, 4],
    )
    with pytest.raises(ValueError):
        packer.view_sorted_series_as_list_array(series)


@pytest.mark.parametrize(
    "index,offsets",
    [
        (pd.Index([1, 2, 3, 4]), np.array([0, 1, 2, 3, 4])),
        (pd.Index([1, 1, 2, 2, 3, 3, 4, 4, 4]), np.array([0, 2, 4, 6, 9])),
        (pd.Index([1, 1, 1, 1, 1, 1, 1, 1, 1]), np.array([0, 9])),
        (pd.Index([1, 2, 2, 2, 3, 3, 4]), np.array([0, 1, 4, 6, 7])),
        (
            pd.MultiIndex.from_product([[1, 2, 3], ["a", "a", "b", "b", "b"]]),
            np.array([0, 2, 5, 7, 10, 12, 15]),
        ),
        (
            pd.MultiIndex.from_arrays(
                (
                    [1, 1, 1, 1, 1, 1, 2, 2],
                    ["a", "a", "a", "a", "b", "b", "z", "z"],
                    [1, 2, 2, 2, 9, 9, 9, 9],
                ),
                names=["id1", "id2", "id3"],
            ),
            np.array([0, 1, 4, 6, 8]),
        ),
    ],
)
def test_calculate_sorted_index_offsets(index, offsets):
    """Test calculate_sorted_index_offsets()."""
    actual = packer.calculate_sorted_index_offsets(index)
    assert actual.dtype == np.int32
    assert_array_equal(actual, offsets)


def test_calculate_sorted_index_offsets_raises_when_not_sorted():
    """Test calculate_sorted_index_offsets() raises when not sorted."""
    index = pd.Index([1, 2, 1, 2, 3, 3, 4, 4, 4])
    with pytest.raises(ValueError):
        packer.calculate_sorted_index_offsets(index)
