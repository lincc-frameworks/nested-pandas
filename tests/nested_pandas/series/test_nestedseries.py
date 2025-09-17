import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas import NestedDtype
from nested_pandas.series.nestedseries import NestedSeries


def test_init_nestedseries():
    """Test initialization of NestedSeries."""
    series = pd.Series(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
            (np.array([5, 6]), np.array([0, 1])),
        ],
        index=[0, 1, 2],
        dtype=NestedDtype(pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.int64()))])),
    )
    nested_series = NestedSeries(series)

    assert isinstance(nested_series, NestedSeries)
    assert nested_series.dtype == NestedDtype(
        pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.int64()))])
    )
    assert nested_series.index.equals(pd.Index([0, 1, 2]))


def test_nestedonly_decorator():
    """Test nested_only decorator."""

    series = NestedSeries([1, 2, 3, 4, 5])

    # Check nested only properties for decorator functionality
    for prop in ["columns", "flat_length", "list_lengths"]:
        with pytest.raises(TypeError, match=f"'{prop}' can only be used with a NestedDtype"):
            getattr(series, prop)

    # Check nested only methods for decorator functionality
    for func in ["to_flat", "to_lists"]:
        with pytest.raises(TypeError, match=f"'{func}' can only be used with a NestedDtype"):
            getattr(series, func)()


def test_nestedseries_columns():
    """Test columns property of NestedSeries."""
    series = NestedSeries(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
        ],
        index=[0, 1],
        dtype=NestedDtype(pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.int64()))])),
    )

    assert series.columns == ["a", "b"]


def test_nestedseries_flat_length():
    """Test flat_length property of NestedSeries."""
    series = NestedSeries(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
        ],
        index=[0, 1],
        dtype=NestedDtype(pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.int64()))])),
    )

    assert series.flat_length == 4


def test_nestedseries_list_lengths():
    """Test list_lengths property of NestedSeries."""
    series = NestedSeries(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
        ],
        index=[0, 1],
        dtype=NestedDtype(pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.int64()))])),
    )

    assert list(series.list_lengths) == [2, 2]


def test_nestedseries_getitem_single_column():
    """Test getitem for a single column in NestedSeries."""
    series = NestedSeries(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
        ],
        index=[0, 1],
        dtype=NestedDtype(pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.int64()))])),
    )

    result = series["a"]
    expected = pd.Series([1, 2, 3, 4], index=[0, 0, 1, 1], dtype=pd.ArrowDtype(pa.int64()), name="a")
    pd.testing.assert_series_equal(result, expected)


def test_nestedseries_getitem_multiple_columns():
    """Test getitem for multiple columns in NestedSeries."""
    series = NestedSeries(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
        ],
        index=[0, 1],
        dtype=NestedDtype(pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.int64()))])),
    )

    result = series[["a", "b"]]
    expected = series  # Full selection returns the original structure
    pd.testing.assert_series_equal(result, expected)


def test_nestedseries_getitem_masking():
    """Test getitem with boolean masking in NestedSeries."""
    series = NestedSeries(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
        ],
        index=[0, 1],
        dtype=NestedDtype(pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.int64()))])),
        name="nested",
    )

    mask = pd.Series([True, False, False, True], index=[0, 0, 1, 1], dtype=bool, name="mask")
    result = series[mask]
    assert result.flat_length == 2


def test_nestedseries_getitem_index():
    """Test getitem with ordinary index selection in NestedSeries."""
    series = NestedSeries(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
        ],
        index=[0, 1],
        dtype=NestedDtype(pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.int64()))])),
    )

    result = series[0]
    expected = pd.DataFrame({"a": [1, 2], "b": [0, 1]}, index=[0, 1])
    pd.testing.assert_frame_equal(result, expected)


def test_nestedseries_getitem_non_nested_dtype():
    """Test setitem with a non-nested dtype."""
    series = NestedSeries(
        data=[1, 2, 3],
        index=[0, 1, 2],
        dtype=pd.ArrowDtype(pa.int64()),
    )

    assert series[0] == 1


def test_nestedseries_setitem_non_nested_dtype():
    """Test setitem with a non-nested dtype."""
    series = NestedSeries(
        data=[1, 2, 3],
        index=[0, 1, 2],
        dtype=pd.ArrowDtype(pa.int64()),
    )

    series[0] = 10
    assert series[0] == 10


def test_nestedseries_setitem_single_column():
    """Test setitem for a single column in NestedSeries."""
    series = NestedSeries(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
        ],
        index=[0, 1],
        dtype=NestedDtype(pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.int64()))])),
    )

    series["a"] = pd.Series([10, 20, 30, 40], index=[0, 0, 1, 1])
    expected = pd.Series([10, 20, 30, 40], index=[0, 0, 1, 1], dtype=pd.ArrowDtype(pa.int64()), name="a")
    pd.testing.assert_series_equal(series["a"], expected)

    series["a"] = 5
    expected = pd.Series([5, 5, 5, 5], index=[0, 0, 1, 1], dtype=pd.ArrowDtype(pa.int64()), name="a")
    pd.testing.assert_series_equal(series["a"], expected)


def test_nestedseries_explode():
    """Test explode method of NestedSeries."""
    series = NestedSeries(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
        ],
        index=[0, 1],
        dtype=NestedDtype(pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.int64()))])),
    )

    flat_df = series.explode()
    assert isinstance(flat_df, pd.DataFrame)
    assert list(flat_df.columns) == ["a", "b"]
    assert flat_df.shape == (4, 2)


def test_nestedseries_to_lists():
    """Test to_lists method of NestedSeries."""
    series = NestedSeries(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
        ],
        index=[0, 1],
        dtype=NestedDtype(pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.int64()))])),
    )

    lists = series.to_lists()
    assert len(lists) == 2
    assert lists["a"][0] == [1, 2]
    assert lists["a"][1] == [3, 4]
