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
    for prop in ["fields", "flat_length", "list_lengths"]:
        with pytest.raises(TypeError, match=f"'{prop}' can only be used with a NestedDtype"):
            getattr(series, prop)

    # Check nested only methods for decorator functionality
    for func in ["to_flat", "to_lists"]:
        with pytest.raises(TypeError, match=f"'{func}' can only be used with a NestedDtype"):
            getattr(series, func)()


def test_nestedseries_fields():
    """Test fields property of NestedSeries."""
    series = NestedSeries(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
        ],
        index=[0, 1],
        dtype=NestedDtype(pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.int64()))])),
    )

    assert series.fields == ["a", "b"]


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


def test_nestedseries_to_flat():
    """Test to_flat method of NestedSeries."""
    series = NestedSeries(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
        ],
        index=[0, 1],
        dtype=NestedDtype(pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.int64()))])),
    )

    flat_df = series.to_flat()
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
