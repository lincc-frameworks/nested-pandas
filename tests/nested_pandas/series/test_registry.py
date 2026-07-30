import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas import (
    NestedDtype,
    NestedFrame,
    NestedSeries,
    register_series_class,
    unregister_series_class,
)
from nested_pandas.series.registry import get_series_class, wrap_series


class _IntSeries(pd.Series):
    """Series subclass used to test registration against ArrowDtype(int64)."""

    def double(self):
        return _IntSeries(self * 2)


@pytest.fixture
def int_series_registered():
    """Temporarily register _IntSeries for ArrowDtype(int64) columns."""
    register_series_class(pd.ArrowDtype, _IntSeries)
    yield
    unregister_series_class(pd.ArrowDtype)


def test_nested_dtype_is_preregistered():
    """NestedDtype -> NestedSeries is registered on import."""
    dtype = NestedDtype.from_columns({"a": pa.int64()})
    assert get_series_class(dtype) is NestedSeries


def test_getitem_returns_nested_series():
    """Nested column access still returns NestedSeries through the registry."""
    base = NestedFrame(data={"a": [1, 2, 3], "c": [[0, 1], [2, 3], [4, 5]]}, index=[0, 1, 2])
    ndf = base.nest_lists(columns=["c"], name="nested")
    assert type(ndf["nested"]) is NestedSeries
    # Backticked nested column names resolve to the same wrapped series
    assert type(ndf["`nested`"]) is NestedSeries


def test_getitem_returns_registered_class(int_series_registered):
    """Column access wraps columns of a registered dtype in the registered class."""
    ndf = NestedFrame({"a": pd.array([1, 2, 3], dtype=pd.ArrowDtype(pa.int64())), "b": [1.0, 2.0, 3.0]})
    assert type(ndf["a"]) is _IntSeries
    assert list(ndf["a"].double()) == [2, 4, 6]
    # Unregistered dtypes are unaffected
    assert type(ndf["b"]) is pd.Series


def test_unregister_series_class(int_series_registered):
    """After unregistering, column access returns a plain series again."""
    ndf = NestedFrame({"a": pd.array([1, 2, 3], dtype=pd.ArrowDtype(pa.int64()))})
    assert type(ndf["a"]) is _IntSeries
    unregister_series_class(pd.ArrowDtype)
    assert type(ndf["a"]) is pd.Series
    # Unregistering twice is a no-op
    unregister_series_class(pd.ArrowDtype)


def test_get_series_class_walks_mro(int_series_registered):
    """A dtype subclass resolves to the class registered for its parent."""

    class _SubArrowDtype(pd.ArrowDtype):
        pass

    assert get_series_class(_SubArrowDtype(pa.int64())) is _IntSeries


def test_wrap_series_no_registration():
    """wrap_series returns the input unchanged for unregistered dtypes."""
    series = pd.Series([1, 2, 3])
    assert wrap_series(series) is series


def test_wrap_series_already_wrapped(int_series_registered):
    """wrap_series does not re-wrap a series already of the registered class."""
    series = _IntSeries(pd.array([1, 2, 3], dtype=pd.ArrowDtype(pa.int64())))
    assert wrap_series(series) is series
