import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from tensor_test_utils import TENSOR_SHAPE, tensor, tensor_stack

from nested_pandas import NestedFrame, TensorDtype, TensorSeries
from nested_pandas.tensors import TensorExtensionArray


def test_init_tensorseries():
    """Test initialization of a TensorSeries from a Series and from an extension array."""
    array = TensorExtensionArray.from_stack(tensor_stack([0, 1, 2]))
    series = TensorSeries(pd.Series(array, index=[10, 11, 12], name="t"))

    assert isinstance(series, TensorSeries)
    assert series.dtype == TensorDtype(pa.fixed_shape_tensor(pa.float64(), list(TENSOR_SHAPE)))
    assert series.index.equals(pd.Index([10, 11, 12]))
    assert series.name == "t"

    from_array = TensorSeries(array, index=[10, 11, 12], name="t")
    assert from_array.equals(series)


def test_from_stack():
    """Test from_stack builds the series with the given index, name and dtype."""
    stack = tensor_stack([0, 1, 2])
    series = TensorSeries.from_stack(stack, index=["a", "b", "c"], name="t")

    assert isinstance(series, TensorSeries)
    assert series.tensor_shape == TENSOR_SHAPE
    assert series.index.equals(pd.Index(["a", "b", "c"]))
    assert series.name == "t"
    np.testing.assert_array_equal(series["b"], tensor(1))

    # Default index and explicit dtype
    typed = TensorSeries.from_stack(stack, dtype="tensor[float, (2, 3)]")
    assert typed.index.equals(pd.RangeIndex(3))
    assert typed.dtype.value_type == pa.float32()

    with pytest.raises(ValueError, match="at least two dimensions"):
        TensorSeries.from_stack(np.arange(3.0))


def test_getitem_returns_numpy():
    """Test that scalar access returns a read-only numpy view and positional selection keeps the dtype."""
    series = TensorSeries.from_stack(tensor_stack([0, 1, 2]), index=[10, 11, 12])

    value = series[11]
    assert isinstance(value, np.ndarray)
    assert value.shape == TENSOR_SHAPE
    assert not value.flags.writeable
    np.testing.assert_array_equal(value, tensor(1))
    np.testing.assert_array_equal(series.iloc[2], tensor(2))

    assert isinstance(series.iloc[1:].dtype, TensorDtype)
    assert len(series.iloc[1:]) == 2


def test_constructor_keeps_subclass_for_tensor_results():
    """Test that operations keeping the tensor dtype return a TensorSeries, and others a plain Series."""
    series = TensorSeries.from_stack(tensor_stack([0, 1, 2]), index=[10, 11, 12], name="t")

    for result in [
        series.iloc[1:],
        series.loc[[10, 12]],
        series[series.index > 10],
        series.take([2, 0]),
        series.copy(),
        series.sort_index(ascending=False),
        series.head(2),
        series.reset_index(drop=True),
        series.rename("u"),
        pd.concat([series, series]),
    ]:
        assert type(result) is TensorSeries
        assert isinstance(result.dtype, TensorDtype)
    assert series.iloc[1:].name == "t"
    np.testing.assert_array_equal(series.take([2, 0]).to_stack(), tensor_stack([2, 0]))

    # Results of other dtypes are plain Series, not a TensorSeries that cannot use its methods
    for result in [series.isna(), series.map(np.sum), series.index.to_series()]:
        assert type(result) is pd.Series


def test_tensor_only_decorator():
    """Test that tensor-only members raise a TypeError for other dtypes."""
    series = TensorSeries([1, 2, 3])

    with pytest.raises(TypeError, match="'tensor_shape' can only be used with a TensorDtype"):
        _ = series.tensor_shape

    for func in ["to_stack", "to_arrow_fixed_lists"]:
        with pytest.raises(TypeError, match=f"'{func}' can only be used with a TensorDtype"):
            getattr(series, func)()


def test_to_stack():
    """Test to_stack gives the (n, *shape) array, as a view without missing values and filled with them."""
    stack = tensor_stack([0, 1, 2])
    series = TensorSeries.from_stack(stack)

    result = series.to_stack()
    assert result.shape == (3, *TENSOR_SHAPE)
    np.testing.assert_array_equal(result, stack)
    assert np.shares_memory(result, stack)

    with_missing = TensorSeries(TensorExtensionArray.from_sequence([tensor(0), None, tensor(2)]))
    filled = with_missing.to_stack(na_value=-1.0)
    np.testing.assert_array_equal(filled[1], np.full(TENSOR_SHAPE, -1.0))
    np.testing.assert_array_equal(filled[[0, 2]], tensor_stack([0, 2]))
    assert np.isnan(with_missing.to_stack()[1]).all()


def test_to_arrow_fixed_lists():
    """Test to_arrow_fixed_lists gives an arrow fixed_size_list series with the same index and name."""
    series = TensorSeries.from_stack(tensor_stack([0, 1]), index=[5, 6], name="t")
    lists = series.to_arrow_fixed_lists()

    assert lists.dtype == pd.ArrowDtype(pa.list_(pa.float64(), int(np.prod(TENSOR_SHAPE))))
    assert lists.index.equals(series.index)
    assert lists.name == "t"
    np.testing.assert_array_equal(lists[6], tensor(1).reshape(-1))


def test_repr_html():
    """Test the HTML repr renders tensor cells like a one-column NestedFrame, and is absent for other
    dtypes."""
    pytest.importorskip("matplotlib")

    series = TensorSeries.from_stack(np.arange(8.0).reshape(2, 2, 2), index=[10, 11], name="t")
    html = series._repr_html_()
    assert html is not None
    assert html.count("<pre") == 2  # small tensors show their values
    assert ">11</th>" in html  # the index is kept
    assert "2 rows x 1 columns" in html
    assert TensorSeries([1, 2, 3])._repr_html_() is None


def test_nestedframe_getitem_returns_tensorseries():
    """Test that NestedFrame column access returns a TensorSeries for tensor columns only."""
    nf = NestedFrame(
        {
            "a": [1, 2, 3],
            "t": TensorExtensionArray.from_stack(tensor_stack([0, 1, 2])),
            "with.dot": TensorExtensionArray.from_stack(tensor_stack([3, 4, 5])),
        },
        index=[10, 11, 12],
    )

    series = nf["t"]
    assert isinstance(series, TensorSeries)
    assert series.name == "t"
    assert series.index.equals(nf.index)
    np.testing.assert_array_equal(series.to_stack(), tensor_stack([0, 1, 2]))

    # Column names with dots and backticks go through the same wrapping
    assert isinstance(nf["with.dot"], TensorSeries)
    assert isinstance(nf["`with.dot`"], TensorSeries)

    # Other columns are unchanged
    assert type(nf["a"]) is pd.Series
