import re

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


# Assignment of tensors (#539) #


def _series_for_setitem() -> TensorSeries:
    """A four-row series shared by the assignment tests."""
    return TensorSeries.from_stack(tensor_stack([0, 1, 2, 3]), index=[10, 11, 12, 13], name="t")


_SETTERS = [
    ("getitem-label", lambda s, v: s.__setitem__(11, v), [1]),
    ("loc-label", lambda s, v: s.loc.__setitem__(11, v), [1]),
    ("iloc-position", lambda s, v: s.iloc.__setitem__(1, v), [1]),
    ("getitem-list", lambda s, v: s.__setitem__([11, 12], v), [1, 2]),
    ("getitem-mask", lambda s, v: s.__setitem__(np.array([True, False, True, False]), v), [0, 2]),
    (
        "getitem-bool-series",
        lambda s, v: s.__setitem__(pd.Series([False, False, True, True], index=s.index), v),
        [2, 3],
    ),
    ("loc-slice", lambda s, v: s.loc.__setitem__(slice(11, 12), v), [1, 2]),
    ("loc-list", lambda s, v: s.loc.__setitem__([10, 13], v), [0, 3]),
    ("loc-mask", lambda s, v: s.loc.__setitem__(np.array([True, False, True, False]), v), [0, 2]),
    ("iloc-slice", lambda s, v: s.iloc.__setitem__(slice(1, 3), v), [1, 2]),
    ("iloc-list", lambda s, v: s.iloc.__setitem__([0, 3], v), [0, 3]),
]
"""Every assignment path, as (id, setter(series, value), positions it sets)."""

SETTERS = [pytest.param(setter, rows, id=name) for name, setter, rows in _SETTERS]
MULTI_ROW_SETTERS = [pytest.param(setter, rows, id=name) for name, setter, rows in _SETTERS if len(rows) > 1]
"""The assignment paths selecting several rows, where a sequence of tensors can be assigned."""


@pytest.mark.parametrize("setter,rows", SETTERS)
@pytest.mark.parametrize(
    "value,expected_tensor",
    [
        (tensor(100), tensor(100)),
        (7, np.full(TENSOR_SHAPE, 7.0)),
        (np.array(7.0), np.full(TENSOR_SHAPE, 7.0)),
    ],
    ids=["tensor", "number", "0d-array"],
)
def test_setitem_broadcasts_single_value(setter, rows, value, expected_tensor):
    """Test that a single tensor, or a number filling a tensor, is assigned to every selected row through
    [], .loc and .iloc, keeping the dtype and name."""
    series = _series_for_setitem()

    setter(series, value)

    assert isinstance(series.dtype, TensorDtype)
    expected = tensor_stack([0, 1, 2, 3])
    expected[rows] = expected_tensor
    np.testing.assert_array_equal(series.to_stack(), expected)
    assert series.name == "t"
    assert series.index.tolist() == [10, 11, 12, 13]


@pytest.mark.parametrize("setter,rows", SETTERS)
@pytest.mark.parametrize("value", [None, np.nan, pd.NA], ids=["None", "nan", "NA"])
def test_setitem_missing_value_makes_rows_missing(setter, rows, value):
    """Test that None, np.nan and pd.NA make the selected rows missing through every path."""
    series = _series_for_setitem()

    setter(series, value)

    expected = np.zeros(4, dtype=bool)
    expected[rows] = True
    np.testing.assert_array_equal(series.isna(), expected)
    unchanged = ~expected
    np.testing.assert_array_equal(series.to_stack()[unchanged], tensor_stack([0, 1, 2, 3])[unchanged])


@pytest.mark.parametrize("setter,rows", MULTI_ROW_SETTERS)
@pytest.mark.parametrize("value_kind", ["stack", "list", "tensorseries"])
def test_setitem_sequence_assigns_one_tensor_per_row(setter, rows, value_kind):
    """Test that a stack, a list of tensors or another TensorSeries assigns one tensor per selected row."""
    series = _series_for_setitem()
    new = tensor_stack(range(100, 100 + len(rows)))
    if value_kind == "stack":
        value = new
    elif value_kind == "list":
        value = list(new)
    else:
        value = TensorSeries.from_stack(new, index=series.index[rows])

    setter(series, value)

    expected = tensor_stack([0, 1, 2, 3])
    expected[rows] = new
    np.testing.assert_array_equal(series.to_stack(), expected)


def test_setitem_stack_of_vectors():
    """Test that a stack of 1-d tensors, a 2-d array pandas would read as a column, is assigned per row."""
    vectors = TensorSeries.from_stack(np.zeros((3, 4)))

    vectors.iloc[1:] = np.ones((2, 4))
    vectors.loc[0] = np.full(4, 2.0)

    np.testing.assert_array_equal(vectors.to_stack(), [[2.0] * 4, [1.0] * 4, [1.0] * 4])


@pytest.mark.parametrize("setter,rows", SETTERS)
def test_setitem_wrong_shape_raises_and_leaves_series_unchanged(setter, rows):
    """Test that a tensor or stack of another shape is rejected with a ValueError naming the shape, and
    the series is left as it was."""
    series = _series_for_setitem()
    wrong_shape = TENSOR_SHAPE[::-1]

    with pytest.raises(ValueError, match=rf"Expected a tensor of shape {re.escape(str(TENSOR_SHAPE))}"):
        setter(series, np.ones(wrong_shape))
    with pytest.raises(ValueError, match="Expected a stack of tensors of shape"):
        setter(series, np.ones((len(rows), *wrong_shape)))

    np.testing.assert_array_equal(series.to_stack(), tensor_stack([0, 1, 2, 3]))
    assert not series.isna().any()


def test_fillna_with_tensor():
    """Test that fillna accepts a single tensor, or a number, for every missing row, not in place."""
    series = TensorSeries(TensorExtensionArray.from_sequence([tensor(0), None, tensor(2), None]), name="t")

    filled = series.fillna(tensor(9))

    assert type(filled) is TensorSeries
    assert not filled.isna().any()
    np.testing.assert_array_equal(filled.to_stack(), tensor_stack([0, 9, 2, 9]))
    assert filled.name == "t"
    np.testing.assert_array_equal(series.isna(), [False, True, False, True])

    np.testing.assert_array_equal(series.fillna(0).to_stack()[1], np.zeros(TENSOR_SHAPE))
    with pytest.raises(TypeError, match="value"):  # required, as in pandas
        series.fillna()


def test_where_and_mask_with_tensor():
    """Test that where and mask take a row condition and, optionally, a single tensor or a number for the
    replaced rows."""
    series = _series_for_setitem()
    cond = pd.Series([True, False, True, False], index=series.index)

    np.testing.assert_array_equal(series.where(cond).isna(), [False, True, False, True])
    np.testing.assert_array_equal(series.mask(cond).isna(), [True, False, True, False])
    np.testing.assert_array_equal(series.where(cond, tensor(9)).to_stack(), tensor_stack([0, 9, 2, 9]))
    np.testing.assert_array_equal(series.mask(cond, tensor(9)).to_stack(), tensor_stack([9, 1, 9, 3]))
    np.testing.assert_array_equal(series.mask(cond, 0).to_stack()[0], np.zeros(TENSOR_SHAPE))
    assert type(series.where(cond, tensor(9))) is TensorSeries
    assert series.where(cond, tensor(9)).name == "t"

    series.mask(cond, tensor(9), inplace=True)
    np.testing.assert_array_equal(series.to_stack(), tensor_stack([9, 1, 9, 3]))


def test_setitem_other_dtype_unchanged():
    """Test that a TensorSeries of another dtype assigns, fills and masks as a plain Series."""
    series = TensorSeries([1.0, 2.0, 3.0, np.nan])

    series[0] = 5
    series.loc[1] = 6
    series.iloc[2] = 7

    assert series.tolist()[:3] == [5, 6, 7]
    assert series.fillna(0).tolist() == [5, 6, 7, 0]
    assert series.where(series > 5, 0).tolist() == [0, 6, 7, 0]
    assert series.mask(series > 5, 0).fillna(-1).tolist() == [5, 0, 0, -1]


def test_indexers_are_pandas_subclasses():
    """Test that .loc and .iloc are the pandas indexers, so their private base classes still exist."""
    from pandas.core.indexing import _iLocIndexer, _LocIndexer

    series = _series_for_setitem()
    assert isinstance(series.loc, _LocIndexer)
    assert isinstance(series.iloc, _iLocIndexer)
    assert type(series.loc) is not _LocIndexer
    assert type(series.iloc) is not _iLocIndexer
    np.testing.assert_array_equal(series.loc[11], tensor(1))
    np.testing.assert_array_equal(series.iloc[1], tensor(1))
