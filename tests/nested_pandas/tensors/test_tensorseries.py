import operator
import re
import warnings

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


def _non_tensor_series(data=(1.0, 2.0, 3.0, np.nan)) -> TensorSeries:
    """A TensorSeries holding ``data`` of another dtype, built around the constructor's TypeError.

    The constructor rejects such data and the results of operations fall back to a plain Series, but pandas
    can still leave a TensorSeries holding another dtype by modifying it in place (enlarging it with a new
    label does so under pandas 3.0, while 3.1 keeps the tensor dtype), so the tests build one directly.
    """
    series = TensorSeries.__new__(TensorSeries)
    pd.Series.__init__(series, list(data))
    assert type(series) is TensorSeries
    assert not isinstance(series.dtype, TensorDtype)
    return series


def test_init_rejects_other_dtypes():
    """Test that a TensorSeries cannot be constructed from data of another dtype, with or without data."""
    with pytest.raises(TypeError, match="TensorSeries requires a TensorDtype, not 'int64'"):
        TensorSeries([1, 2, 3])
    with pytest.raises(TypeError, match="TensorSeries requires a TensorDtype, not 'object'"):
        TensorSeries()

    empty = TensorSeries([], dtype=_series_for_setitem().dtype)
    assert len(empty) == 0
    assert isinstance(empty.dtype, TensorDtype)


def test_tensor_only_decorator():
    """Test that tensor-only members raise a TypeError for other dtypes."""
    series = _non_tensor_series()

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
    assert _non_tensor_series()._repr_html_() is None


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
    """Test that a TensorSeries pandas left holding another dtype assigns, fills and masks as a plain
    Series."""
    series = _non_tensor_series([1.0, 2.0, 3.0, np.nan])

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


# Arithmetic (#540) #


def _series_for_arithmetic() -> TensorSeries:
    """A three-row series with a missing row, shared by the arithmetic tests."""
    return TensorSeries(
        TensorExtensionArray.from_sequence([tensor(0), tensor(1), None, tensor(3)]),
        index=[10, 11, 12, 13],
        name="t",
    )


ARITHMETIC_OPERATORS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
]
"""The binary arithmetic operators, applied as ``op(left, right)``."""

STACK = tensor_stack([0, 1, 2, 3])
"""The values of :func:`_series_for_arithmetic`, with an arbitrary tensor in the missing row."""
WEIGHTS = 1 + np.arange(np.prod(TENSOR_SHAPE), dtype=np.float64).reshape(TENSOR_SHAPE)
"""A tensor of the element shape, applied to every row."""
PER_ROW = np.array([1.0, 2.0, 3.0, 4.0])
"""One scalar per row."""

OPERANDS = [
    pytest.param(lambda s: 2.0, lambda stack: 2.0, id="scalar"),
    pytest.param(lambda s: WEIGHTS, lambda stack: WEIGHTS, id="tensor"),
    pytest.param(lambda s: STACK + 1, lambda stack: STACK + 1, id="stack"),
    pytest.param(lambda s: PER_ROW, lambda stack: PER_ROW.reshape(-1, 1, 1), id="per-row-array"),
    pytest.param(
        lambda s: pd.Series(PER_ROW, index=s.index, name=s.name),
        lambda stack: PER_ROW.reshape(-1, 1, 1),
        id="per-row-series",
    ),
    pytest.param(
        lambda s: TensorSeries.from_stack(STACK + 1, index=s.index, name=s.name),
        lambda stack: STACK + 1,
        id="tensorseries",
    ),
    pytest.param(lambda s: pa.scalar(2.0), lambda stack: 2.0, id="pa-scalar"),
    pytest.param(lambda s: pa.array(PER_ROW), lambda stack: PER_ROW.reshape(-1, 1, 1), id="pa-array"),
    pytest.param(
        lambda s: pd.Series(PER_ROW, index=s.index, name=s.name, dtype="double[pyarrow]"),
        lambda stack: PER_ROW.reshape(-1, 1, 1),
        id="per-row-arrow-series",
    ),
]
"""Right operands, as (operand for the series, the numpy equivalent for its stack)."""


def _assert_tensor_result(result, series, expected_stack, missing):
    """Assert an arithmetic result is a TensorSeries with the index and name of ``series`` and the given
    values in the present rows."""
    assert type(result) is TensorSeries
    assert isinstance(result.dtype, TensorDtype)
    assert result.index.equals(series.index)
    assert result.name == series.name
    np.testing.assert_array_equal(result.isna(), missing)
    np.testing.assert_array_equal(result.to_stack()[~missing], expected_stack[~missing])


@pytest.mark.parametrize("op", ARITHMETIC_OPERATORS, ids=lambda op: op.__name__)
@pytest.mark.parametrize("operand,numpy_operand", OPERANDS)
def test_arithmetic_operators(op, operand, numpy_operand):
    """Test that the binary operators apply elementwise against a scalar, a tensor, a stack, per-row
    scalars from numpy or a Series, and another TensorSeries, with missing rows staying missing."""
    series = _series_for_arithmetic()
    missing = series.isna().to_numpy()

    result = op(series, operand(series))

    _assert_tensor_result(result, series, op(STACK, numpy_operand(STACK)), missing)


@pytest.mark.parametrize("op", ARITHMETIC_OPERATORS, ids=lambda op: op.__name__)
@pytest.mark.parametrize("operand,numpy_operand", OPERANDS)
def test_arithmetic_reflected_operators(op, operand, numpy_operand):
    """Test that the reflected forms, with the series on the right, give the same as numpy would."""
    series = _series_for_arithmetic()
    missing = series.isna().to_numpy()

    result = op(operand(series), series)

    with np.errstate(divide="ignore", invalid="ignore"):  # the stack has a zero
        expected = op(numpy_operand(STACK), STACK)
    _assert_tensor_result(result, series, expected, missing)


@pytest.mark.parametrize(
    "op", [operator.iadd, operator.isub, operator.imul, operator.itruediv], ids=lambda op: op.__name__
)
def test_arithmetic_inplace_operators(op):
    """Test that the in-place forms update the series, including a dtype-changing one."""
    series = TensorSeries.from_stack(np.arange(6).reshape(3, 2), index=[10, 11, 12], name="t")
    stack = np.arange(6).reshape(3, 2)

    result = op(series, 2)

    assert result is series
    assert type(series) is TensorSeries
    np.testing.assert_array_equal(
        series.to_stack(), op(stack.copy(), 2) if op is not operator.itruediv else stack / 2
    )
    assert series.name == "t"
    assert series.index.tolist() == [10, 11, 12]
    assert str(series.dtype) == ("tensor[double, (2,)]" if op is operator.itruediv else "tensor[int64, (2,)]")


def test_arithmetic_aligns_on_index_with_outer_join():
    """Test that two tensor series align on the index like pandas, with rows missing on either side
    missing in the result."""
    left = TensorSeries.from_stack(tensor_stack([0, 1, 2]), index=[10, 11, 12], name="t")
    right = TensorSeries.from_stack(tensor_stack([10, 20, 30]), index=[12, 11, 13], name="t")

    result = left + right

    assert type(result) is TensorSeries
    assert result.index.tolist() == [10, 11, 12, 13]
    assert result.name == "t"
    np.testing.assert_array_equal(result.isna(), [True, False, False, True])
    np.testing.assert_array_equal(result.to_stack()[1:3], tensor_stack([1, 2]) + tensor_stack([20, 10]))


def test_arithmetic_null_propagation():
    """Test that a missing row on either side gives a missing row, and a missing scalar makes every row
    missing."""
    left = _series_for_arithmetic()
    right = TensorSeries(
        TensorExtensionArray.from_sequence([None, tensor(1), tensor(2), tensor(3)]), index=left.index
    )

    np.testing.assert_array_equal((left * right).isna(), [True, False, True, False])
    np.testing.assert_array_equal((right * left).isna(), [True, False, True, False])
    np.testing.assert_array_equal((left * right).to_stack()[[1, 3]], tensor_stack([1, 3]) ** 2)

    for missing in [pd.NA, None]:
        result = left + missing
        assert type(result) is TensorSeries
        assert result.isna().all()
        assert len(result) == 4


def test_arithmetic_with_pyarrow_operands():
    """Test that pyarrow scalars and arrays, and arrow-backed Series, are operands: a tensor scalar or
    array applies as tensors, a numeric or boolean array as one scalar per row, and other types raise a
    TypeError."""
    series = _series_for_arithmetic()
    missing = series.isna().to_numpy()
    per_row = PER_ROW.reshape(-1, 1, 1)

    tensor_scalar = series.array.pa_array[1]
    assert isinstance(tensor_scalar, pa.ExtensionScalar)
    _assert_tensor_result(series + tensor_scalar, series, STACK + tensor(1), missing)
    _assert_tensor_result(series + pa.scalar(None, pa.float64()), series, STACK, np.ones(4, dtype=bool))

    _assert_tensor_result(series + series.array.pa_array, series, STACK * 2, missing)
    _assert_tensor_result(series + series.array.pa_array.combine_chunks(), series, STACK * 2, missing)

    _assert_tensor_result(series * pa.array([1, 2, 3, 4]), series, STACK * per_row, missing)
    _assert_tensor_result(
        series * pa.chunked_array([[1.0, 2.0], [3.0, 4.0]]), series, STACK * per_row, missing
    )
    bools = pd.Series([True, False, True, False], index=series.index, name="t", dtype="bool[pyarrow]")
    _assert_tensor_result(series * bools, series, STACK * np.array([1, 0, 1, 0]).reshape(-1, 1, 1), missing)
    ints = pd.Series([1, 2, 3, 4], index=series.index, name="t", dtype="int64[pyarrow]")
    _assert_tensor_result(ints * series, series, STACK * per_row, missing)

    with pytest.raises(TypeError):
        series + pa.array(["a", "b", "c", "d"])
    with pytest.raises(TypeError):
        series + pd.Series(["a", "b", "c", "d"], index=series.index, dtype="string[pyarrow]")
    with pytest.raises(ValueError, match="Lengths must match"):
        series + pa.array([1.0, 2.0])


def test_arithmetic_missing_per_row_scalars_give_missing_rows():
    """Test that a missing scalar in a per-row operand makes that row missing, whether it is NaN in a numpy
    array or float Series, a null in a pyarrow array or arrow-backed Series, or a label the Series lacks."""
    series = TensorSeries.from_stack(tensor_stack([0, 1, 2, 3]), index=[10, 11, 12, 13], name="t")
    present = np.array([1.0, 3.0, 4.0]).reshape(-1, 1, 1)
    operands = [
        np.array([1.0, np.nan, 3.0, 4.0]),
        pd.Series([1.0, np.nan, 3.0, 4.0], index=series.index),
        pa.array([1.0, None, 3.0, 4.0]),
        pd.Series([1.0, None, 3.0, 4.0], index=series.index, dtype="double[pyarrow]"),
        pd.Series([1, None, 3, 4], index=series.index, dtype="int64[pyarrow]"),
        pd.Series([1.0, 3.0, 4.0], index=[10, 12, 13]),
    ]

    for operand in operands:
        for result in (series * operand, operand * series):
            assert type(result) is TensorSeries
            np.testing.assert_array_equal(result.isna(), [False, True, False, False])
            np.testing.assert_array_equal(result.to_stack()[[0, 2, 3]], STACK[[0, 2, 3]] * present)


def test_arithmetic_dtype_promotion_and_dim_names():
    """Test that the value type follows numpy, integer tensors dividing to doubles, and that dim_names
    are kept when the element shape is unchanged and dropped when it changes."""
    ints = TensorSeries.from_stack(np.arange(6).reshape(3, 2), dtype="tensor[int64, (2,), dim_names=[x]]")

    assert str((ints / 2).dtype) == "tensor[double, (2,), dim_names=[x]]"
    assert str((ints * 2).dtype) == "tensor[int64, (2,), dim_names=[x]]"
    assert str((ints + 0.5).dtype) == "tensor[double, (2,), dim_names=[x]]"
    assert str((ints + np.float32(1)).dtype) == "tensor[double, (2,), dim_names=[x]]"

    # Broadcasting changes the element shape as numpy would, dropping the dim_names: (1, 2) tensors
    # against a (2, 2) array, and (2,) tensors against a series of (2, 2) tensors
    rows = TensorSeries.from_stack(
        np.arange(6).reshape(3, 1, 2), dtype="tensor[int64, (1, 2), dim_names=[y, x]]"
    )
    broadcast = rows * np.ones((2, 2))
    assert str(broadcast.dtype) == "tensor[double, (2, 2)]"
    np.testing.assert_array_equal(broadcast.to_stack(), np.arange(6).reshape(3, 1, 2) * np.ones((2, 2)))

    squares = TensorSeries.from_stack(np.ones((3, 2, 2)))
    assert str((ints + squares).dtype) == "tensor[double, (2, 2)]"
    assert str((squares + ints).dtype) == "tensor[double, (2, 2)]"
    np.testing.assert_array_equal(
        (ints + squares).to_stack(), np.arange(6).reshape(3, 1, 2) + np.ones((3, 2, 2))
    )


def test_arithmetic_shape_and_length_errors():
    """Test that an operand that cannot broadcast raises a ValueError naming both shapes, and one of the
    wrong length raises a ValueError."""
    series = TensorSeries.from_stack(tensor_stack([0, 1, 2]))

    with pytest.raises(ValueError, match=r"\(3, 2\).*\(2, 3\)"):
        series * np.ones((3, 2))
    with pytest.raises(ValueError, match=r"\(1, 1, 1, 1\).*\(2, 3\)"):
        series * np.ones((1, 1, 1, 1))
    with pytest.raises(ValueError, match="Lengths must match"):
        series * np.ones((2, *TENSOR_SHAPE))
    with pytest.raises(ValueError, match=r"\(2,\).*\(2, 3\)"):
        series * np.ones(2)
    with pytest.raises(ValueError, match=r"\(1, 2\).*\(2, 3\)"):
        series * TensorSeries.from_stack(np.ones((3, 1, 2)))
    with pytest.raises(ValueError, match="Lengths must match"):
        series * [tensor(1), tensor(2)]


def test_division_by_zero_is_silent():
    """Test that dividing by zero gives inf or nan without numpy warnings, as pandas does."""
    series = TensorSeries.from_stack(tensor_stack([0, 1]))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = series / 0
        zeros = TensorSeries.from_stack(np.zeros((2, *TENSOR_SHAPE))) // 0

    assert np.isinf(result.to_stack()[1]).all()
    assert np.isnan(result.to_stack()[0, 0, 0])  # 0 / 0
    assert np.isnan(zeros.to_stack()).all()


def test_unary_operators():
    """Test that -, + and abs apply per element and keep the index, name and missing rows."""
    series = _series_for_arithmetic()
    missing = series.isna().to_numpy()

    _assert_tensor_result(-series, series, -STACK, missing)
    _assert_tensor_result(+series, series, STACK, missing)
    _assert_tensor_result(abs(-series), series, STACK, missing)


def test_divmod():
    """Test that divmod gives a pair of TensorSeries, in both operand orders."""
    series = _series_for_arithmetic()
    missing = series.isna().to_numpy()

    quotient, remainder = divmod(series, 0.7)
    _assert_tensor_result(quotient, series, STACK // 0.7, missing)
    _assert_tensor_result(remainder, series, STACK % 0.7, missing)

    quotient, remainder = divmod(5.0, series)
    with np.errstate(divide="ignore", invalid="ignore"):  # the stack has a zero
        expected_quotient, expected_remainder = 5.0 // STACK, 5.0 % STACK
    _assert_tensor_result(quotient, series, expected_quotient, missing)
    _assert_tensor_result(remainder, series, expected_remainder, missing)


def test_ufuncs():
    """Test that unary and binary numpy ufuncs apply per element, returning TensorSeries, including a
    tuple-output ufunc and a boolean-output ufunc."""
    series = _series_for_arithmetic()
    missing = series.isna().to_numpy()

    _assert_tensor_result(np.sqrt(series), series, np.sqrt(STACK), missing)
    _assert_tensor_result(np.exp(series), series, np.exp(STACK), missing)
    _assert_tensor_result(np.add(series, 2), series, STACK + 2, missing)
    _assert_tensor_result(np.minimum(series, WEIGHTS), series, np.minimum(STACK, WEIGHTS), missing)
    _assert_tensor_result(np.minimum(WEIGHTS, series), series, np.minimum(STACK, WEIGHTS), missing)

    fractional, integral = np.modf(series)
    _assert_tensor_result(fractional, series, np.modf(STACK)[0], missing)
    _assert_tensor_result(integral, series, np.modf(STACK)[1], missing)
    quotient, remainder = np.divmod(series, 2)
    _assert_tensor_result(quotient, series, STACK // 2, missing)
    _assert_tensor_result(remainder, series, STACK % 2, missing)

    finite = np.isfinite(series / series)  # the first element of the first row is 0 / 0
    assert type(finite) is TensorSeries
    assert str(finite.dtype) == f"tensor[bool, {TENSOR_SHAPE}]"
    np.testing.assert_array_equal(finite.isna(), missing)
    expected = np.ones((4, *TENSOR_SHAPE), dtype=bool)
    expected[0, 0, 0] = False
    np.testing.assert_array_equal(finite.to_stack(na_value=True), expected)

    # A ufunc between two tensor series aligns on the index
    other = TensorSeries.from_stack(tensor_stack([3, 2, 1, 0]), index=[13, 12, 11, 10])
    np.testing.assert_array_equal(np.add(series, other).to_stack()[[0, 1, 3]], STACK[[0, 1, 3]] * 2)


@pytest.mark.parametrize(
    "op",
    [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge],
    ids=lambda op: op.__name__,
)
@pytest.mark.parametrize("other", [1.0, "tensorseries"])
def test_comparison_operators_raise(op, other):
    """Test that the comparison operators raise a TypeError naming the operator."""
    series = _series_for_arithmetic()
    if other == "tensorseries":
        other = series.copy()

    with pytest.raises(TypeError, match=op.__name__):
        op(series, other)
    with pytest.raises(TypeError, match=op.__name__):
        op(series.array, other if np.isscalar(other) else other.array)


@pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor], ids=lambda op: op.__name__)
def test_logical_operators_raise(op):
    """Test that &, | and ^ raise a TypeError naming the operator, in both operand orders."""
    series = _series_for_arithmetic()
    other = np.isfinite(series)

    with pytest.raises(TypeError, match=op.__name__):
        op(series, other)
    with pytest.raises(TypeError, match=op.__name__):
        op(other, series)
    with pytest.raises(TypeError, match=op.__name__):
        op(series.array, True)


@pytest.mark.parametrize(
    "ufunc", [np.less, np.equal, np.greater_equal, np.logical_and, np.logical_or, np.logical_xor], ids=str
)
def test_comparison_and_logical_ufuncs_raise(ufunc):
    """Test that the comparison and logical ufuncs raise a TypeError like the operators."""
    series = _series_for_arithmetic()

    with pytest.raises(TypeError):
        ufunc(series, 1.0)
    with pytest.raises(TypeError):
        ufunc(series.array, WEIGHTS)
    with pytest.raises(TypeError):
        ufunc(WEIGHTS, series.array)


def test_arithmetic_other_dtype_unchanged():
    """Test that a TensorSeries of another dtype computes as a plain Series."""
    series = _non_tensor_series([1.0, 2.0, 3.0])

    assert (series + 1).tolist() == [2.0, 3.0, 4.0]
    assert (series == 2.0).tolist() == [False, True, False]
    assert (series > 1.5).tolist() == [False, True, True]
    assert np.sqrt(series).tolist() == [1.0, np.sqrt(2.0), np.sqrt(3.0)]


def test_nestedframe_arithmetic_with_tensor_columns():
    """Test that frame-level arithmetic works with tensor columns, for a scalar and another frame."""
    nf = NestedFrame(
        {"a": [1.0, 2.0, 3.0], "t": TensorExtensionArray.from_stack(tensor_stack([0, 1, 2]))},
        index=[10, 11, 12],
    )

    plus_one = nf + 1
    assert plus_one["a"].tolist() == [2.0, 3.0, 4.0]
    assert type(plus_one["t"]) is TensorSeries
    np.testing.assert_array_equal(plus_one["t"].to_stack(), tensor_stack([0, 1, 2]) + 1)

    doubled = nf + nf
    assert doubled["a"].tolist() == [2.0, 4.0, 6.0]
    np.testing.assert_array_equal(doubled["t"].to_stack(), tensor_stack([0, 1, 2]) * 2)
    assert doubled.index.equals(nf.index)
