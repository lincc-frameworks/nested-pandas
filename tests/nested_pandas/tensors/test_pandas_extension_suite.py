"""Run pandas' extension array conformance suite against TensorExtensionArray.

The suites live in ``pandas.tests.extension.base`` and are driven by the
fixtures in ``conftest.py``. Suites that make no sense for tensors are left
out entirely: arithmetic (for now), reductions and accumulations (tensors are not
numeric scalars), groupby (tensors are not hashable), parsing (tensors are
not read from CSV) and unary ufuncs.

Within the included suites, tests whose assertions compare elements with
``==`` are overridden below with the same logic using ``np.array_equal``,
because a tensor element is an ndarray and ``==`` on it is elementwise.
Tests that cannot pass for tensors are marked xfail from ``conftest.py``,
which lists the reasons: pandas treating an ndarray value as a sequence
rather than a scalar, elements not being hashable, and arrow arrays not
supporting numpy-style views.
"""

import operator

import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest
from pandas.tests.extension import base


def assert_tensor_equal(left, right) -> None:
    """Assert two tensor elements (ndarrays) are equal."""
    assert np.array_equal(left, right)


def assert_tensor_not_equal(left, right) -> None:
    """Assert two tensor elements (ndarrays) differ."""
    assert not np.array_equal(left, right)


class TestConstructors(base.BaseConstructorsTests):
    """Constructing Series and DataFrames from the array, dtype strings and scalars."""


class TestGetitem(base.BaseGetitemTests):
    """Indexing, slicing, take and iteration."""

    def test_get(self, data):
        """Series.get with scalar, list and slice keys."""
        s = pd.Series(data, index=[2 * i for i in range(len(data))])
        assert_tensor_equal(s.get(4), s.iloc[2])

        result = s.get([4, 6])
        expected = s.iloc[[2, 3]]
        tm.assert_series_equal(result, expected)

        result = s.get(slice(2))
        expected = s.iloc[[0, 1]]
        tm.assert_series_equal(result, expected)

        assert s.get(-1) is None
        assert s.get(s.index.max() + 1) is None

        s = pd.Series(data[:6], index=list("abcdef"))
        assert_tensor_equal(s.get("c"), s.iloc[2])

        result = s.get(slice("b", "d"))
        expected = s.iloc[[1, 2, 3]]
        tm.assert_series_equal(result, expected)

        result = s.get("Z")
        assert result is None

        # As of 3.0, getitem with int keys treats them as labels
        assert s.get(4) is None
        assert s.get(-1) is None
        assert s.get(len(s)) is None

        s = pd.Series(data)
        with tm.assert_produces_warning(None):
            s2 = s[::2]
        assert s2.get(1) is None

    def test_take_sequence(self, data):
        """Series positional take with a list."""
        result = pd.Series(data)[[0, 1, 3]]
        assert_tensor_equal(result.iloc[0], data[0])
        assert_tensor_equal(result.iloc[1], data[1])
        assert_tensor_equal(result.iloc[2], data[3])

    def test_take(self, data, na_value, na_cmp):
        """Array take with and without allow_fill."""
        result = data.take([0, -1])
        assert result.dtype == data.dtype
        assert_tensor_equal(result[0], data[0])
        assert_tensor_equal(result[1], data[-1])

        result = data.take([0, -1], allow_fill=True, fill_value=na_value)
        assert_tensor_equal(result[0], data[0])
        assert na_cmp(result[1], na_value)

        with pytest.raises(IndexError, match="out of bounds"):
            data.take([len(data) + 1])

    def test_item(self, data):
        """Series.item on a length-1 Series."""
        s = pd.Series(data)
        result = s[:1].item()
        assert_tensor_equal(result, data[0])

        msg = "can only convert an array of size 1 to a Python scalar"
        with pytest.raises(ValueError, match=msg):
            s[:0].item()
        with pytest.raises(ValueError, match=msg):
            s.item()

    def test_array_item(self, data):
        """Array.item on a length-1 array."""
        arr = data[:1]
        assert_tensor_equal(arr.item(), data[0])

        msg = "can only convert an array of size 1 to a Python scalar"
        with pytest.raises(ValueError, match=msg):
            data[:2].item()
        with pytest.raises(ValueError, match=msg):
            data[:0].item()

    def test_array_item_with_index(self, data):
        """Array.item with an index."""
        assert_tensor_equal(data.item(0), data[0])
        assert_tensor_equal(data.item(-1), data[-1])

        with tm.external_error_raised(IndexError):
            data.item(len(data))

        msg = "index must be an integer"
        with pytest.raises(TypeError, match=msg):
            data.item([0])


class TestSetitem(base.BaseSetitemTests):
    """Assignment with scalars, sequences, masks and indexers."""

    def test_is_immutable(self, data):
        """The array is mutable."""
        assert not data.dtype._is_immutable
        data[0] = data[1]
        assert_tensor_equal(data[0], data[1])

    def test_setitem_scalar_series(self, data, box_in_series):
        """Assigning one element to one position."""
        if box_in_series:
            data = pd.Series(data)
        data[0] = data[1]
        assert_tensor_equal(data[0], data[1])

    def test_setitem_sequence(self, data, box_in_series):
        """Assigning a sequence of elements to integer positions."""
        if box_in_series:
            data = pd.Series(data)
        original = data.copy()
        data[[0, 1]] = [data[1], data[0]]
        assert_tensor_equal(data[0], original[1])
        assert_tensor_equal(data[1], original[0])

    def test_setitem_sequence_broadcasts(self, data, box_in_series):
        """Broadcasting one element to several positions."""
        if box_in_series:
            data = pd.Series(data)
        data[[0, 1]] = data[2]
        assert_tensor_equal(data[0], data[2])
        assert_tensor_equal(data[1], data[2])

    @pytest.mark.parametrize("as_callable", [True, False])
    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_aligned(self, data, as_callable, setter):
        """Assigning with an aligned boolean Series mask."""
        ser = pd.Series(data)
        mask = np.zeros(len(data), dtype=bool)
        mask[:2] = True

        mask2 = (lambda x: mask) if as_callable else mask
        target = getattr(ser, setter) if setter else ser

        target[mask2] = data[5:7]

        ser[mask2] = data[5:7]
        assert_tensor_equal(ser[0], data[5])
        assert_tensor_equal(ser[1], data[6])


class TestInterface(base.BaseInterfaceTests):
    """The ExtensionArray interface: len, ndim, nbytes, isna, copy, array conversion."""

    def test_array_interface(self, data):
        """np.array of the extension array."""
        result = np.array(data)
        assert_tensor_equal(result[0], data[0])

        result = np.array(data, dtype=object)
        expected = np.empty(len(data), dtype=object)
        expected[:] = list(data)
        tm.assert_numpy_array_equal(result, expected)

    def test_copy(self, data):
        """copy gives an independent array."""
        assert_tensor_not_equal(data[0], data[1])
        result = data.copy()
        data[1] = data[0]
        assert_tensor_not_equal(result[1], result[0])

    def test_tolist(self, data):
        """tolist gives the elements."""
        result = data.tolist()
        expected = list(data)
        assert isinstance(result, list)
        assert len(result) == len(expected)
        for left, right in zip(result, expected, strict=True):
            assert_tensor_equal(left, right)


class TestMethods(base.BaseMethodsTests):
    """General methods: shift, fillna, unique, factorize, where, insert and friends."""

    def test_shift_0_periods(self, data):
        """shift(0) returns a copy, not the same object."""
        result = data.shift(0)
        assert_tensor_not_equal(data[0], data[1])
        data[0] = data[1]
        assert_tensor_not_equal(result[0], result[1])

    def test_where_series(self, data, na_value, as_frame):
        """Series.where with a boolean condition, and with `other`."""
        assert_tensor_not_equal(data[0], data[1])
        cls = type(data)
        a, b = data[:2]

        orig = pd.Series(cls._from_sequence([a, a, b, b], dtype=data.dtype))
        ser = orig.copy()
        cond = np.array([True, True, False, False])

        if as_frame:
            ser = ser.to_frame(name="a")
            cond = cond.reshape(-1, 1)

        result = ser.where(cond)
        expected = pd.Series(cls._from_sequence([a, a, na_value, na_value], dtype=data.dtype))

        if as_frame:
            expected = expected.to_frame(name="a")
        tm.assert_equal(result, expected)

        ser.mask(~cond, inplace=True)
        tm.assert_equal(ser, expected)

        # array other
        ser = orig.copy()
        if as_frame:
            ser = ser.to_frame(name="a")
        cond = np.array([True, False, True, True])
        other = cls._from_sequence([a, b, a, b], dtype=data.dtype)
        if as_frame:
            other = pd.DataFrame({"a": other})
            cond = pd.DataFrame({"a": cond})
        result = ser.where(cond, other)
        expected = pd.Series(cls._from_sequence([a, b, b, b], dtype=data.dtype))
        if as_frame:
            expected = expected.to_frame(name="a")
        tm.assert_equal(result, expected)

        ser.mask(~cond, other, inplace=True)
        tm.assert_equal(ser, expected)


class TestMissing(base.BaseMissingTests):
    """Missing value handling: isna, dropna, fillna."""


class TestPrinting(base.BasePrintingTests):
    """repr of arrays, Series and DataFrames."""


class TestReshaping(base.BaseReshapingTests):
    """concat, merge, stack, unstack, transpose and ravel."""


class TestCasting(base.BaseCastingTests):
    """astype to object, string and self."""

    def test_tolist(self, data):
        """Series.tolist gives the elements."""
        result = pd.Series(data).tolist()
        expected = list(data)
        assert len(result) == len(expected)
        for left, right in zip(result, expected, strict=True):
            assert_tensor_equal(left, right)


class TestIndex(base.BaseIndexTests):
    """Holding the array in an Index."""


class TestComparisonOps(base.BaseComparisonOpsTests):
    """== and != against scalars and arrays."""

    def _compare_other(self, ser: pd.Series, data, op, other):
        """Check the vectorized comparison against a pointwise np.array_equal.

        The base implementation builds the expectation with ``Series.combine``,
        which applies the operator to pairs of ndarrays and gets elementwise
        results rather than one boolean per row.
        """
        result = op(ser, other)
        others = list(other) if isinstance(other, pd.Series | pd.Index) else [other] * len(ser)
        equal = [np.array_equal(left, right) for left, right in zip(ser, others, strict=True)]
        if op is operator.ne:
            equal = [not value for value in equal]
        expected = pd.Series(equal, dtype="boolean", index=ser.index, name=ser.name)
        tm.assert_series_equal(result, expected)
