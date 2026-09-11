"""Fixtures for the tensor tests, including those pandas' extension test suite expects.

pandas ships a conformance suite for third-party extension arrays in
``pandas.tests.extension.base``. Its tests are driven by fixtures that pandas
defines in its own ``conftest.py`` files, which are not importable, so the ones
the tensor suites use are replicated here with the same names and semantics.
"""

import operator

import pandas as pd
import pyarrow as pa
import pytest
from tensor_test_utils import TENSOR_SHAPE, tensor, tensor_stack

from nested_pandas import TensorDtype
from nested_pandas.tensors.ext_array import TensorExtensionArray

# Expected failures in pandas' conformance suite (test_pandas_extension_suite.py) #

PANDAS_NDARRAY_SCALAR = (
    "pandas treats an ndarray value as a sequence, not a scalar, before the extension array is consulted"
)
NOT_HASHABLE = "tensors are not hashable"
NO_VIEWS = (
    "an arrow-backed array cannot share buffers with a view (pandas' ArrowExtensionArray xfails this too)"
)
NO_ORDERING = "tensors have no ordering"

XFAIL_STRICT = {
    # Series/DataFrame __setitem__ and fillna inspect an ndarray value themselves and reject it,
    # or misread a 2-d tensor as a (N, 1) column, before our __setitem__ runs. The array-level
    # operations work, see test_tensor_ext_array.py.
    "test_series_constructor_scalar_with_index": PANDAS_NDARRAY_SCALAR,
    "test_setitem_scalar": PANDAS_NDARRAY_SCALAR,
    "test_setitem_loc_scalar_mixed": PANDAS_NDARRAY_SCALAR,
    "test_setitem_loc_scalar_single": PANDAS_NDARRAY_SCALAR,
    "test_setitem_loc_scalar_multiple_homogoneous": PANDAS_NDARRAY_SCALAR,
    "test_setitem_iloc_scalar_mixed": PANDAS_NDARRAY_SCALAR,
    "test_setitem_iloc_scalar_single": PANDAS_NDARRAY_SCALAR,
    "test_setitem_iloc_scalar_multiple_homogoneous": PANDAS_NDARRAY_SCALAR,
    "test_setitem_loc_iloc_slice": PANDAS_NDARRAY_SCALAR,
    "test_setitem_mask_broadcast": PANDAS_NDARRAY_SCALAR,
    "test_setitem_with_expansion_row": PANDAS_NDARRAY_SCALAR,
    "test_fillna_scalar": PANDAS_NDARRAY_SCALAR,
    "test_fillna_series": PANDAS_NDARRAY_SCALAR,
    "test_fillna_frame": PANDAS_NDARRAY_SCALAR,
    "test_fillna_no_op_returns_copy": PANDAS_NDARRAY_SCALAR,
    "test_fillna_readonly": PANDAS_NDARRAY_SCALAR,
    "test_fillna_copy_frame": PANDAS_NDARRAY_SCALAR,
    "test_fillna_copy_series": PANDAS_NDARRAY_SCALAR,
    "test_fillna_limit_frame": PANDAS_NDARRAY_SCALAR,
    "test_fillna_limit_series": PANDAS_NDARRAY_SCALAR,
    # Hash tables
    "test_unique": NOT_HASHABLE,
    "test_duplicated": NOT_HASHABLE,
    "test_value_counts_with_normalize": NOT_HASHABLE,
    "test_hash_pandas_object": NOT_HASHABLE,
    "test_hash_pandas_object_works": NOT_HASHABLE,
    "test_factorize_empty": NOT_HASHABLE,
    "test_merge_on_extension_array": NOT_HASHABLE,
    "test_merge_on_extension_array_duplicates": NOT_HASHABLE,
    # Ordering
    "test_combine_le": NO_ORDERING,
}
"""Suite tests that must fail, by test function name."""

XFAIL_NO_RUN = {
    "test_view": NO_VIEWS,
    "test_ravel": NO_VIEWS,
    "test_transpose": NO_VIEWS,
    "test_setitem_preserves_views": NO_VIEWS,
}
"""Suite tests that are not run at all, as pandas does for its own arrow array."""

XFAIL_FOR_PARAM = {
    # Broadcasting one tensor to several positions of a Series: pandas length-checks the ndarray.
    # Series.__setitem__ with a single position and with a sequence of tensors both work.
    "test_setitem_sequence_broadcasts": ("box_in_series", True, PANDAS_NDARRAY_SCALAR),
    "test_setitem_integer_array": ("box_in_series", True, PANDAS_NDARRAY_SCALAR),
    "test_setitem_mask": ("box_in_series", True, PANDAS_NDARRAY_SCALAR),
    "test_setitem_mask_boolean_array_with_na": ("box_in_series", True, PANDAS_NDARRAY_SCALAR),
    "test_setitem_slice": ("box_in_series", True, PANDAS_NDARRAY_SCALAR),
    # value_counts only hashes when there is more than one distinct element
    "test_value_counts": ("all_data", "data", NOT_HASHABLE),
}
"""Suite tests that must fail only for one value of one parameter: name -> (param, value, reason)."""


def pytest_collection_modifyitems(items):
    """Mark the pandas conformance tests that cannot pass for tensors as xfail.

    Adding markers here rather than overriding the tests keeps their
    parametrization intact.
    """
    for item in items:
        if "test_pandas_extension_suite.py" not in item.nodeid:
            continue
        name = getattr(item, "originalname", None) or item.name
        if name in XFAIL_STRICT:
            item.add_marker(pytest.mark.xfail(reason=XFAIL_STRICT[name], strict=True))
        elif name in XFAIL_NO_RUN:
            item.add_marker(pytest.mark.xfail(reason=XFAIL_NO_RUN[name], run=False))
        elif name in XFAIL_FOR_PARAM:
            param, value, reason = XFAIL_FOR_PARAM[name]
            params = getattr(getattr(item, "callspec", None), "params", {})
            if params.get(param) == value:
                item.add_marker(pytest.mark.xfail(reason=reason, strict=True))


# Fixtures with the names and semantics of pandas/tests/extension/conftest.py #


@pytest.fixture
def dtype():
    """The dtype the pandas suite is run against."""
    return TensorDtype(pa.fixed_shape_tensor(pa.float64(), list(TENSOR_SHAPE)))


@pytest.fixture
def data(dtype):
    """Length-10 array without missing values, where data[0] != data[1], as the pandas suite expects."""
    return TensorExtensionArray.from_stack(tensor_stack(range(10)), dtype=dtype)


@pytest.fixture
def data_for_twos(dtype):
    """Length-100 array where every element is 2; only meaningful for numeric dtypes."""
    pytest.skip(f"{dtype} is not a numeric dtype")


@pytest.fixture
def data_missing(dtype):
    """Length-2 array of [missing, valid]."""
    return TensorExtensionArray.from_sequence([None, tensor(1)], dtype=dtype)


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving both ``data`` and ``data_missing``."""
    if request.param == "data":
        return data
    return data_missing


@pytest.fixture
def data_repeated(data):
    """Generator yielding ``data`` repeatedly."""

    def gen(count):
        for _ in range(count):
            yield data

    return gen


@pytest.fixture
def data_for_sorting():
    """Length-3 array with a known ordering; tensors have none."""
    pytest.skip("tensors have no ordering")


@pytest.fixture
def data_missing_for_sorting():
    """Length-3 array with a known ordering and a missing value; tensors have no ordering."""
    pytest.skip("tensors have no ordering")


@pytest.fixture
def na_cmp():
    """Binary operator comparing two NA values; pd.NA is a singleton."""
    return operator.is_


@pytest.fixture
def na_value(dtype):
    """The scalar missing value for the dtype."""
    return dtype.na_value


@pytest.fixture
def data_for_grouping():
    """Data for grouping tests; requires hashable values, which tensors are not."""
    pytest.skip("tensors are not hashable, so they cannot be grouped")


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series."""
    return request.param


@pytest.fixture(params=[True, False])
def as_frame(request):
    """Whether to convert to a DataFrame."""
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request):
    """Whether to convert to a Series."""
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request):
    """Whether to use numpy functions rather than pandas methods."""
    return request.param


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request):
    """Method parameter for fillna tests."""
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request):
    """Whether to convert to an ExtensionArray."""
    return request.param


@pytest.fixture
def invalid_scalar(data):
    """A scalar that cannot be held by this extension array."""
    return object.__new__(object)


# Fixtures with the names and semantics of pandas/conftest.py #


@pytest.fixture(params=[True, False])
def skipna(request):
    """Boolean skipna parameter."""
    return request.param


@pytest.fixture(params=[True, False])
def ascending(request):
    """Boolean ascending parameter."""
    return request.param


@pytest.fixture(params=[True, False])
def dropna(request):
    """Boolean dropna parameter."""
    return request.param


@pytest.fixture(params=[None, "ignore"])
def na_action(request):
    """na_action parameter for map."""
    return request.param


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    """Key parameter for sort_values."""
    return request.param


@pytest.fixture(params=[operator.eq, operator.ne], ids=["eq", "ne"])
def comparison_op(request):
    """Comparison operators the tensor array supports; ordering comparisons are not defined."""
    return request.param


@pytest.fixture
def using_nan_is_na() -> bool:
    """Whether pandas treats NaN as missing for nullable dtypes."""
    try:
        return bool(pd.get_option("mode.nan_is_na"))
    except (pd.errors.OptionError, KeyError):
        return True
