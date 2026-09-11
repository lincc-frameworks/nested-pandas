import io
import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_array_equal
from pandas.core.arrays import ArrowExtensionArray  # type: ignore[attr-defined]
from pandas.testing import assert_series_equal
from tensor_test_utils import TENSOR_SHAPE, tensor_stack

from nested_pandas import TensorDtype
from nested_pandas.tensors.ext_array import (
    TENSOR_FORMATTING_MAX_ELEMENTS,
    TensorExtensionArray,
    _format_tensor,
)


@pytest.fixture
def stack():
    """A (4, 2, 3) stack of distinct tensors."""
    return tensor_stack(range(4))


@pytest.fixture
def array(stack, dtype):
    """A TensorExtensionArray of the four tensors in ``stack``."""
    return TensorExtensionArray.from_stack(stack, dtype=dtype)


@pytest.fixture
def array_with_missing(stack, dtype):
    """[stack[0], NA, NA, stack[3]]."""
    return TensorExtensionArray.from_sequence([stack[0], None, pd.NA, stack[3]], dtype=dtype)


def storage_array(stack, mask=None) -> pa.FixedSizeListArray:
    """The fixed_size_list storage array for a stack, optionally with a row-null mask."""
    values = pa.array(np.ascontiguousarray(stack).reshape(-1))
    return pa.FixedSizeListArray.from_arrays(values, int(np.prod(stack.shape[1:])), mask=mask)


# Constructor #


def test___init___from_extension_array(stack, dtype):
    """Test constructing from a fixed_shape_tensor extension array infers the dtype."""
    ext = pa.ExtensionArray.from_storage(dtype.pyarrow_dtype, storage_array(stack))
    array = TensorExtensionArray(ext)
    assert array.dtype == dtype
    assert len(array) == 4
    assert array.num_chunks == 1
    assert_array_equal(array.to_stack(), stack)


def test___init___drops_identity_permutation(stack, dtype):
    """Test that an identity permutation is normalized away from the stored array as from the dtype.

    ``pa.FixedShapeTensorArray.from_numpy_ndarray()`` always sets one, and pyarrow compares the
    types with and without it equal, so the array must check for it explicitly.
    """
    ext = pa.FixedShapeTensorArray.from_numpy_ndarray(stack)
    assert ext.type.permutation is not None
    array = TensorExtensionArray(ext)
    assert array.dtype == dtype
    assert array.pa_array.type.permutation is None
    assert array.pa_array.type == dtype.pyarrow_dtype
    assert pa.array(array).type.permutation is None
    assert_array_equal(array.to_stack(), stack)

    # Also when the dtype is given explicitly
    array = TensorExtensionArray(ext, dtype=dtype)
    assert array.pa_array.type.permutation is None


def test___init___from_chunked_array(stack, dtype):
    """Test constructing from a chunked array keeps the chunks."""
    ext = pa.ExtensionArray.from_storage(dtype.pyarrow_dtype, storage_array(stack))
    array = TensorExtensionArray(pa.chunked_array([ext[:1], ext[1:]]))
    assert array.num_chunks == 2
    assert len(array) == 4
    assert_array_equal(array.to_stack(), stack)


def test___init___from_storage_with_dtype(stack, dtype):
    """Test constructing from the fixed_size_list storage array when the dtype is given."""
    array = TensorExtensionArray(storage_array(stack), dtype=dtype)
    assert array.dtype == dtype
    assert isinstance(array.pa_array.type, pa.FixedShapeTensorType)
    assert_array_equal(array.to_stack(), stack)


def test___init___from_storage_without_dtype_raises(stack):
    """Test that a bare storage array is ambiguous without a dtype."""
    with pytest.raises(ValueError, match="dtype is required"):
        TensorExtensionArray(storage_array(stack))


def test___init___casts_storage_to_dtype(stack, dtype):
    """Test that a dtype with a different element type casts the storage."""
    float32_dtype = TensorDtype(pa.fixed_shape_tensor(pa.float32(), list(TENSOR_SHAPE)))
    array = TensorExtensionArray(storage_array(stack), dtype=float32_dtype)
    assert array.dtype == float32_dtype
    assert array.to_stack().dtype == np.float32
    assert_array_equal(array.to_stack(), stack.astype(np.float32))

    # Also from an extension array of another tensor type
    array = TensorExtensionArray(TensorExtensionArray.from_stack(stack).pa_array, dtype=float32_dtype)
    assert array.dtype == float32_dtype


def test___init___raises_for_size_mismatch(stack):
    """Test that a dtype with a different number of elements is rejected with a clear message."""
    other = TensorDtype(pa.fixed_shape_tensor(pa.float64(), [4]))
    with pytest.raises(ValueError, match="6 elements to .* which has 4 elements"):
        TensorExtensionArray(storage_array(stack), dtype=other)


@pytest.mark.parametrize(
    "values",
    [
        pa.array([1.0, 2.0]),
        pa.array([[1.0, 2.0]]),
        pa.array([{"a": [1.0]}]),
        pa.chunked_array([pa.array([1, 2])]),
    ],
)
def test___init___raises_for_other_arrow_types(values, dtype):
    """Test that arrays that are neither tensor nor fixed_size_list are rejected."""
    with pytest.raises(ValueError, match="fixed_shape_tensor or fixed_size_list"):
        TensorExtensionArray(values, dtype=dtype)


def test___init___raises_for_non_arrow(stack):
    """Test that non-arrow input is rejected."""
    with pytest.raises(TypeError, match="pyarrow Array or ChunkedArray"):
        TensorExtensionArray(stack)


# from_sequence / from_stack #


def test_from_sequence_with_list_of_arrays(stack, dtype):
    """Test from_sequence with a list of numpy arrays and a dtype."""
    array = TensorExtensionArray.from_sequence(list(stack), dtype=dtype)
    assert array.dtype == dtype
    assert_array_equal(array.to_stack(), stack)


def test_from_sequence_infers_dtype(stack):
    """Test that the dtype is inferred from the first non-missing element."""
    array = TensorExtensionArray.from_sequence([None, stack[1].astype(np.int32), stack[2].astype(np.int32)])
    assert array.dtype == TensorDtype(pa.fixed_shape_tensor(pa.int32(), list(TENSOR_SHAPE)))
    assert array.isna().tolist() == [True, False, False]


def test_from_sequence_with_missing(stack, dtype):
    """Test that None and pd.NA both give missing tensors."""
    array = TensorExtensionArray.from_sequence([stack[0], None, pd.NA], dtype=dtype)
    assert array.isna().tolist() == [False, True, True]
    assert_array_equal(array[0], stack[0])


def test_from_sequence_with_lists(dtype):
    """Test that nested lists are accepted as tensors."""
    array = TensorExtensionArray.from_sequence([[[0, 1, 2], [3, 4, 5]]], dtype=dtype)
    assert_array_equal(array[0], np.arange(6.0).reshape(TENSOR_SHAPE))


def test_from_sequence_with_pyarrow_array(stack, dtype):
    """Test from_sequence with an extension array and with a storage array plus dtype."""
    ext = pa.ExtensionArray.from_storage(dtype.pyarrow_dtype, storage_array(stack))
    expected = TensorExtensionArray(ext)
    assert TensorExtensionArray.from_sequence(ext).equals(expected)
    assert TensorExtensionArray.from_sequence(storage_array(stack), dtype=dtype).equals(expected)


def test_from_sequence_with_tensor_scalars(stack, dtype):
    """Test from_sequence with pyarrow tensor scalars and storage scalars."""
    ext = pa.ExtensionArray.from_storage(dtype.pyarrow_dtype, storage_array(stack))
    array = TensorExtensionArray.from_sequence([ext[0], ext[1]], dtype=dtype)
    assert_array_equal(array.to_stack(), stack[:2])
    array = TensorExtensionArray.from_sequence([storage_array(stack)[2]], dtype=dtype)
    assert_array_equal(array[0], stack[2])
    # The dtype is inferred from a tensor scalar too
    inferred = TensorExtensionArray.from_sequence([None, ext[1]])
    assert inferred.dtype == dtype
    assert_array_equal(inferred[1], stack[1])


def test_from_sequence_with_stack(stack, dtype):
    """Test that an (n, *shape) numpy array is accepted as a stack."""
    array = TensorExtensionArray.from_sequence(stack)
    assert array.dtype == dtype
    assert_array_equal(array.to_stack(), stack)


def test_from_sequence_with_self(array):
    """Test that from_sequence with an existing array returns it, or casts it."""
    assert TensorExtensionArray.from_sequence(array) is array
    assert TensorExtensionArray.from_sequence(array, dtype=array.dtype) is array
    float32_dtype = TensorDtype(pa.fixed_shape_tensor(pa.float32(), list(TENSOR_SHAPE)))
    assert TensorExtensionArray.from_sequence(array, dtype=float32_dtype).dtype == float32_dtype


@pytest.mark.parametrize(
    "dtype_spec",
    [
        "tensor[double, (2, 3)]",
        pa.fixed_shape_tensor(pa.float64(), [2, 3]),
        pd.ArrowDtype(pa.fixed_shape_tensor(pa.float64(), [2, 3])),
    ],
)
def test_from_sequence_dtype_spellings(stack, dtype, dtype_spec):
    """Test that the dtype may be given as a string, a pyarrow type or a pd.ArrowDtype."""
    array = TensorExtensionArray.from_sequence(list(stack), dtype=dtype_spec)
    assert array.dtype == dtype


def test_from_sequence_raises_for_wrong_shape(dtype):
    """Test that a tensor of the wrong shape is rejected."""
    with pytest.raises(ValueError, match="Expected a tensor of shape"):
        TensorExtensionArray.from_sequence([np.zeros((3, 2))], dtype=dtype)


def test_from_sequence_raises_for_all_missing_without_dtype():
    """Test that the dtype cannot be inferred from missing values only."""
    with pytest.raises(ValueError, match="Cannot infer TensorDtype"):
        TensorExtensionArray.from_sequence([None, None])
    assert TensorExtensionArray.from_sequence([None, None], dtype="tensor[double, (2, 3)]").isna().all()


def test_from_sequence_raises_for_bad_dtype(stack):
    """Test that an unrelated dtype is rejected."""
    with pytest.raises(TypeError, match="Expected TensorDtype"):
        TensorExtensionArray.from_sequence(list(stack), dtype="int64")


def test_from_stack(stack, dtype):
    """Test from_stack infers the dtype and is zero-copy for C-ordered input."""
    array = TensorExtensionArray.from_stack(stack)
    assert array.dtype == dtype
    assert np.shares_memory(array.to_stack(), stack)


def test_from_stack_with_dtype(stack):
    """Test from_stack casts to the given element type."""
    float32_dtype = TensorDtype(pa.fixed_shape_tensor(pa.float32(), list(TENSOR_SHAPE)))
    array = TensorExtensionArray.from_stack(stack, dtype=float32_dtype)
    assert array.dtype == float32_dtype
    assert_array_equal(array.to_stack(), stack.astype(np.float32))


def test_from_stack_non_contiguous(stack):
    """Test that a non C-contiguous stack is copied into C order rather than mis-read."""
    transposed = np.ascontiguousarray(stack.transpose(0, 2, 1)).transpose(0, 2, 1)
    assert not transposed.flags.c_contiguous
    array = TensorExtensionArray.from_stack(transposed)
    assert_array_equal(array.to_stack(), stack)


def test_from_stack_raises(stack, dtype):
    """Test from_stack rejects too few dimensions and a shape mismatch."""
    with pytest.raises(ValueError, match="at least two dimensions"):
        TensorExtensionArray.from_stack(np.arange(6.0))
    with pytest.raises(ValueError, match="Expected a stack of tensors of shape"):
        TensorExtensionArray.from_stack(stack.reshape(4, 3, 2), dtype=dtype)


# to_stack #


def test_to_stack_zero_copy(array, stack):
    """Test that to_stack is a read-only view for a single chunk without missing values."""
    result = array.to_stack()
    assert_array_equal(result, stack)
    assert np.shares_memory(result, stack)
    assert not result.flags.writeable


def test_to_stack_multiple_chunks(array, stack):
    """Test that to_stack combines chunks."""
    chunked = TensorExtensionArray._concat_same_type([array[:2], array[2:]])
    assert chunked.num_chunks == 2
    assert_array_equal(chunked.to_stack(), stack)


def test_to_stack_with_missing(array_with_missing, stack):
    """Test that missing tensors are filled with na_value, widening the dtype if needed."""
    result = array_with_missing.to_stack()
    assert result.dtype == np.float64
    assert np.isnan(result[1]).all() and np.isnan(result[2]).all()
    assert_array_equal(result[0], stack[0])
    assert_array_equal(result[3], stack[3])
    assert_array_equal(array_with_missing.to_stack(na_value=-1.0)[1], np.full(TENSOR_SHAPE, -1.0))


def test_to_stack_int_widening():
    """Test that integer tensors widen to float for a nan fill, and keep their dtype otherwise."""
    array = TensorExtensionArray.from_sequence([np.ones(TENSOR_SHAPE, dtype=np.int32), None])
    assert array.to_stack().dtype == np.float64
    assert array.to_stack(na_value=0).dtype == np.int32
    no_missing = TensorExtensionArray.from_sequence([np.ones(TENSOR_SHAPE, dtype=np.int32)])
    assert no_missing.to_stack().dtype == np.int32


def test_to_stack_empty(array):
    """Test to_stack for an empty array."""
    result = array[:0].to_stack()
    assert result.shape == (0, *TENSOR_SHAPE)
    assert result.dtype == np.float64


def test_to_stack_sliced(array, stack):
    """Test that a sliced array, whose arrow buffers have an offset, gives the right rows."""
    assert_array_equal(array[1:3].to_stack(), stack[1:3])
    assert_array_equal(array[1:][1:].to_stack(), stack[2:])


# Element-level nulls, which only arrow input can produce #


@pytest.fixture
def element_null_array(stack, dtype):
    """stack with element [0, 0, 1] null and row 1 null."""
    values = pa.array(np.ascontiguousarray(stack).reshape(-1).tolist(), type=pa.float64())
    values = pa.compute.if_else(pa.array(np.arange(len(values)) == 1), None, values)
    storage = pa.FixedSizeListArray.from_arrays(values, 6, mask=pa.array([False, True, False, False]))
    return TensorExtensionArray(pa.ExtensionArray.from_storage(dtype.pyarrow_dtype, storage), dtype=dtype)


def test_element_null_is_not_row_null(element_null_array):
    """Test that a missing element does not make the tensor missing."""
    assert element_null_array.isna().tolist() == [False, True, False, False]


def test_element_null_becomes_nan(element_null_array, stack):
    """Test that a missing element reads as NaN through every path, as numpy does."""
    assert np.isnan(element_null_array[0][0, 1])
    assert np.isnan(element_null_array.to_stack()[0, 0, 1])
    assert np.isnan(element_null_array.to_numpy()[0][0, 1])
    assert_array_equal(element_null_array[2], stack[2])
    assert_array_equal(element_null_array[1:][1:].to_stack()[0], stack[2])
    # nan != nan
    assert (element_null_array == element_null_array).tolist() == [False, pd.NA, True, True]


def test_element_null_int_upcast():
    """Test that integer tensors with a missing element are read as float."""
    dtype = TensorDtype(pa.fixed_shape_tensor(pa.int64(), [2, 3]))
    values = pa.array([0, None, 2, 3, 4, 5], type=pa.int64())
    storage = pa.FixedSizeListArray.from_arrays(values, 6)
    array = TensorExtensionArray(pa.ExtensionArray.from_storage(dtype.pyarrow_dtype, storage), dtype=dtype)
    assert array.to_stack().dtype == np.float64
    assert array[0].dtype == np.float64
    assert np.isnan(array[0][0, 1])


# dtype and Series construction #


def test_ext_array_dtype(array, dtype):
    """Test the dtype attribute."""
    assert array.dtype == dtype
    assert isinstance(array.dtype, TensorDtype)


def test_series_dtype(array, dtype):
    """Test that a Series holding the array has the tensor dtype."""
    series = pd.Series(array)
    assert series.dtype == dtype
    assert isinstance(series.array, TensorExtensionArray)
    assert series.array.equals(array)


def test_series_built_with_dtype(stack, dtype):
    """Test building a Series from tensors with the dtype given as an object and as a string."""
    series = pd.Series(list(stack), dtype=dtype)
    assert series.dtype == dtype
    assert_array_equal(series.iloc[2], stack[2])
    series = pd.Series([stack[0], None], dtype="tensor[double, (2, 3)]")
    assert series.dtype == dtype
    assert series.isna().tolist() == [False, True]


def test_series_built_raises(dtype):
    """Test that a Series cannot be built from tensors of the wrong shape."""
    with pytest.raises(ValueError):
        pd.Series([np.zeros((3, 2))], dtype=dtype)


# __getitem__ #


def test___getitem___with_integer(array, stack):
    """Test scalar access gives a read-only view of the right tensor, negative indices included."""
    result = array[1]
    assert isinstance(result, np.ndarray)
    assert result.shape == TENSOR_SHAPE
    assert_array_equal(result, stack[1])
    assert np.shares_memory(result, stack)
    assert not result.flags.writeable
    assert_array_equal(array[-1], stack[-1])


def test___getitem___with_integer_out_of_bounds(array):
    """Test that an out of bounds scalar index raises IndexError."""
    with pytest.raises(IndexError):
        array[4]


def test___getitem___missing(array_with_missing):
    """Test that a missing tensor is returned as pd.NA."""
    assert array_with_missing[1] is pd.NA


def test___getitem___with_slice(array, stack):
    """Test slicing returns a new array."""
    result = array[1:3]
    assert isinstance(result, TensorExtensionArray)
    assert len(result) == 2
    assert_array_equal(result.to_stack(), stack[1:3])
    assert_array_equal(array[::2].to_stack(), stack[::2])


def test___getitem___with_integer_ndarray(array, stack):
    """Test indexing with integer arrays and lists."""
    assert_array_equal(array[np.array([3, 0])].to_stack(), stack[[3, 0]])
    assert_array_equal(array[[3, 0, 0]].to_stack(), stack[[3, 0, 0]])
    assert_array_equal(array[np.array([1], dtype=np.uint8)].to_stack(), stack[[1]])


def test___getitem___with_boolean_ndarray(array, stack):
    """Test indexing with a boolean mask."""
    mask = np.array([True, False, True, False])
    assert_array_equal(array[mask].to_stack(), stack[mask])
    assert_array_equal(array[mask.tolist()].to_stack(), stack[mask])


def test___getitem___with_empty_ndarray(array):
    """Test indexing with an empty index array."""
    result = array[np.array([], dtype=int)]
    assert len(result) == 0
    assert result.dtype == array.dtype


def test___getitem___raises_for_invalid_ndarray_dtype(array):
    """Test that a float index array is rejected."""
    with pytest.raises(IndexError):
        array[np.array([0.5, 1.5])]


def test___getitem___with_ellipsis(array):
    """Test that Ellipsis returns the whole array."""
    assert array[...].equals(array)


def test___getitem___with_single_element_tuple(array, stack):
    """Test that a single-element tuple is unpacked."""
    assert_array_equal(array[(1,)], stack[1])
    assert_array_equal(array[(slice(1, 3),)].to_stack(), stack[1:3])


def test_series___getitem__(array, stack):
    """Test the Series indexers over a tensor column."""
    series = pd.Series(array, index=list("abcd"))
    assert_array_equal(series["b"], stack[1])
    assert_array_equal(series.iloc[1], stack[1])
    assert_array_equal(series.iloc[1:3].array.to_stack(), stack[1:3])
    assert_array_equal(series.iloc[[3, 0]].array.to_stack(), stack[[3, 0]])
    assert_array_equal(series[np.array([True, False, True, False])].array.to_stack(), stack[[0, 2]])
    assert_array_equal(series.loc[["d", "a"]].array.to_stack(), stack[[3, 0]])


# __setitem__ #


def test___setitem___single_tensor(array, stack):
    """Test assigning a single tensor to one position."""
    new = np.full(TENSOR_SHAPE, -1.0)
    array[1] = new
    assert_array_equal(array[1], new)
    assert_array_equal(array[0], stack[0])
    assert_array_equal(array[2], stack[2])


def test___setitem___with_negative_index(array, stack):
    """Test assigning with a negative index."""
    array[-1] = np.zeros(TENSOR_SHAPE)
    assert_array_equal(array[3], np.zeros(TENSOR_SHAPE))
    assert_array_equal(array[2], stack[2])


def test___setitem___with_tuple(array):
    """Test assigning with a single-element tuple key."""
    array[(1,)] = np.zeros(TENSOR_SHAPE)
    assert_array_equal(array[1], np.zeros(TENSOR_SHAPE))


@pytest.mark.parametrize(
    "key",
    [np.array([], dtype=int), [], slice(3, 1), slice(0, 0), np.zeros(4, dtype=bool)],
    ids=["empty_int_array", "empty_list", "empty_slice", "zero_slice", "all_false_mask"],
)
@pytest.mark.parametrize("value_kind", ["tensor", "missing", "empty_sequence"])
def test___setitem___with_empty_key(array, stack, key, value_kind):
    """Test that assigning to no positions is a no-op, for a tensor, a missing value or nothing."""
    if value_kind == "tensor":
        value = np.zeros(TENSOR_SHAPE)
    elif value_kind == "missing":
        value = None
    else:
        value = TensorExtensionArray.from_stack(stack[:0])
    array[key] = value
    assert_array_equal(array.to_stack(), stack)


def test___setitem___with_empty_key_still_validates_value(array):
    """Test that a sequence of the wrong length is rejected even when nothing is selected."""
    with pytest.raises(ValueError, match="Cannot set 0 elements from a sequence of length 2"):
        array[np.zeros(4, dtype=bool)] = [np.zeros(TENSOR_SHAPE), np.zeros(TENSOR_SHAPE)]


def test___setitem___single_tensor_to_all_rows(array):
    """Test broadcasting one tensor to every row."""
    array[:] = np.ones(TENSOR_SHAPE)
    assert_array_equal(array.to_stack(), np.ones((4, *TENSOR_SHAPE)))


def test___setitem___list_of_tensors(array, stack):
    """Test assigning a sequence of tensors to integer positions, out of order."""
    array[[3, 0]] = [np.zeros(TENSOR_SHAPE), np.ones(TENSOR_SHAPE)]
    assert_array_equal(array[3], np.zeros(TENSOR_SHAPE))
    assert_array_equal(array[0], np.ones(TENSOR_SHAPE))
    assert_array_equal(array[1], stack[1])


def test___setitem___with_duplicate_integer_keys(array):
    """Test that duplicate integer keys are handled like numpy: the value for each unique key is used."""
    array[[1, 1, 2]] = [np.zeros(TENSOR_SHAPE), np.zeros(TENSOR_SHAPE), np.ones(TENSOR_SHAPE)]
    assert_array_equal(array[1], np.zeros(TENSOR_SHAPE))
    assert_array_equal(array[2], np.ones(TENSOR_SHAPE))


def test___setitem___with_boolean_mask(array, stack):
    """Test assigning with a boolean mask, with a single tensor and with a sequence."""
    mask = np.array([True, False, False, True])
    array[mask] = np.zeros(TENSOR_SHAPE)
    assert_array_equal(array.to_stack()[mask], np.zeros((2, *TENSOR_SHAPE)))
    array[mask] = stack[[3, 0]]
    assert_array_equal(array[0], stack[3])
    assert_array_equal(array[3], stack[0])


def test___setitem___other_ext_array(array, stack):
    """Test assigning from another TensorExtensionArray, including one of another element type."""
    other = TensorExtensionArray.from_stack(stack[[3, 2]])
    array[[0, 1]] = other
    assert_array_equal(array.to_stack(), stack[[3, 2, 2, 3]])
    float32 = TensorExtensionArray.from_stack(np.ones((1, *TENSOR_SHAPE), dtype=np.float32))
    array[[0]] = float32
    assert array.dtype.value_type == pa.float64()
    assert_array_equal(array[0], np.ones(TENSOR_SHAPE))


def test___setitem___series_of_tensors(array, stack):
    """Test assigning from a Series of tensors."""
    array[[0, 1]] = pd.Series([stack[3], stack[2]], dtype=array.dtype)
    assert_array_equal(array.to_stack(), stack[[3, 2, 2, 3]])


def test___setitem___missing(array):
    """Test assigning None and pd.NA makes tensors missing."""
    array[0] = None
    array[np.array([False, False, True, False])] = pd.NA
    assert array.isna().tolist() == [True, False, True, False]
    assert array[0] is pd.NA


def test___setitem___raises_for_wrong_shape(array):
    """Test that a tensor of the wrong shape cannot be assigned."""
    with pytest.raises(ValueError, match="Expected a tensor of shape"):
        array[0] = np.zeros((3, 2))


@pytest.mark.parametrize("key", [[4], [0, 4], [-5]])
def test___setitem___raises_for_out_of_bounds_key(array, key):
    """Test that integer keys outside the array raise IndexError."""
    with pytest.raises(IndexError):
        array[key] = np.zeros(TENSOR_SHAPE)


def test___setitem___raises_for_length_mismatch(array):
    """Test that a sequence of the wrong length cannot be assigned."""
    with pytest.raises(ValueError, match="Cannot set 2 elements from a sequence of length 3"):
        array[[0, 1]] = [np.zeros(TENSOR_SHAPE)] * 3


def test___setitem___does_not_modify_other_views(array, stack):
    """Test that arrow immutability keeps copies independent."""
    copy = array.copy()
    array[0] = np.zeros(TENSOR_SHAPE)
    assert_array_equal(copy[0], stack[0])


def test_series___setitem__(array, stack):
    """Test assignment through a Series.

    A single tensor can be assigned to one label with ``series[label] = tensor``, and sequences
    of tensors, TensorExtensionArrays and missing values assign anywhere. Broadcasting one
    tensor to several positions, and ``loc``/``iloc`` with a single tensor, are not supported
    because pandas inspects the ndarray itself and treats it as a sequence; use ``series.array``
    for those.
    """
    series = pd.Series(array.copy())
    series[0] = np.ones(TENSOR_SHAPE)
    series.iloc[[1]] = TensorExtensionArray.from_stack(np.zeros((1, *TENSOR_SHAPE)))
    series[[2, 3]] = [stack[3], stack[2]]
    assert_array_equal(series.iloc[0], np.ones(TENSOR_SHAPE))
    assert_array_equal(series.iloc[1], np.zeros(TENSOR_SHAPE))
    assert_array_equal(series.iloc[2], stack[3])
    series[series.index == 3] = None
    series.array[[0, 2]] = np.full(TENSOR_SHAPE, 7.0)
    assert series.isna().tolist() == [False, False, False, True]
    assert_array_equal(series.iloc[2], np.full(TENSOR_SHAPE, 7.0))
    # The original array is untouched
    assert_array_equal(array.to_stack(), stack)


# isna #


def test_isna_when_none_na(array):
    """Test isna and _hasna with no missing values."""
    assert not array.isna().any()
    assert not array._hasna


def test_isna_when_all_na(dtype):
    """Test isna and _hasna when everything is missing."""
    array = TensorExtensionArray.from_sequence([None, None], dtype=dtype)
    assert array.isna().all()
    assert array._hasna


def test_isna_when_some_na(array_with_missing):
    """Test isna and _hasna with some missing values."""
    assert array_with_missing.isna().tolist() == [False, True, True, False]
    assert array_with_missing._hasna


# take #


def test_take(array, stack):
    """Test take with positive and negative indices."""
    assert_array_equal(array.take([3, 0, 0]).to_stack(), stack[[3, 0, 0]])
    assert_array_equal(array.take([-1, -4]).to_stack(), stack[[3, 0]])
    assert_array_equal(array.take(np.array([1, 2])).to_stack(), stack[1:3])
    assert len(array.take([])) == 0


def test_take_allow_fill(array, stack):
    """Test take with allow_fill: -1 gives missing, or the fill tensor."""
    result = array.take([0, -1, 3], allow_fill=True)
    assert result.isna().tolist() == [False, True, False]
    assert_array_equal(result[2], stack[3])
    result = array.take([0, -1], allow_fill=True, fill_value=np.zeros(TENSOR_SHAPE))
    assert not result.isna().any()
    assert_array_equal(result[1], np.zeros(TENSOR_SHAPE))
    # No fill needed
    assert_array_equal(array.take([1, 2], allow_fill=True).to_stack(), stack[1:3])


def test_take_allow_fill_raises_below_minus_one(array):
    """Test that with allow_fill only -1 is a valid negative index."""
    with pytest.raises(ValueError):
        array.take([0, -2], allow_fill=True)


def test_take_raises_for_empty_array_and_non_empty_index(array):
    """Test that taking from an empty array raises."""
    with pytest.raises(IndexError, match="non-empty take from the empty array"):
        array[:0].take([0])


@pytest.mark.parametrize("indices", [[4], [0, 4], [-5], [0, -5]])
def test_take_raises_for_out_of_bounds_index(array, indices):
    """Test that out of bounds indices, above or below, raise our IndexError rather than pyarrow's."""
    with pytest.raises(IndexError, match="out of bounds value in 'indices'"):
        array.take(indices)


# Misc ExtensionArray API #


def test_copy(array):
    """Test that copy gives an equal but independent array."""
    copy = array.copy()
    assert copy is not array
    assert copy.equals(array)
    copy[0] = None
    assert not array.isna().any()


def test__formatter_unboxed(array):
    """Test the unboxed formatter is repr."""
    assert array._formatter(boxed=False) is repr


def test__formatter_boxed(array, stack):
    """Test the boxed formatter shows small tensors in full and larger ones as a descriptor."""
    formatter = array._formatter(boxed=True)
    assert formatter(array[0]) == str(stack[0].tolist())
    assert formatter(pd.NA) == "<NA>"
    assert formatter(None) == "<NA>"
    big = np.zeros((8, 8), dtype=np.float32)
    assert big.size > TENSOR_FORMATTING_MAX_ELEMENTS
    assert formatter(big) == "[8×8] float32"
    assert _format_tensor(np.zeros(5, dtype=np.int16)) == "[0, 0, 0, 0, 0]"
    assert _format_tensor(np.zeros((3, 4, 5))) == "[3×4×5] float64"


def test_series_repr(array_with_missing, stack):
    """Test the text repr of a Series and a DataFrame with a tensor column."""
    text = repr(pd.Series(array_with_missing, name="t"))
    assert str(stack[0].tolist()) in text
    assert "<NA>" in text
    assert "tensor[double, (2, 3)]" in text
    big = TensorExtensionArray.from_stack(np.zeros((2, 8, 8)))
    assert "[8×8] float64" in repr(pd.DataFrame({"a": [1, 2], "t": big}))


def test_nbytes(array, stack):
    """Test nbytes accounts for the tensor data."""
    assert array.nbytes >= stack.nbytes


def test_pickability(array_with_missing):
    """Test pickling of the array and of a Series holding it."""
    assert pickle.loads(pickle.dumps(array_with_missing)).equals(array_with_missing)
    series = pd.Series(array_with_missing, name="t")
    assert_series_equal(pickle.loads(pickle.dumps(series)), series)


def test_pickle_slice_is_compact(dtype):
    """Test that pickling a slice does not serialize the whole parent buffer."""
    big = TensorExtensionArray.from_stack(np.zeros((10_000, 8, 8)))
    pickled = pickle.dumps(big[:10])
    assert len(pickled) < 10 * 8 * 8 * 8 * 2  # about the ten rows of float64, not all 5 MB
    assert pickle.loads(pickled).equals(big[:10])


def test_pickle_edge_cases(array):
    """Test pickling empty and multi-chunk arrays."""
    assert pickle.loads(pickle.dumps(array[:0])).equals(array[:0])
    chunked = TensorExtensionArray._concat_same_type([array[:2], array[2:]])
    assert pickle.loads(pickle.dumps(chunked)).equals(chunked)


def test__concat_same_type(array, array_with_missing, stack):
    """Test concatenation keeps the chunks and the dtype."""
    result = TensorExtensionArray._concat_same_type([array, array_with_missing])
    assert len(result) == 8
    assert result.num_chunks == 2
    assert result.dtype == array.dtype
    assert result.isna().tolist() == [False] * 4 + [False, True, True, False]
    assert_array_equal(result[:4].to_stack(), stack)


def test_equals(array, array_with_missing):
    """Test equals for equal, copied and different arrays."""
    assert array.equals(array)
    assert array.equals(array.copy())
    assert array.equals(TensorExtensionArray._concat_same_type([array[:2], array[2:]]))
    assert not array.equals(array_with_missing)
    assert not array.equals(array[:3])
    float32_dtype = TensorDtype(pa.fixed_shape_tensor(pa.float32(), list(TENSOR_SHAPE)))
    assert not array.equals(array.astype(float32_dtype))


def test_equals_when_other_is_different_type(array):
    """Test equals with something that is not a TensorExtensionArray."""
    assert not array.equals(array.to_stack())
    assert not array.equals(pd.Series(array))
    assert not array.equals(None)


def test_dropna(array_with_missing, stack):
    """Test dropna removes missing tensors."""
    result = array_with_missing.dropna()
    assert len(result) == 2
    assert_array_equal(result.to_stack(), stack[[0, 3]])


def test___iter__(array_with_missing, stack):
    """Test iteration yields tensors and pd.NA."""
    items = list(array_with_missing)
    assert [type(item).__name__ for item in items] == ["ndarray", "NAType", "NAType", "ndarray"]
    assert_array_equal(items[3], stack[3])


def test_to_numpy(array_with_missing, stack):
    """Test to_numpy gives an object array of tensors with na_value for missing ones."""
    result = array_with_missing.to_numpy()
    assert result.dtype == object
    assert result.shape == (4,)
    assert_array_equal(result[0], stack[0])
    assert result[1] is pd.NA
    assert array_with_missing.to_numpy(na_value=None)[1] is None
    assert not result[0].flags.writeable
    assert array_with_missing.to_numpy(copy=True)[0].flags.writeable


def test___array__(array_with_missing, stack):
    """Test numpy conversion gives the object array, also through a Series."""
    result = np.asarray(array_with_missing)
    assert result.dtype == object and result.shape == (4,)
    assert_array_equal(result[3], stack[3])
    assert np.asarray(pd.Series(array_with_missing)).shape == (4,)


def test_num_chunks_and_pa_array(array):
    """Test the pyarrow accessors."""
    assert array.num_chunks == 1
    assert isinstance(array.pa_array, pa.ChunkedArray)
    assert isinstance(array.pa_array.type, pa.FixedShapeTensorType)
    assert array.storage.type == array.dtype.storage_type
    chunked = TensorExtensionArray._concat_same_type([array[:1], array[1:]])
    assert chunked.num_chunks == 2
    assert chunked.storage.num_chunks == 2


# astype #


def test_astype_same_dtype(array):
    """Test astype to the same dtype copies or returns self."""
    assert array.astype(array.dtype, copy=False) is array
    result = array.astype(array.dtype)
    assert result is not array
    assert result.equals(array)


def test_astype_other_tensor_dtype(array, stack):
    """Test astype to another element type and to another shape with the same size."""
    float32_dtype = TensorDtype(pa.fixed_shape_tensor(pa.float32(), list(TENSOR_SHAPE)))
    result = array.astype(float32_dtype)
    assert result.dtype == float32_dtype
    assert_array_equal(result.to_stack(), stack.astype(np.float32))
    reshaped = array.astype(TensorDtype(pa.fixed_shape_tensor(pa.float64(), [3, 2])))
    assert reshaped[0].shape == (3, 2)
    assert_array_equal(reshaped[0], stack[0].reshape(3, 2))
    assert array.astype("tensor[float, (2, 3)]").dtype == float32_dtype


def test_astype_raises_for_size_mismatch(array):
    """Test astype to a tensor dtype with a different number of elements raises."""
    with pytest.raises(ValueError, match="6 elements"):
        array.astype(TensorDtype(pa.fixed_shape_tensor(pa.float64(), [4])))


def test_astype_to_pandas_arrow_dtype(array, dtype, stack):
    """Test astype to pd.ArrowDtype of the tensor type and of list types."""
    result = array.astype(dtype.to_pandas_arrow_dtype())
    assert isinstance(result, ArrowExtensionArray)
    assert result.dtype == dtype.to_pandas_arrow_dtype()
    result = array.astype(dtype.to_pandas_arrow_dtype(storage=True))
    assert result.dtype == pd.ArrowDtype(pa.list_(pa.float64(), 6))
    assert result[0] == stack[0].reshape(-1).tolist()
    result = array.astype(pd.ArrowDtype(pa.list_(pa.float32())))
    assert result.dtype == pd.ArrowDtype(pa.list_(pa.float32()))


def test_astype_to_object(array_with_missing, stack):
    """Test astype to object gives the object array."""
    result = array_with_missing.astype(object)
    assert result.dtype == object
    assert_array_equal(result[0], stack[0])
    assert result[1] is pd.NA


# Arrow interop #


def test___arrow_array__(array, dtype):
    """Test conversion to arrow gives the extension array by default."""
    result = pa.array(array)
    assert result.type == dtype.pyarrow_dtype
    assert isinstance(array.__arrow_array__(), pa.ChunkedArray)
    assert pa.table({"t": array}).schema.field("t").type == dtype.pyarrow_dtype


def test___arrow_array___with_type(array, stack):
    """Test conversion to arrow with an explicit type."""
    float32_type = pa.fixed_shape_tensor(pa.float32(), list(TENSOR_SHAPE))
    assert array.__arrow_array__(type=float32_type).type == float32_type
    as_list = pa.array(array, type=pa.list_(pa.float64(), 6))
    assert as_list.type == pa.list_(pa.float64(), 6)
    assert as_list[0].as_py() == stack[0].reshape(-1).tolist()
    assert pa.array(array, type=pa.large_list(pa.float64())).type == pa.large_list(pa.float64())


def test_table_from_pandas_and_parquet_round_trip(array_with_missing, dtype, stack):
    """Test that the tensor type survives pa.Table.from_pandas and a parquet round trip."""
    df = pd.DataFrame({"a": [1, 2, 3, 4], "t": array_with_missing})
    table = pa.Table.from_pandas(df)
    assert table.schema.field("t").type == dtype.pyarrow_dtype

    # The parquet round trip is only checked without missing values: writing a nullable
    # fixed_size_list is a pyarrow limitation fixed after 19.0 (see the array docstring)
    table = pa.Table.from_pandas(pd.DataFrame({"t": array_with_missing.dropna()}))
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)
    read = pq.read_table(buffer)
    assert read.schema.field("t").type == dtype.pyarrow_dtype
    back = TensorExtensionArray(read.column("t"))
    assert_array_equal(back.to_stack(), stack[[0, 3]])
    # And via __from_arrow__ with a types_mapper
    df_back = read.to_pandas(types_mapper=lambda t: dtype if isinstance(t, pa.FixedShapeTensorType) else None)
    assert df_back["t"].dtype == dtype


def test_from_arrow_ext_array_and_back(array, dtype):
    """Test conversion to and from pandas' ArrowExtensionArray, of the tensor and the storage type."""
    as_tensor = array.to_arrow_ext_array()
    assert as_tensor.dtype == dtype.to_pandas_arrow_dtype()
    assert TensorExtensionArray.from_arrow_ext_array(as_tensor).equals(array)
    as_storage = array.to_arrow_ext_array(storage=True)
    assert as_storage.dtype == dtype.to_pandas_arrow_dtype(storage=True)
    assert TensorExtensionArray.from_arrow_ext_array(as_storage, dtype=dtype).equals(array)


def test_dtype___from_arrow__(array, dtype, stack):
    """Test the dtype's __from_arrow__ hook with extension and storage arrays."""
    assert dtype.__from_arrow__(array.pa_array).equals(array)
    assert_array_equal(dtype.__from_arrow__(storage_array(stack)).to_stack(), stack)


# __eq__ #


def test___eq___same_values(array, stack):
    """Test elementwise equality with another array, a single tensor, a list and a stack."""
    result = array == array.copy()
    assert isinstance(result, pd.arrays.BooleanArray)
    assert result.all() and not result.isna().any()
    assert (array == array.take([1, 2, 3, 0])).tolist() == [False] * 4
    assert (array == stack[0]).tolist() == [True, False, False, False]
    assert (array == [stack[0], stack[1], stack[0], stack[3]]).tolist() == [True, True, False, True]
    assert (array == array.to_stack()).all()
    assert (array != array.copy()).sum() == 0


def test___eq___with_missing(array, array_with_missing):
    """Test that missing values give NA."""
    result = array == array_with_missing
    assert result.isna().tolist() == [False, True, True, False]
    assert bool(result[0]) and bool(result[3])
    assert (array == pd.NA).isna().all()
    assert (array == None).isna().all()  # noqa: E711


def test___eq___incomparable(array):
    """Test that anything not interpretable as tensors of this dtype compares False."""
    assert (array == 5).tolist() == [False] * 4
    assert (array == "abc").tolist() == [False] * 4
    assert (array == np.zeros((3, 2))).tolist() == [False] * 4
    float32_dtype = TensorDtype(pa.fixed_shape_tensor(pa.float32(), list(TENSOR_SHAPE)))
    assert (array == array.astype(float32_dtype)).tolist() == [False] * 4


def test___eq___raises_for_length_mismatch(array):
    """Test that arrays of different lengths cannot be compared."""
    with pytest.raises(ValueError, match="Lengths must match"):
        array == array[:2]  # noqa: B015


def test___eq___nan(dtype):
    """Test nan != nan, as in numpy."""
    array = TensorExtensionArray.from_sequence([np.full(TENSOR_SHAPE, np.nan)], dtype=dtype)
    assert (array == array).tolist() == [False]
    assert (array[:0] == array[:0]).tolist() == []


def test_series___eq__(array, array_with_missing, stack):
    """Test comparison through pandas Series."""
    result = pd.Series(array) == pd.Series(array_with_missing)
    assert str(result.dtype) == "boolean"
    assert result.isna().tolist() == [False, True, True, False]
    others = [stack[0], stack[1], stack[0], stack[3]]
    assert (pd.Series(array) == others).tolist() == [True, True, False, True]
    df = pd.DataFrame({"t": array})
    assert (df["t"] != df["t"]).sum() == 0
    # With a Series on the right, the array defers to pandas, which dispatches back with the values
    result = array == pd.Series(array_with_missing)
    assert isinstance(result, pd.Series)
    assert result.isna().tolist() == [False, True, True, False]


# Pandas integration #


def test_dataframe_operations(array, stack):
    """Test filtering, concat and column access on a DataFrame with a tensor column."""
    df = pd.DataFrame({"a": [1, 2, 3, 4], "t": array})
    assert df["t"].dtype == array.dtype
    sub = df[df["a"] > 2]
    assert len(sub) == 2
    assert_array_equal(sub["t"].iloc[0], stack[2])
    both = pd.concat([df, df], ignore_index=True)
    assert both["t"].dtype == array.dtype
    assert len(both) == 8
    assert isinstance(df.to_html(), str)


def test_series_apply_udf_argument(array, stack):
    """Test that a user function applied to a tensor Series receives the tensors."""
    result = pd.Series(array).apply(np.sum)
    assert_array_equal(result.to_numpy(), stack.sum(axis=(1, 2)))
    result = pd.Series(array).map(lambda t: t.shape)
    assert result.tolist() == [TENSOR_SHAPE] * 4
