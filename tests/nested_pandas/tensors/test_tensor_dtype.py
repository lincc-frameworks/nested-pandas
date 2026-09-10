import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.tests.extension import base

from nested_pandas import TensorDtype


class _NotATensorType(pa.ExtensionType):
    """An arrow extension type that is not a tensor, to check we test for the specific type."""

    def __init__(self):
        super().__init__(pa.list_(pa.float64(), 6), "nested_pandas.test.not_a_tensor")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls()


@pytest.mark.parametrize(
    "pyarrow_dtype",
    [
        pa.fixed_shape_tensor(pa.float64(), [3]),
        pa.fixed_shape_tensor(pa.float32(), [2, 3]),
        pa.fixed_shape_tensor(pa.int16(), [4, 5, 6]),
        pa.fixed_shape_tensor(pa.bool_(), [8, 8]),
        pa.fixed_shape_tensor(pa.timestamp("ns"), [2]),
        pa.fixed_shape_tensor(pa.float64(), [2, 3], dim_names=["y", "x"]),
    ],
)
def test_from_pyarrow_dtype(pyarrow_dtype):
    """Test that we can construct TensorDtype from pyarrow fixed_shape_tensor type."""
    dtype = TensorDtype(pyarrow_dtype)
    assert dtype.pyarrow_dtype == pyarrow_dtype
    assert dtype.pyarrow_dtype is pyarrow_dtype


def test_init_from_pandas_arrow_dtype():
    """Test that we can construct TensorDtype from pandas.ArrowDtype in __init__."""
    pyarrow_dtype = pa.fixed_shape_tensor(pa.float64(), [2, 3])
    dtype = TensorDtype(pd.ArrowDtype(pyarrow_dtype))
    assert dtype.pyarrow_dtype == pyarrow_dtype


@pytest.mark.parametrize(
    "pyarrow_dtype",
    [
        pa.int64(),
        pa.list_(pa.float64()),
        # The storage type of a tensor is not a tensor
        pa.list_(pa.float64(), 6),
        pa.struct([pa.field("a", pa.list_(pa.int64()))]),
        pd.ArrowDtype(pa.list_(pa.float64(), 6)),
        # An extension type with the same storage as a tensor is still not a tensor
        _NotATensorType(),
        "tensor[double, (2, 3)]",
        (2, 3),
        None,
    ],
)
def test_from_pyarrow_dtype_raises(pyarrow_dtype):
    """Test that we raise a TypeError when constructing TensorDtype from anything but a tensor type."""
    with pytest.raises(TypeError):
        TensorDtype(pyarrow_dtype)


def test_permutation_raises():
    """Test that tensor types with a non-trivial permutation are rejected."""
    pyarrow_dtype = pa.fixed_shape_tensor(pa.float64(), [2, 3], permutation=[1, 0])
    with pytest.raises(NotImplementedError):
        TensorDtype(pyarrow_dtype)


@pytest.mark.parametrize(
    "shape,dim_names",
    [
        ([5], None),
        ([2, 3], None),
        ([2, 3, 4], ["z", "y", "x"]),
    ],
)
def test_identity_permutation_is_normalized(shape, dim_names):
    """Test that an identity permutation is accepted and dropped, so the dtype equals and hashes as if
    it had none."""
    permutation = list(range(len(shape)))
    with_permutation = pa.fixed_shape_tensor(
        pa.float64(), shape, dim_names=dim_names, permutation=permutation
    )
    without_permutation = pa.fixed_shape_tensor(pa.float64(), shape, dim_names=dim_names)
    # pyarrow itself compares these equal but hashes them differently
    assert with_permutation == without_permutation
    assert hash(with_permutation) != hash(without_permutation)

    dtype = TensorDtype(with_permutation)
    expected = TensorDtype(without_permutation)
    assert dtype.pyarrow_dtype == without_permutation
    assert dtype.pyarrow_dtype.permutation is None
    assert dtype.dim_names == expected.dim_names
    assert dtype == expected
    assert hash(dtype) == hash(expected)
    assert dtype.name == expected.name


def test_from_numpy_ndarray_type():
    """Test that the type pyarrow assigns to a tensor array built from a C-ordered numpy array is accepted.

    pyarrow sets an identity permutation on these, so this is the type of any tensor column written
    through pyarrow's own numpy conversion.
    """
    array = pa.FixedShapeTensorArray.from_numpy_ndarray(np.zeros((4, 2, 3)))
    assert array.type.permutation is not None
    dtype = TensorDtype(array.type)
    assert dtype == TensorDtype(pa.fixed_shape_tensor(pa.float64(), [2, 3]))


def test_properties():
    """Test the accessors for the tensor type parameters."""
    dtype = TensorDtype(pa.fixed_shape_tensor(pa.float32(), [2, 3, 4], dim_names=["z", "y", "x"]))
    assert dtype.value_type == pa.float32()
    assert dtype.shape == (2, 3, 4)
    assert isinstance(dtype.shape, tuple)
    assert dtype.ndim == 3
    assert dtype.size == 24
    assert dtype.dim_names == ("z", "y", "x")
    assert dtype.storage_type == pa.list_(pa.float32(), 24)


def test_properties_no_dim_names():
    """Test that dim_names is None when not set."""
    dtype = TensorDtype(pa.fixed_shape_tensor(pa.int8(), [5]))
    assert dtype.dim_names is None
    assert dtype.shape == (5,)
    assert dtype.ndim == 1
    assert dtype.size == 5


def test_na_value():
    """Test that TensorDtype.na_value is pd.NA."""
    dtype = TensorDtype(pa.fixed_shape_tensor(pa.float64(), [2]))
    assert dtype.na_value is pd.NA


def test_type():
    """Test that the element type is np.ndarray."""
    dtype = TensorDtype(pa.fixed_shape_tensor(pa.float64(), [2]))
    assert dtype.type is np.ndarray


def test_kind():
    """Test that the dtype kind is object, as for other non-numpy extension dtypes."""
    dtype = TensorDtype(pa.fixed_shape_tensor(pa.float64(), [2]))
    assert dtype.kind == "O"


@pytest.mark.parametrize(
    "pyarrow_dtype,name",
    [
        (pa.fixed_shape_tensor(pa.float64(), [3]), "tensor[double, (3,)]"),
        (pa.fixed_shape_tensor(pa.float32(), [2, 3]), "tensor[float, (2, 3)]"),
        (pa.fixed_shape_tensor(pa.int16(), [4, 5, 6]), "tensor[int16, (4, 5, 6)]"),
        (pa.fixed_shape_tensor(pa.bool_(), [8, 8]), "tensor[bool, (8, 8)]"),
        (pa.fixed_shape_tensor(pa.timestamp("ns"), [2]), "tensor[timestamp[ns], (2,)]"),
        (
            pa.fixed_shape_tensor(pa.float64(), [2, 3], dim_names=["y", "x"]),
            "tensor[double, (2, 3), dim_names=[y, x]]",
        ),
        (pa.fixed_shape_tensor(pa.uint8(), [1], dim_names=["a"]), "tensor[uint8, (1,), dim_names=[a]]"),
    ],
)
def test_name(pyarrow_dtype, name):
    """Test the string representation of the dtype."""
    dtype = TensorDtype(pyarrow_dtype)
    assert dtype.name == name
    assert str(dtype) == name
    assert repr(dtype) == name


def test_name_cannot_be_set():
    """Test that the name attribute is read-only."""
    dtype = TensorDtype(pa.fixed_shape_tensor(pa.float64(), [2]))
    with pytest.raises(TypeError):
        dtype.name = "something"


@pytest.mark.parametrize(
    "pyarrow_dtype",
    [
        pa.fixed_shape_tensor(pa.float64(), [3]),
        pa.fixed_shape_tensor(pa.float32(), [2, 3]),
        pa.fixed_shape_tensor(pa.int16(), [4, 5, 6]),
        pa.fixed_shape_tensor(pa.bool_(), [8, 8]),
        pa.fixed_shape_tensor(pa.timestamp("ns"), [2]),
        pa.fixed_shape_tensor(pa.timestamp("ms", tz="UTC"), [2]),
        pa.fixed_shape_tensor(pa.float64(), [2, 3], dim_names=["y", "x"]),
        pa.fixed_shape_tensor(pa.string(), [2], dim_names=["with space"]),
    ],
)
def test_construct_from_string_round_trip(pyarrow_dtype):
    """Test that the dtype can be reconstructed from its own name."""
    dtype = TensorDtype(pyarrow_dtype)
    assert TensorDtype.construct_from_string(dtype.name) == dtype
    assert pd.api.types.pandas_dtype(str(dtype)) == dtype


def test_construct_from_string_is_registered():
    """Test that pandas resolves tensor dtype strings through the registry."""
    dtype = pd.api.types.pandas_dtype("tensor[double, (2, 3)]")
    assert isinstance(dtype, TensorDtype)
    assert dtype == TensorDtype(pa.fixed_shape_tensor(pa.float64(), [2, 3]))


def test_construct_from_string_aliases():
    """Test that pyarrow type aliases are accepted for the element type."""
    assert TensorDtype.construct_from_string("tensor[float64, (2,)]").value_type == pa.float64()
    assert TensorDtype.construct_from_string("tensor[f8, (2,)]").value_type == pa.float64()
    assert TensorDtype.construct_from_string("tensor[float, (2,)]").value_type == pa.float32()
    assert TensorDtype.construct_from_string("tensor[halffloat, (2,)]").value_type == pa.float16()
    assert TensorDtype.construct_from_string("tensor[string, (2,)]").value_type == pa.string()
    # Parametric temporal types go through pandas' ArrowDtype parsing
    dtype = TensorDtype.construct_from_string("tensor[timestamp[ms, tz=UTC], (2,)]")
    assert dtype.value_type == pa.timestamp("ms", tz="UTC")


@pytest.mark.parametrize(
    "string",
    [
        "nested<a: [int64]>",
        "tensor",
        "tensor[]",
        "tensor[double]",
        "tensor[(2, 3)]",
        "tensor[double, 2, 3]",
        "tensor[double, [2, 3]]",
        "tensor[double, (2, 3)",
        "tensor[double,(2, 3)]",
        "tensor[double, (2, 3), permutation=[1, 0]]",
        "tensor[double, (2, 3), dim_names=(y, x)]",
        "tensor[double, (2, 3), unknown=[1]]",
        "tensor[not_a_type, (2, 3)]",
        "tensor[decimal128(10, 2), (2, 3)]",
        "tensor[list<item: int64>, (2, 3)]",
        # dim_names must match the number of dimensions
        "tensor[double, (2, 3), dim_names=[x]]",
        "int64",
        "int64[pyarrow]",
        "",
    ],
)
def test_construct_from_string_raises(string):
    """Test that invalid strings raise TypeError, so pandas can try the next registered dtype."""
    with pytest.raises(TypeError):
        TensorDtype.construct_from_string(string)


def test_construct_from_string_error_explains_format():
    """Test that the error for an unrecognized string names the offending string and shows the format."""
    match = r"Cannot construct a 'TensorDtype' from 'tensor\[double\]'"
    with pytest.raises(TypeError, match=match) as excinfo:
        TensorDtype.construct_from_string("tensor[double]")
    assert "tensor[<element type>, (<shape>)]" in str(excinfo.value)
    assert "tensor[double, (2, 3)]" in str(excinfo.value)


@pytest.mark.parametrize("not_a_string", [0, None, pa.fixed_shape_tensor(pa.float64(), [2]), ["tensor"]])
def test_construct_from_string_not_a_string_raises(not_a_string):
    """Test that non-string input raises TypeError, as required by pandas."""
    with pytest.raises(TypeError):
        TensorDtype.construct_from_string(not_a_string)


def test_eq_and_hash():
    """Test equality and hashing of TensorDtype."""
    dtype = TensorDtype(pa.fixed_shape_tensor(pa.float64(), [2, 3]))
    same = TensorDtype(pa.fixed_shape_tensor(pa.float64(), [2, 3]))
    assert dtype == same
    assert hash(dtype) == hash(same)
    assert dtype == "tensor[double, (2, 3)]"

    # Same number of elements, different shape
    assert dtype != TensorDtype(pa.fixed_shape_tensor(pa.float64(), [3, 2]))
    assert dtype != TensorDtype(pa.fixed_shape_tensor(pa.float64(), [6]))
    # Different element type
    assert dtype != TensorDtype(pa.fixed_shape_tensor(pa.float32(), [2, 3]))
    # dim_names are part of the identity
    assert dtype != TensorDtype(pa.fixed_shape_tensor(pa.float64(), [2, 3], dim_names=["y", "x"]))
    # Not a tensor
    assert dtype != pd.ArrowDtype(pa.fixed_shape_tensor(pa.float64(), [2, 3]))
    assert dtype != pd.ArrowDtype(pa.list_(pa.float64(), 6))
    assert dtype != "tensor[double, (3, 2)]"
    assert dtype != "int64"
    assert dtype != np.dtype(object)


def test_is_dtype():
    """Test the pandas is_dtype classmethod with instances and strings."""
    dtype = TensorDtype(pa.fixed_shape_tensor(pa.float64(), [2, 3]))
    assert TensorDtype.is_dtype(dtype)
    assert TensorDtype.is_dtype("tensor[double, (2, 3)]")
    assert not TensorDtype.is_dtype("int64")
    assert not TensorDtype.is_dtype(pd.ArrowDtype(pa.list_(pa.float64(), 6)))
    assert not TensorDtype.is_dtype(np.dtype(float))


def test_common_dtype():
    """Test that equal tensor dtypes have a common dtype and different ones fall back to object."""
    dtype = TensorDtype(pa.fixed_shape_tensor(pa.float64(), [2, 3]))
    assert find_common_type([dtype, dtype]) == dtype
    other = TensorDtype(pa.fixed_shape_tensor(pa.float32(), [2, 3]))
    assert find_common_type([dtype, other]) == np.dtype(object)
    assert find_common_type([dtype, np.dtype(float)]) == np.dtype(object)


def test_to_pandas_arrow_dtype():
    """Test conversion to pd.ArrowDtype of the tensor type and of its storage type."""
    pyarrow_dtype = pa.fixed_shape_tensor(pa.float64(), [2, 3])
    dtype = TensorDtype(pyarrow_dtype)
    assert dtype.to_pandas_arrow_dtype() == pd.ArrowDtype(pyarrow_dtype)
    assert dtype.to_pandas_arrow_dtype(storage=True) == pd.ArrowDtype(pa.list_(pa.float64(), 6))
    # And back
    assert TensorDtype(dtype.to_pandas_arrow_dtype()) == dtype


def test_pickle():
    """Test that the dtype survives a pickle round trip."""
    dtype = TensorDtype(pa.fixed_shape_tensor(pa.float64(), [2, 3], dim_names=["y", "x"]))
    assert pickle.loads(pickle.dumps(dtype)) == dtype


class TestPandasBaseDtype(base.BaseDtypeTests):
    """Run pandas' extension dtype conformance suite against TensorDtype.

    The ``dtype``, ``data``, ``data_missing`` and ``skipna`` fixtures it uses
    come from ``conftest.py``.
    """
