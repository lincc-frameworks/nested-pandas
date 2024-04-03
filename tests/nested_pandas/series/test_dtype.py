import pyarrow as pa
import pytest

from nested_pandas.series.dtype import NestedDtype
from nested_pandas.series.ext_array import NestedExtensionArray


@pytest.mark.parametrize(
    "pyarrow_dtype",
    [
        pa.struct([pa.field("a", pa.list_(pa.int64()))]),
        pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
        pa.struct(
            [
                pa.field("a", pa.list_(pa.int64())),
                pa.field("b", pa.list_(pa.struct([pa.field("c", pa.int64())]))),
            ]
        ),
    ],
)
def test_from_pyarrow_dtype(pyarrow_dtype):
    dtype = NestedDtype(pyarrow_dtype)
    assert dtype.pyarrow_dtype == pyarrow_dtype


@pytest.mark.parametrize(
    "pyarrow_dtype",
    [
        pa.int64(),
        pa.list_(pa.int64()),
        pa.list_(pa.struct([pa.field("a", pa.int64())])),
        pa.struct([pa.field("a", pa.int64())]),
        pa.struct([pa.field("a", pa.int64()), pa.field("b", pa.float64())]),
        pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.float64())]),
    ],
)
def test_from_pyarrow_dtype_raises(pyarrow_dtype):
    with pytest.raises(ValueError):
        NestedDtype(pyarrow_dtype)


def test_from_fields():
    fields = {"a": pa.int64(), "b": pa.float64()}
    dtype = NestedDtype.from_fields(fields)
    assert dtype.pyarrow_dtype == pa.struct(
        [pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"a": pa.int64(), "b": pa.float64()},
        {"a": pa.int64(), "b": pa.float64(), "c": pa.int64()},
        {"a": pa.string(), "b": pa.float64()},
        # Nested / parametric types are not implemented.
        # {"a": pa.list_(pa.int64()), "b": pa.float64()},
        # {"a": pa.list_(pa.int64()), "b": pa.list_(pa.string())},
        # {"a": pa.struct([pa.field("a", pa.int64())]), "b": pa.list_(pa.int64())},
    ],
)
def test_name_vs_construct_from_string(fields):
    dtype = NestedDtype.from_fields(fields)
    assert dtype == NestedDtype.construct_from_string(dtype.name)


def test_construct_array_type():
    assert NestedDtype.construct_array_type() is NestedExtensionArray
