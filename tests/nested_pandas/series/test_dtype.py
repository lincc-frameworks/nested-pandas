import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas.series.dtype import NestedDtype
from nested_pandas.series.ext_array import NestedExtensionArray
from nested_pandas.series.na import NA


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
    """Test that we can construct NestedDtype from pyarrow struct type."""
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
    """Test that we raise an error when constructing NestedDtype from invalid pyarrow type."""
    with pytest.raises(ValueError):
        NestedDtype(pyarrow_dtype)


def test_to_pandas_pyarrow_dtype():
    """Test that NestedDtype.to_pandas_pyarrow_dtype() returns the correct pyarrow struct type."""
    dtype = NestedDtype.from_fields({"a": pa.int64(), "b": pa.int64()})
    assert dtype.to_pandas_pyarrow_dtype() == pd.ArrowDtype(
        pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))])
    )


def test_from_fields():
    """Test NestedDtype.from_fields()."""
    fields = {"a": pa.int64(), "b": pa.float64()}
    dtype = NestedDtype.from_fields(fields)
    assert dtype.pyarrow_dtype == pa.struct(
        [pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]
    )


def test_na_value():
    """Test that NestedDtype.na_value is a singleton instance of NAType."""
    dtype = NestedDtype(pa.struct([pa.field("a", pa.list_(pa.int64()))]))
    assert dtype.na_value is NA


def test_fields():
    """Test NestedDtype.fields property"""
    dtype = NestedDtype(
        pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))])
    )
    assert dtype.fields == {"a": pa.int64(), "b": pa.float64()}


def test_field_names():
    """Test NestedDtype.field_names property"""
    dtype = NestedDtype(
        pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))])
    )
    assert dtype.field_names == ["a", "b"]


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
    """Test that dtype.name is consistent with dtype.construct_from_string(dtype.name)."""
    dtype = NestedDtype.from_fields(fields)
    assert dtype == NestedDtype.construct_from_string(dtype.name)


def test_construct_array_type():
    """Test that NestedDtype.construct_array_type() returns NestedExtensionArray."""
    assert NestedDtype.construct_array_type() is NestedExtensionArray
