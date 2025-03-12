import pandas as pd
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
    """Test that we can construct NestedDtype from pyarrow struct type."""
    dtype1 = NestedDtype(pyarrow_dtype)
    dtype2 = NestedDtype(pd.ArrowDtype(pyarrow_dtype))
    assert dtype1.pyarrow_dtype == dtype2.pyarrow_dtype == pyarrow_dtype


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
    with pytest.raises(ValueError):
        NestedDtype(pd.ArrowDtype(pyarrow_dtype))


def test_to_pandas_arrow_dtype():
    """Test that NestedDtype.to_pandas_arrow_dtype() returns the correct pyarrow struct type."""
    dtype = NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()})
    assert dtype.to_pandas_arrow_dtype() == pd.ArrowDtype(
        pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))])
    )


def test_from_fields():
    """Test NestedDtype.from_fields()."""
    fields1 = {"a": pa.int64(), "b": pd.ArrowDtype(pa.float64())}
    dtype1 = NestedDtype.from_fields(fields1)
    assert dtype1.pyarrow_dtype == pa.struct(
        [pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]
    )

    fields2 = {"x": pa.string(), "y": pa.bool_(), "nested": dtype1}
    dtype2 = NestedDtype.from_fields(fields2)
    assert dtype2 == NestedDtype(
        pa.struct(
            [
                pa.field("x", pa.list_(pa.string())),
                pa.field("y", pa.list_(pa.bool_())),
                pa.field("nested", pa.list_(dtype1.pyarrow_dtype)),
            ]
        ),
        inner_dtypes={"nested": dtype1},
    )


def test_na_value():
    """Test that NestedDtype.na_value is a singleton instance of NAType."""
    dtype = NestedDtype(pa.struct([pa.field("a", pa.list_(pa.int64()))]))
    assert dtype.na_value is pd.NA


def test_fields():
    """Test NestedDtype.fields property"""
    dtype1 = NestedDtype(
        pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))])
    )
    assert dtype1.fields == {"a": pd.ArrowDtype(pa.int64()), "b": pd.ArrowDtype(pa.float64())}

    dtype2 = NestedDtype(
        pa.struct(
            [
                pa.field("x", pa.list_(pa.float64())),
                pa.field("y", pa.list_(pa.string())),
                pa.field("nested", pa.list_(dtype1.pyarrow_dtype)),
            ]
        ),
        inner_dtypes={"x": pd.Float64Dtype(), "nested": dtype1},
    )
    assert dtype2.fields == {"x": pd.Float64Dtype(), "y": pd.ArrowDtype(pa.string()), "nested": dtype1}

    # field name missmatch
    with pytest.raises(ValueError):
        NestedDtype(pa.struct([pa.field("a", pa.list_(pa.int64()))]), inner_dtypes={"xyz": pa.int64()})

    # element type is not compatible with inner dtype
    with pytest.raises(TypeError):
        NestedDtype(pa.struct([pa.field("a", pa.list_(pa.int64()))]), inner_dtypes={"a": pa.bool_()})


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


@pytest.mark.parametrize(
    "s",
    [
        "float",  # not a nested type
        "nested(f: [int64])",  # must be <> instead
        "ts<in64>",  # 'ts' was a previous name, now we use 'nested'
        "nested",  # no type specified
        "nested<a: [int64]",  # missed closing bracket
        "nested<>",  # no field specified
        "nested<int64>",  # no field name specified
        "nested<[int64]>",  # no field name specified
        "nested<a:[int64]>",  # separator must be ": " with space
        "nested<a: [int64],b: [float32]>",  # separator must be ", " with space
        "nested<a: int64>",  # missed [] - nested list
        "nested<a: [complex64]>",  # not an arrow type
        "nested<a: [list<item: double>]>",  # complex arrow types are not supported
    ],
)
def test_construct_from_string_raises(s):
    """Test that we raise an error when constructing NestedDtype from invalid string."""
    with pytest.raises(TypeError):
        NestedDtype.construct_from_string(s)


def test_construct_array_type():
    """Test that NestedDtype.construct_array_type() returns NestedExtensionArray."""
    assert NestedDtype.construct_array_type() is NestedExtensionArray
