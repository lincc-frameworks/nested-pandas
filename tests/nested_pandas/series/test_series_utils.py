import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas import NestedDtype
from nested_pandas.series.utils import (
    align_chunked_struct_list_offsets,
    align_struct_list_offsets,
    nested_types_mapper,
    struct_field_names,
    transpose_list_struct_array,
    transpose_list_struct_scalar,
    transpose_list_struct_type,
    transpose_struct_list_array,
    transpose_struct_list_type,
    validate_struct_list_type,
)


def test_align_struct_list_offsets():
    """Test align_struct_list_offsets function."""
    # Raises for wrong types
    with pytest.raises(ValueError):
        align_struct_list_offsets(pa.array([], type=pa.int64()))
    with pytest.raises(ValueError):
        align_struct_list_offsets(pa.array([], type=pa.list_(pa.int64())))

    # Raises if one of the fields is not a ListArray
    with pytest.raises(ValueError):
        align_struct_list_offsets(
            pa.StructArray.from_arrays([pa.array([[1, 2], [3, 4, 5]]), pa.array([1, 2])], ["a", "b"])
        )

    # Raises for mismatched lengths
    with pytest.raises(ValueError):
        align_struct_list_offsets(
            pa.StructArray.from_arrays(
                [pa.array([[1, 2], [3, 4, 5]]), pa.array([[1, 2, 3], [4, 5]])], ["a", "b"]
            )
        )

    input_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([[1, 2], [3, 4], [], [5, 6, 7]]),
            pa.array([["x", "y"], ["y", "x"], [], ["d", "e", "f"]]),
        ],
        names=["a", "b"],
    )
    assert align_struct_list_offsets(input_array) is input_array

    a = pa.array([[0, 0, 0], [1, 2], [3, 4], [], [5, 6, 7]])[1:]
    assert a.offsets[0].as_py() == 3
    b = pa.array([["x", "y"], ["y", "x"], [], ["d", "e", "f"]])
    assert b.offsets[0].as_py() == 0
    input_array = pa.StructArray.from_arrays(
        arrays=[a, b],
        names=["a", "b"],
    )
    aligned_array = align_struct_list_offsets(input_array)
    assert aligned_array is not input_array
    assert aligned_array.equals(input_array)


def test_align_chunked_struct_list_offsets():
    """Test align_chunked_struct_list_offsets function."""
    # Input is an array, output is chunked array
    a = pa.array([[1, 2], [3, 4], [], [5, 6, 7]])
    b = pa.array([["x", "y"], ["y", "x"], [], ["d", "e", "f"]])
    input_array = pa.StructArray.from_arrays(
        arrays=[a, b],
        names=["a", "b"],
    )
    output_array = align_chunked_struct_list_offsets(input_array)
    assert isinstance(output_array, pa.ChunkedArray)
    assert output_array.equals(pa.chunked_array([input_array]))

    # Input is an "aligned" chunked array
    input_array = pa.chunked_array(
        [
            pa.StructArray.from_arrays(
                arrays=[a, b],
                names=["a", "b"],
            )
        ]
        * 2
    )
    output_array = align_chunked_struct_list_offsets(input_array)
    assert output_array.equals(input_array)

    # Input is an "aligned" chunked array, but offsets do not start with zero
    a = pa.array([[0, 0, 0], [1, 2], [3, 4], [], [5, 6, 7]])[1:]
    b = pa.array([["a", "a", "a", "a"], ["x", "y"], ["y", "x"], [], ["d", "e", "f"]])[1:]
    input_array = pa.chunked_array(
        [
            pa.StructArray.from_arrays(
                arrays=[a, b],
                names=["a", "b"],
            )
        ]
        * 3
    )
    output_array = align_chunked_struct_list_offsets(input_array)
    assert output_array.equals(input_array)

    # Input is a "non-aligned" chunked array
    a = pa.array([[0, 0, 0], [1, 2], [3, 4], [], [5, 6, 7]])[1:]
    b = pa.array([["x", "y"], ["y", "x"], [], ["d", "e", "f"]])
    input_array = pa.chunked_array(
        [
            pa.StructArray.from_arrays(
                arrays=[a, b],
                names=["a", "b"],
            )
        ]
        * 4
    )
    output_array = align_chunked_struct_list_offsets(input_array)
    assert output_array.equals(input_array)


def test_validate_struct_list_type():
    """Test validate_struct_list_type function."""
    with pytest.raises(ValueError):
        validate_struct_list_type(pa.float64())

    with pytest.raises(ValueError):
        validate_struct_list_type(pa.list_(pa.struct({"a": pa.int64()})))

    with pytest.raises(ValueError):
        validate_struct_list_type(pa.struct({"a": pa.float64()}))

    with pytest.raises(ValueError):
        validate_struct_list_type(pa.struct({"a": pa.list_(pa.float64()), "b": pa.float64()}))

    assert (
        validate_struct_list_type(pa.struct({"a": pa.list_(pa.float64()), "b": pa.list_(pa.float64())}))
        is None
    )


def test_transpose_struct_list_type():
    """Test transpose_struct_list_type function."""
    # Raises for wrong types
    with pytest.raises(ValueError):
        transpose_struct_list_type(pa.int64())
    with pytest.raises(ValueError):
        transpose_struct_list_type(pa.list_(pa.int64()))

    # Raises if one of the fields is not a ListType
    with pytest.raises(ValueError):
        transpose_struct_list_type(pa.struct([("a", pa.int64()), ("b", pa.int64())]))

    input_type = pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.string()))])
    expected_output = pa.list_(pa.struct([("a", pa.int64()), ("b", pa.string())]))
    assert transpose_struct_list_type(input_type) == expected_output


def test_transpose_list_struct_type():
    """Test transpose_list_struct_type function."""
    # Raises for wrong types
    with pytest.raises(ValueError):
        transpose_list_struct_type(pa.int64())
    with pytest.raises(ValueError):
        transpose_list_struct_type(pa.struct([("a", pa.int64()), ("b", pa.int64())]))

    input_type = pa.list_(pa.struct([("a", pa.int64()), ("b", pa.string())]))
    expected_output = pa.struct([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.string()))])
    assert transpose_list_struct_type(input_type) == expected_output


def test_transpose_struct_list_array():
    """Test transpose_struct_list_array function."""
    input_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([[1, 2], [3, 4], [], [5, 6, 7]]),
            pa.array([["x", "y"], ["y", "x"], [], ["d", "e", "f"]]),
        ],
        names=["a", "b"],
    )
    desired = pa.array(
        [
            [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
            [{"a": 3, "b": "y"}, {"a": 4, "b": "x"}],
            [],
            [{"a": 5, "b": "d"}, {"a": 6, "b": "e"}, {"a": 7, "b": "f"}],
        ]
    )
    actual = transpose_struct_list_array(input_array)
    assert actual == desired


def test_transpose_list_struct_array():
    """Test transpose_list_struct_array function."""
    input_array = pa.array(
        [
            [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
            [{"a": 3, "b": "y"}, {"a": 4, "b": "x"}],
            [],
            [{"a": 5, "b": "d"}, {"a": 6, "b": "e"}, {"a": 7, "b": "f"}],
        ]
    )
    desired = pa.StructArray.from_arrays(
        arrays=[
            pa.array([[1, 2], [3, 4], [], [5, 6, 7]]),
            pa.array([["x", "y"], ["y", "x"], [], ["d", "e", "f"]]),
        ],
        names=["a", "b"],
    )
    actual = transpose_list_struct_array(input_array)
    assert actual == desired


def test_transpose_list_struct_scalar():
    """Test transpose_list_struct_scalar function."""
    input_scalar = pa.scalar([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    desired = pa.scalar({"a": [1, 2], "b": ["x", "y"]})
    actual = transpose_list_struct_scalar(input_scalar)
    assert actual == desired


def test_struct_field_names():
    """Test struct_field_names and guard against requirement bumps."""

    # Otherwise, validate the shim works as expected (for pyarrow<=17 requirement)
    t = pa.struct(
        [
            pa.field("a", pa.list_(pa.int64())),
            pa.field("b", pa.list_(pa.float64())),
            pa.field("c", pa.list_(pa.string())),
        ]
    )
    # Ensure we get names in the correct order
    assert struct_field_names(t) == ["a", "b", "c"]


@pytest.mark.parametrize(
    "pa_type,is_nested",
    [
        (pa.float64(), False),
        (pa.list_(pa.float64()), False),
        (pa.list_(pa.struct([("a", pa.float64()), ("b", pa.float64())])), True),
    ],
)
def test_nested_types_mapper(pa_type, is_nested):
    """Test nested_types_mapper function."""
    dtype = nested_types_mapper(pa_type)
    if is_nested:
        assert isinstance(dtype, NestedDtype)
        assert dtype.list_struct_pa_dtype == pa_type
    else:
        assert isinstance(dtype, pd.ArrowDtype)
        assert dtype.pyarrow_dtype == pa_type
