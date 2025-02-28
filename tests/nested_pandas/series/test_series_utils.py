import pyarrow as pa
import pytest
from nested_pandas.series.utils import (
    transpose_list_struct_array,
    transpose_list_struct_type,
    transpose_struct_list_array,
    transpose_struct_list_type,
    validate_struct_list_array_for_equal_lengths,
)


def test_validate_struct_list_array_for_equal_lengths():
    """Test validate_struct_list_array_for_equal_lengths function."""
    # Raises for wrong types
    with pytest.raises(ValueError):
        validate_struct_list_array_for_equal_lengths(pa.array([], type=pa.int64()))
    with pytest.raises(ValueError):
        validate_struct_list_array_for_equal_lengths(pa.array([], type=pa.list_(pa.int64())))

    # Raises if one of the fields is not a ListArray
    with pytest.raises(ValueError):
        validate_struct_list_array_for_equal_lengths(
            pa.StructArray.from_arrays([pa.array([[1, 2], [3, 4, 5]]), pa.array([1, 2])], ["a", "b"])
        )

    # Raises for mismatched lengths
    with pytest.raises(ValueError):
        validate_struct_list_array_for_equal_lengths(
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
    assert validate_struct_list_array_for_equal_lengths(input_array) is None


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
