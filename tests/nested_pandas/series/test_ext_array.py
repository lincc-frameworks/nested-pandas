import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from nested_pandas import NestedDtype
from nested_pandas.series.ext_array import NestedExtensionArray, convert_df_to_pa_scalar, replace_with_mask
from numpy.testing import assert_array_equal
from pandas.core.arrays import ArrowExtensionArray
from pandas.testing import assert_frame_equal, assert_series_equal


def test_replace_with_mask():
    """Test replace_with_mask with struct array of lists."""
    array = pa.chunked_array(
        pa.array([{"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}, {"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]}]),
        type=pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
    )
    mask = pa.array([True, False])
    value = pa.array([{"a": [0, 0, 0], "b": [0.0, 0.0, 0.0]}], type=array.chunk(0).type)
    desired = pa.chunked_array(
        [
            pa.array(
                [{"a": [0, 0, 0], "b": [0.0, 0.0, 0.0]}, {"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]}],
                type=array.chunk(0).type,
            )
        ]
    )
    # pyarrow returns a single bool for ==
    assert replace_with_mask(array, mask, value) == desired


@pytest.mark.parametrize(
    "array,mask,value",
    [
        (
            pa.array([1, 2, 3]),
            pa.array([True, False, False]),
            pa.array([0]),
        ),
        (
            pa.array([1, 2, 3, 4, 5]),
            pa.array([True, False, True, False, True]),
            pa.array([0, 0, 0]),
        ),
    ],
)
def test_replace_with_mask_vs_pyarrow(array, mask, value):
    """Test that replace_with_mask is equivalent to pyarrow.compute.replace_with_mask."""
    chunked_array = pa.chunked_array([array])
    desired = pc.replace_with_mask(chunked_array, mask, value)
    actual = replace_with_mask(chunked_array, mask, value)
    # pyarrow returns a single bool for ==
    assert actual == desired


def test_from_sequence_with_pyarrow_array():
    """Test that we can convert a pyarrow array to a NestedExtensionArray."""
    sequence = pa.array(
        [{"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}, {"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]}],
        type=pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
    )
    actual = NestedExtensionArray.from_sequence(sequence, dtype=None)
    desired = NestedExtensionArray(sequence)
    # pyarrow returns a single bool for ==
    assert actual.equals(desired)


def test_from_sequence_with_ndarray_of_dicts():
    """Test that we can convert a numpy array of dictionaries to a NestedExtensionArray."""
    sequence = np.array(
        [
            {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]},
            {"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]},
        ]
    )
    actual = NestedExtensionArray.from_sequence(sequence, dtype=None)
    desired = NestedExtensionArray(
        pa.array(
            sequence,
            type=pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
        )
    )
    assert actual.equals(desired)


def test_from_sequence_with_list_of_dicts_with_dtype():
    """Test that we can convert a list of dictionaries to a NestedExtensionArray."""
    a = [1, 2, 3]
    b = [-4.0, np.nan, -6.0]
    # pyarrow doesn't convert pandas boxed missing values to nulls in nested arrays
    b_desired = [-4.0, None, -6.0]
    sequence = [
        {"a": a, "b": b},
        {"a": np.array(a), "b": np.array(b)},
        {"a": pd.Series(a), "b": b},
        {"a": pa.array(a), "b": pd.Series(b, dtype=pd.ArrowDtype(pa.float64()))},
        None,
    ]
    actual = NestedExtensionArray.from_sequence(
        sequence, dtype=NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()})
    )
    desired = NestedExtensionArray(
        pa.array(
            [{"a": a, "b": b_desired}] * 4 + [None],
            type=pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
        )
    )
    assert actual.equals(desired)


def test_from_sequence_with_list_of_df():
    """Test that we can convert a list of DataFrames to a NestedExtensionArray."""
    a = [1, 2, 3]
    b = [-4.0, np.nan, -6.0]
    # pyarrow doesn't convert pandas boxed missing values to nulls in nested arrays
    b_desired = [-4.0, None, -6.0]
    sequence = [
        pd.DataFrame({"a": a, "b": b}),
        pd.DataFrame(
            {
                "a": pd.Series(a, dtype=pd.ArrowDtype(pa.int64())),
                "b": pd.Series(b, dtype=pd.ArrowDtype(pa.float64())),
            }
        ),
        None,
        pd.NA,
    ]

    actual = NestedExtensionArray.from_sequence(sequence, dtype=None)
    desired = NestedExtensionArray(
        pa.array(
            [{"a": a, "b": b_desired}] * 2 + [None] * 2,
            type=pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
        )
    )
    assert actual.equals(desired)


def test_from_sequence_with_ndarray_of_df_with_dtype():
    """Test that we can convert a numpy array of DataFrames to a NestedExtensionArray."""
    a = [1, 2, 3]
    b = [-4.0, np.nan, -6.0]
    # pyarrow doesn't convert pandas boxed missing values to nulls in nested arrays
    b_desired = [-4.0, None, -6.0]
    sequence_list = [
        pd.DataFrame({"a": a, "b": b}),
        pd.DataFrame(
            {
                "a": pd.Series(a, dtype=pd.ArrowDtype(pa.float64())),
                "b": pd.Series(b, dtype=pd.ArrowDtype(pa.float64())),
            }
        ),
        None,
    ]
    sequence = np.empty(len(sequence_list), dtype=object)
    sequence[:] = sequence_list
    actual = NestedExtensionArray.from_sequence(
        sequence, dtype=NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()})
    )
    desired = NestedExtensionArray(
        pa.array(
            [{"a": a, "b": b_desired}] * 2 + [None],
            type=pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
        )
    )
    assert actual.equals(desired)


def test_from_sequence_with_arrow_dtyped_series():
    """Test that we can convert pd.Series(..., dtype=pd.ArrowDtype) to a NestedExtensionArray."""
    a = [1, 2, 3]
    b = [-4.0, np.nan, -6.0]
    # pyarrow doesn't convert pandas boxed missing values to nulls in nested arrays
    b_desired = [-4.0, None, -6.0]

    pa_type = pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))])
    arrow_dtype = pd.ArrowDtype(pa_type)

    sequence = pd.Series([{"a": a, "b": b}] * 2 + [None], dtype=arrow_dtype)

    actual = NestedExtensionArray.from_sequence(sequence, dtype=arrow_dtype)
    desired = NestedExtensionArray(
        pa.array(
            [{"a": a, "b": b_desired}] * 2 + [None],
            type=pa_type,
        )
    )
    assert actual.equals(desired)


def test_from_sequence_with_arrow_array_and_dtype():
    """Test that we can convert pa.StructArray to a NestedExtensionArray."""
    a = [1, 2, 3]
    b = [-4.0, None, -6.0]

    pa_type = pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))])
    new_pa_type = pa.struct([pa.field("a", pa.list_(pa.float64())), pa.field("b", pa.list_(pa.float64()))])

    pa_array = pa.array(
        [{"a": a, "b": b}, {"a": a, "b": b}, None],
        type=pa_type,
    )

    actual = NestedExtensionArray.from_sequence(pa_array, dtype=new_pa_type).chunked_array
    desired = pa.chunked_array([pa_array.cast(new_pa_type)])
    # pyarrow doesn't convert pandas boxed missing values to nulls in nested arrays
    assert actual == desired


def test_ext_array_dtype():
    """Test that the dtype of the extension array is correct."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)
    assert ext_array.dtype == NestedDtype(struct_array.type)


def test_series_dtype():
    """Test that the dtype of the series is correct."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)
    series = pd.Series(ext_array)
    assert series.dtype == NestedDtype(struct_array.type)


def test_series_built_with_dtype():
    """Test that the series is built correctly with the given dtype."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    dtype = NestedDtype(struct_array.type)
    series = pd.Series(struct_array, dtype=dtype)
    assert isinstance(series.array, NestedExtensionArray)


def test_series_built_from_dict():
    """Test that the series is built correctly from a dictionary."""
    data = [
        {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]},
        {"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]},
    ]
    dtype = NestedDtype.from_fields({"a": pa.uint8(), "b": pa.float64()})
    series = pd.Series(data, dtype=dtype)

    assert isinstance(series.array, NestedExtensionArray)
    assert series.array.dtype == dtype

    desired_ext_array = NestedExtensionArray(
        pa.StructArray.from_arrays(
            arrays=[
                pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])], type=pa.list_(pa.uint8())),
                pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
            ],
            names=["a", "b"],
        )
    )
    assert_series_equal(series, pd.Series(desired_ext_array))


def test_convert_df_to_pa_scalar():
    """Test that we can convert a DataFrame to a pyarrow struct_scalar."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]})
    pa_scalar = convert_df_to_pa_scalar(df, pa_type=None)

    assert pa_scalar == pa.scalar(
        {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]},
        type=pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
    )


def test_convert_df_to_pa_from_scalar():
    """Test that we can convert a DataFrame to a pyarrow struct_scalar."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]})
    pa_scalar = convert_df_to_pa_scalar(df, pa_type=None)

    assert pa_scalar == pa.scalar(
        {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]},
        type=pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
    )


def test__box_pa_array_from_series_of_df():
    """Test that we can convert a DataFrame to a pyarrow scalar."""
    series = pd.Series(
        [
            pd.DataFrame({"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}),
            pd.DataFrame({"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]}),
        ]
    )
    list_of_dicts = list(NestedExtensionArray._box_pa_array(series, pa_type=None))

    desired_type = pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))])

    assert list_of_dicts == [
        pa.scalar({"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}, type=desired_type),
        pa.scalar({"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]}, type=desired_type),
    ]


def test__box_pa_array_from_list_of_df():
    """Test that we can convert a DataFrame to a pyarrow struct_scalar."""
    list_of_dfs = [
        pd.DataFrame({"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}),
        pd.DataFrame({"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]}),
    ]
    list_of_dicts = list(NestedExtensionArray._box_pa_array(list_of_dfs, pa_type=None))

    desired_type = pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))])

    assert list_of_dicts == [
        pa.scalar({"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}, type=desired_type),
        pa.scalar({"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]}, type=desired_type),
    ]


def test_from_sequence():
    """Test that we can convert a list of DataFrames to a NestedExtensionArray."""
    list_of_dfs = [
        pd.DataFrame({"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}),
        pd.DataFrame({"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]}),
    ]
    ext_array = NestedExtensionArray.from_sequence(list_of_dfs, dtype=None)

    desired = NestedExtensionArray(
        pa.StructArray.from_arrays(
            [pa.array([[1, 2, 3], [1, 2, 1]]), pa.array([[-4.0, -5.0, -6.0], [-3.0, -4.0, -5.0]])],
            names=["a", "b"],
        )
    )
    assert ext_array.equals(desired)


def test___setitem___single_df():
    """Tests nested_ext_array[i] = pd.DataFrame(...) with df of the same size as the struct array."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    ext_array[0] = pd.DataFrame({"a": [5, 6, 7], "b": [100.0, 200.0, 300.0]})

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([5, 6, 7]), np.array([1, 2, 1])]),
            pa.array([np.array([100.0, 200.0, 300.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    desired = NestedExtensionArray(desired_struct_array)

    assert ext_array.equals(desired)


def test___setitem___with_tuple():
    """Tests nested_ext_array[(i,)] = pd.DataFrame(...) with df of the same size as the struct array."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    ext_array[(0,)] = pd.DataFrame({"a": [5, 6, 7], "b": [100.0, 200.0, 300.0]})

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([5, 6, 7]), np.array([1, 2, 1])]),
            pa.array([np.array([100.0, 200.0, 300.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    desired = NestedExtensionArray(desired_struct_array)

    assert ext_array.equals(desired)


def test___setitem___with_empty_key():
    """Test nested_ext_array[[]] = pd.DataFrame(...) does nothing"""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    ext_array[[]] = pd.DataFrame({"a": [5, 6, 7], "b": [100.0, 200.0, 300.0]})

    assert ext_array.equals(NestedExtensionArray(struct_array))


def test___setitem___single_df_different_size():
    """Tests nested_ext_array[i] = pd.DataFrame(...) with df of different size than the struct array."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    ext_array[0] = pd.DataFrame({"a": [5, 6], "b": [100.0, 200.0]})

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([5, 6]), np.array([1, 2, 1])]),
            pa.array([np.array([100.0, 200.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    desired = NestedExtensionArray(desired_struct_array)

    assert ext_array.equals(desired)


def test___setitem___single_df_to_all_rows():
    """Tests nested_ext_array[:] = pd.DataFrame(...)"""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    ext_array[:] = pd.DataFrame({"a": [5, 6], "b": [100.0, 200.0]})

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([5, 6]), np.array([5, 6])]),
            pa.array([np.array([100.0, 200.0]), np.array([100.0, 200.0])]),
        ],
        names=["a", "b"],
    )
    desired = NestedExtensionArray(desired_struct_array)

    assert ext_array.equals(desired)


def test___setitem___list_of_dfs():
    """Tests nested_ext_array[1::2] = [pd.DataFrame(...), pd.DataFrame(...), ...]"""
    struct_array = pa.array(
        [
            {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]},
            {"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]},
            None,
            None,
            {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]},
            None,
        ],
        type=pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
    )
    ext_array = NestedExtensionArray(struct_array)
    ext_array[1::2] = [
        None,
        pd.DataFrame({"a": [5, 6], "b": [100.0, 200.0]}),
        pd.DataFrame({"a": [7, 8], "b": [300.0, 400.0]}),
    ]
    desired_struct_array = pa.array(
        [
            {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]},
            None,
            None,
            {"a": [5, 6], "b": [100.0, 200.0]},
            {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]},
            {"a": [7, 8], "b": [300.0, 400.0]},
        ]
    )
    desired = NestedExtensionArray(desired_struct_array)

    assert ext_array.equals(desired)


def test___setitem___series_of_dfs():
    """Tests nested_ext_array[[1,0]] = pd.Series([pd.DataFrame(...), pd.DataFrame(...), ...])"""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    ext_array[[1, 0]] = pd.Series(
        [
            pd.DataFrame({"a": [5, 6], "b": [100.0, 200.0]}),
            pd.DataFrame({"a": [7, 8], "b": [300.0, 400.0]}),
        ]
    )

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([7, 8]), np.array([5, 6])]),
            pa.array([np.array([300.0, 400.0]), np.array([100.0, 200.0])]),
        ],
        names=["a", "b"],
    )
    desired = NestedExtensionArray(desired_struct_array)

    assert ext_array.chunked_array == desired.chunked_array
    assert ext_array.equals(desired)


def test___setitem___other_ext_array():
    """Tests nested_ext_array[[1,0]] = pd.Series([pd.DataFrame(...), pd.DataFrame(...), ...])"""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    other_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([5, 6]), np.array([7, 8]), np.array([7, 8])]),
            pa.array([np.array([100.0, 200.0]), np.array([300.0, 400.0]), np.array([300.0, 400.0])]),
        ],
        names=["a", "b"],
    )
    other_ext_array = NestedExtensionArray(other_struct_array)

    ext_array[[1, 0, 0]] = other_ext_array

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([7, 8]), np.array([5, 6])]),
            pa.array([np.array([300.0, 400.0]), np.array([100.0, 200.0])]),
        ],
        names=["a", "b"],
    )
    desired = NestedExtensionArray(desired_struct_array)

    assert ext_array.equals(desired)


# Test exception raises for wrong dtype
@pytest.mark.parametrize(
    "data",
    [
        # Must be struct
        [
            1,
            2,
            3,
        ],
        # Must be struct
        {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]},
        # Lists of the same object must have the same length for each field
        [{"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}, {"a": [1, 2, 1], "b": [-3.0, -4.0]}],
        # Struct fields must be lists
        [{"a": 1, "b": [-4.0, -5.0, -6.0]}, {"a": 2, "b": [-3.0, -4.0, -5.0]}],
    ],
)
def test_series_built_raises(data):
    """Test that the extension array raises an error when the data is not valid."""
    pa_array = pa.array(data)
    with pytest.raises(ValueError):
        _array = NestedExtensionArray(pa_array)


def test_chunked_array():
    """Test that the .chunked_array property is correct."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    # pyarrow returns a single bool for ==
    assert ext_array.chunked_array == pa.chunked_array(struct_array)


def test_list_offsets_single_chunk():
    """Test that the .list_offset property is correct for a single chunk."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])], type=pa.list_(pa.uint8())),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    desired = pa.array([0, 3, 6], type=pa.int32())
    # pyarrow returns a single bool for ==
    assert ext_array.list_offsets == desired


def test_list_offsets_multiple_chunks():
    """Test that the .list_offset property is correct for multiple chunks."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])], type=pa.list_(pa.uint8())),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    chunked_arrray = pa.chunked_array([struct_array, struct_array[:1], struct_array])
    ext_array = NestedExtensionArray(chunked_arrray)

    desired = chunked_arrray.combine_chunks().field("a").offsets
    # pyarrow returns a single bool for ==
    assert ext_array.list_offsets == desired


def test___getitem___with_integer():
    """Test [i] is a valid DataFrame."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    second_row_as_df = ext_array[1]
    assert_frame_equal(
        second_row_as_df, pd.DataFrame({"a": np.array([1.0, 2.0, 1.0]), "b": -np.array([3.0, 4.0, 5.0])})
    )


def test___getitem___with_integer_ndarray():
    """Test ext_array[np.array([i1,i2,i3])]"""
    item = {"a": [1.0, 2.0, 3.0], "b": [-4.0, -5.0, -6.0]}
    ext_array = NestedExtensionArray.from_sequence([item, None, item, None, item])
    sliced = ext_array[np.array([3, 1, 0, 0])]
    assert sliced.equals(NestedExtensionArray.from_sequence([None, None, item, item]))


def test___getitem___raises_for_invalid_ndarray_dtype():
    """Test ext_array[np.array([3.14])] fails"""
    ext_array = NestedExtensionArray.from_sequence([{"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}, None])
    with pytest.raises(IndexError):
        _sliced = ext_array[np.array([3.14])]


def test___getitem___with_ellipsis():
    """Test ext_array[...]"""
    item = {"a": [1.0, 2.0, 3.0], "b": [-4.0, -5.0, -6.0]}
    ext_array = NestedExtensionArray.from_sequence([item, None, item, item, None, None, item])
    sliced = ext_array[...]
    assert sliced.equals(ext_array)


def test___getitem___with_single_element_tuple():
    """Test ext_array[(i,)]"""
    item = {"a": [1.0, 2.0, 3.0], "b": [-4.0, -5.0, -6.0]}
    ext_array = NestedExtensionArray.from_sequence([item, None, item, item, None, None, item])
    df = ext_array[(2,)]
    assert_frame_equal(df, pd.DataFrame(item))


def test_series___getitem___with_integer():
    """Tests series[i] is a valid DataFrame."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[100, 101])

    second_row_as_df = series[101]
    assert_frame_equal(
        second_row_as_df, pd.DataFrame({"a": np.array([1.0, 2.0, 1.0]), "b": -np.array([3.0, 4.0, 5.0])})
    )


def test_series___getitem___with_slice():
    """Test series[i:j:step]"""
    item = {"a": [1.0, 2.0, 3.0], "b": [-4.0, -5.0, -6.0]}
    series = pd.Series(
        [item, None, item, item, None, None, item],
        dtype=NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()}),
    )
    sliced = series[-1:0:-2].reset_index(drop=True)
    assert_series_equal(sliced, pd.Series([item, None, item], dtype=series.dtype))


def test_series___getitem___with_slice_object():
    """Test series[slice(first, last, step)]"""
    item = {"a": [1.0, 2.0, 3.0], "b": [-4.0, None, -6.0]}
    series = pd.Series(
        [item, None, item, item, None, None, item],
        dtype=NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()}),
    )
    sliced = series[slice(-1, None, -2)].reset_index(drop=True)
    assert sliced.equals(pd.Series([item, None, item, item], dtype=series.dtype))


def test_series___getitem___with_list_of_integers():
    """Test series[[i, j, k]]"""
    item = {"a": [None, 2.0, 3.0], "b": [-4.0, -5.0, -6.0]}
    series = pd.Series(
        [item, None, item, item, None, None, item],
        dtype=NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()}),
    )
    sliced = series[[0, 2, 5]].reset_index(drop=True)
    assert sliced.equals(pd.Series([item, item, None], dtype=series.dtype))


def test_series___getitem___with_integer_ndarray():
    """Test sesries[np.array([i, j, k])]"""
    item = {"a": [1.0, 2.0, 3.0], "b": [-4.0, pd.NA, -6.0]}
    series = pd.Series(
        [item, None, item, item, None, None, item],
        dtype=NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()}),
    )
    sliced = series[np.array([6, 1, 0, 6])].reset_index(drop=True)
    assert sliced.equals(pd.Series([item, None, item, item], dtype=series.dtype))


def test_series___getitem___with_boolean_ndarray():
    """Test series[np.array([True, False, True, ...])]"""
    item = {"a": [1.0, 2.0, 3.0], "b": [-4.0, -5.0, -6.0]}
    series = pd.Series(
        [item, None, item, item, None, None, item],
        dtype=NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()}),
    )
    sliced = series[np.array([True, False, False, False, False, True, True])].reset_index(drop=True)
    assert_series_equal(sliced, pd.Series([item, None, item], dtype=series.dtype))


def test_isna_when_all_na():
    """Tests isna() when all values are None."""
    ext_array = NestedExtensionArray.from_sequence(
        [None, None, None], dtype=NestedDtype.from_fields({"a": pa.int64()})
    )
    assert_array_equal(ext_array.isna(), np.array([True, True, True]))


def test_isna_when_none_na():
    """Tests isna() when no values are None."""
    ext_array = NestedExtensionArray.from_sequence(
        [{"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}, {"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]}],
        dtype=NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()}),
    )
    assert_array_equal(ext_array.isna(), np.array([False, False]))


def test_isna_when_some_na():
    """Tests isna() when some values are None."""
    ext_array = NestedExtensionArray.from_sequence(
        [None, {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}, pd.NA, pa.scalar(None)],
        dtype=NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()}),
    )
    assert_array_equal(ext_array.isna(), np.array([True, False, True, True]))


@pytest.mark.parametrize(
    "data,desired",
    [([{"a": None, "b": None}], False), ([{"a": [1, 2], "b": [1.0, 2.0]}, pd.NA], True)],
)
def test__hasna(data, desired):
    """Tests _hasna()."""
    ext_array = NestedExtensionArray.from_sequence(
        data, dtype=NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()})
    )
    assert ext_array._hasna == desired


@pytest.mark.parametrize(
    "allow_fill,fill_value,desired_sequence",
    [
        (
            False,
            None,
            [
                None,
                {"a": [1, 2, 3], "b": [-4.0, None, -6.0]},
                {"a": [1, 2, 3], "b": [-4.0, None, -6.0]},
                None,
                {"a": [4, 5, 6], "b": [-7.0, -8.0, -9.0]},
                {"a": [4, 5, 6], "b": [-7.0, -8.0, -9.0]},
            ],
        ),
        (
            True,
            None,
            [
                None,
                {"a": [1, 2, 3], "b": [-4.0, None, -6.0]},
                {"a": [1, 2, 3], "b": [-4.0, None, -6.0]},
                None,
                {"a": [4, 5, 6], "b": [-7.0, -8.0, -9.0]},
                None,
            ],
        ),
        (
            True,
            pd.DataFrame({"a": [-1, pd.NA], "b": [100.0, 100.0]}),
            [
                None,
                {"a": [1, 2, 3], "b": [-4.0, None, -6.0]},
                {"a": [1, 2, 3], "b": [-4.0, None, -6.0]},
                None,
                {"a": [4, 5, 6], "b": [-7.0, -8.0, -9.0]},
                {"a": [-1, None], "b": [100.0, 100.0]},
            ],
        ),
    ],
)
def test_take(allow_fill, fill_value, desired_sequence):
    """Tests .take([i1, i2, i3])."""
    ext_array = NestedExtensionArray.from_sequence(
        [
            {"a": [1, 2, 3], "b": [-4.0, None, -6.0]},
            None,
            pd.NA,
            {"a": [4, 5, 6], "b": [-7.0, -8.0, -9.0]},
        ]
    )
    indices = [1, 0, 0, 2, 3, -1]
    desired = NestedExtensionArray.from_sequence(desired_sequence, dtype=ext_array.dtype)

    result = ext_array.take(indices, allow_fill=allow_fill, fill_value=fill_value)
    assert result.equals(desired)


def test_take_raises_for_empty_array_and_non_empty_index():
    """Tests that .take([i1, i2, i3]) raises for empty array"""
    ext_array = NestedExtensionArray.from_sequence([], dtype=NestedDtype.from_fields({"a": pa.int64()}))
    with pytest.raises(IndexError):
        _result = ext_array.take([0, 1, 2])


@pytest.mark.parametrize(
    "indices",
    [
        [100],
        [1 << 65],
        [-100],
        [0] * 100 + [100],
    ],
)
def test_take_raises_for_out_of_bounds_index(indices):
    """Tests that .take([i1, i2, i3]) raises for out of bounds index."""
    ext_array = NestedExtensionArray.from_sequence(
        [None, None], dtype=NestedDtype.from_fields({"a": pa.int64()})
    )
    with pytest.raises(IndexError):
        ext_array.take(indices)


def test__formatter_unboxed():
    """Tests formatting of array values, when displayed alone."""
    formatter = NestedExtensionArray.from_sequence(
        [], dtype=NestedDtype.from_fields({"a": pa.int64()})
    )._formatter(boxed=False)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]})
    assert formatter(df) == repr(df)


def test__formatter_boxed():
    """Tests formatting of array values, when displayed in a DataFrame or Series"""
    formatter = NestedExtensionArray.from_sequence(
        [], dtype=NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()})
    )._formatter(boxed=True)
    d = {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}
    df = pd.DataFrame(d)
    assert formatter(df) == str(d)


def test__formetter_boxed_na():
    """Tests formatting of NA array value, when displayed in a DataFrame or Series"""
    formatter = NestedExtensionArray.from_sequence(
        [], dtype=NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()})
    )._formatter(boxed=True)
    assert formatter(pd.NA) == str(pd.NA)


def test_nbytes():
    """Test that the nbytes property is correct."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])], type=pa.list_(pa.uint32())),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])], pa.list_(pa.float64())),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    # Assume a typical 64-bit platform
    a_data_size = 6 * 4
    a_validity_buffer = 8  # cannot be smaller than 8 bytes because of alignment
    b_data_size = 6 * 8
    b_validity_buffer = 8  # cannot be smaller than 8 bytes because of alignment

    assert ext_array.nbytes == a_data_size + a_validity_buffer + b_data_size + b_validity_buffer


def test_pickability():
    """Test that the extension array can be dumped and loaded back with pickle."""
    ext_array = NestedExtensionArray.from_sequence(
        [{"a": [1, None, 3], "b": [-4.0, -5.0, None]}, None, {"a": [100] * 10_000, "b": [-7.0] * 10_000}]
    )
    pickled = pickle.loads(pickle.dumps(ext_array))
    assert ext_array.equals(pickled)


def test__concat_same_type():
    """Test concatenating of three NestedExtensionArrays with the same dtype."""
    dtype = NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()})
    array1 = NestedExtensionArray.from_sequence(
        [{"a": [1, 2, None], "b": [-2.0, None, -4.0]}, {"a": [None], "b": [3.14]}], dtype=dtype
    )
    array2 = NestedExtensionArray.from_sequence(
        [{"a": [4, 5, 6], "b": [-7.0, -8.0, -9.0]}, None], dtype=dtype
    )
    array3 = NestedExtensionArray.from_sequence([], dtype=dtype)
    array4 = NestedExtensionArray.from_sequence([None], dtype=dtype)

    desired = NestedExtensionArray.from_sequence(
        [
            {"a": [1, 2, None], "b": [-2.0, None, -4.0]},
            {"a": [None], "b": [3.14]},
            {"a": [4, 5, 6], "b": [-7.0, -8.0, -9.0]},
            None,
            None,
        ]
    )
    actual = NestedExtensionArray._concat_same_type([array1, array2, array3, array4])

    assert actual.equals(desired)


def test_equals():
    """Test that two NestedExtensionArrays are equal."""
    dtype = NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()})
    array1 = NestedExtensionArray.from_sequence(
        [{"a": [1, 2, None], "b": [-2.0, None, -4.0]}, {"a": [None], "b": [3.14]}, None], dtype=dtype
    )
    array2 = NestedExtensionArray.from_sequence(
        [
            pd.DataFrame({"a": [1, 2, pd.NA], "b": [-2.0, pd.NA, -4.0]}),
            pd.DataFrame({"a": [pd.NA], "b": [3.14]}),
            pd.NA,
        ],
        dtype=dtype,
    )

    assert array1.equals(array2)


def test_equals_when_other_is_different_type():
    """Test that equals() raises for different dtypes."""
    ext_array = NestedExtensionArray.from_sequence([{"a": [1, None, 3], "b": [-4.0, -5.0, None]}, None])
    other = ext_array.to_arrow_ext_array()
    assert not ext_array.equals(other)


def test_dropna():
    """Test .dropna()"""
    dtype = NestedDtype.from_fields({"a": pa.int64(), "b": pa.float64()})
    array = NestedExtensionArray.from_sequence(
        [
            {"a": [1, 2, None], "b": [-2.0, None, -4.0]},
            {"a": [None], "b": [3.14]},
            None,
            {"a": [4, 5, 6], "b": [-7.0, -8.0, -9.0]},
        ],
        dtype=dtype,
    )

    desired = NestedExtensionArray.from_sequence(
        [
            {"a": [1, 2, None], "b": [-2.0, None, -4.0]},
            {"a": [None], "b": [3.14]},
            {"a": [4, 5, 6], "b": [-7.0, -8.0, -9.0]},
        ],
        dtype=dtype,
    )
    actual = array.dropna()

    assert actual.equals(desired)


def test___arrow_array__():
    """Test that the extension array can be converted to a pyarrow array."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    arrow_array = pa.array(ext_array)
    assert arrow_array == struct_array


def test___arrow_array___with_type_cast():
    """Test that the extension array can be converted to a pyarrow array with a type cast."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)
    new_pa_type = pa.struct([pa.field("a", pa.list_(pa.float64())), pa.field("b", pa.list_(pa.float64()))])

    arrow_array = pa.array(
        ext_array,
        type=new_pa_type,
    )
    assert arrow_array == struct_array.cast(new_pa_type)


def test___array__():
    """Test that the extensions array can be converted to a numpy array and back."""
    struct_array = pa.array(
        [
            {"a": [1, 2, 3], "b": [-4.0, None, -6.0]},
            None,
            {"a": [1, 2, None], "b": [-3.0, -4.0, -5.0]},
        ],
        type=pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
    )
    ext_array = NestedExtensionArray(struct_array)

    np_array = np.array(ext_array)
    assert np_array.dtype == object

    new_ext_array = NestedExtensionArray.from_sequence(np_array)
    assert new_ext_array.dtype == ext_array.dtype
    assert new_ext_array.equals(ext_array)


@pytest.mark.parametrize(
    "value,pa_type,desired",
    [
        (
            {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]},
            None,
            pa.scalar(
                {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]},
                type=pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
            ),
        ),
        (
            pd.DataFrame({"a": [-4.0, 5.0, None, 7.0], "b": ["hello", "world", "!", ""]}),
            None,
            pa.scalar(
                {"a": [-4.0, 5.0, None, 7.0], "b": ["hello", "world", "!", ""]},
                type=pa.struct([pa.field("a", pa.list_(pa.float64())), pa.field("b", pa.list_(pa.string()))]),
            ),
        ),
        (
            {"a": [None, None, None], "b": [None, None, None]},
            pa.struct([pa.field("a", pa.list_(pa.string())), pa.field("b", pa.list_(pa.float64()))]),
            pa.scalar(
                {"a": [None, None, None], "b": [None, None, None]},
                type=pa.struct([pa.field("a", pa.list_(pa.string())), pa.field("b", pa.list_(pa.float64()))]),
            ),
        ),
        (
            None,
            pa.struct([pa.field("a", pa.list_(pa.string())), pa.field("b", pa.list_(pa.float64()))]),
            pa.scalar(
                None,
                type=pa.struct([pa.field("a", pa.list_(pa.string())), pa.field("b", pa.list_(pa.float64()))]),
            ),
        ),
        (
            pd.NA,
            pa.struct([pa.field("a", pa.list_(pa.string())), pa.field("b", pa.list_(pa.float64()))]),
            pa.scalar(
                None,
                type=pa.struct([pa.field("a", pa.list_(pa.string())), pa.field("b", pa.list_(pa.float64()))]),
            ),
        ),
        (pa.scalar(None), None, pa.scalar(None)),
    ],
)
def test__box_pa_scalar(value, pa_type, desired):
    """Tests _box_pa_scalar()"""
    actual = NestedExtensionArray._box_pa_scalar(value, pa_type=pa_type)
    assert actual == desired


@pytest.mark.parametrize(
    "value,pa_type,desired",
    [
        (
            [None, {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}],
            None,
            pa.array(
                [None, {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}],
                type=pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
            ),
        ),
        (
            [pd.NA, pd.DataFrame({"a": [-4.0, 5.0, pd.NA, 7.0], "b": ["hello", "world", "!", pd.NA]})],
            None,
            pa.array(
                [None, {"a": [-4.0, 5.0, None, 7.0], "b": ["hello", "world", "!", None]}],
                type=pa.struct([pa.field("a", pa.list_(pa.float64())), pa.field("b", pa.list_(pa.string()))]),
            ),
        ),
        (
            [pd.NA] * 3,
            pa.struct([pa.field("a", pa.list_(pa.string())), pa.field("b", pa.list_(pa.float64()))]),
            pa.array(
                [None, None, None],
                type=pa.struct([pa.field("a", pa.list_(pa.string())), pa.field("b", pa.list_(pa.float64()))]),
            ),
        ),
        (
            pa.array(
                [{"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}, None],
                type=pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
            ),
            pa.struct([pa.field("a", pa.list_(pa.float64())), pa.field("b", pa.list_(pa.float64()))]),
            pa.array(
                [{"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}, None],
                type=pa.struct(
                    [pa.field("a", pa.list_(pa.float64())), pa.field("b", pa.list_(pa.float64()))]
                ),
            ),
        ),
    ],
)
def test__box_pa_array(value, pa_type, desired):
    """Tests _box_pa_array"""
    actual = NestedExtensionArray._box_pa_array(value, pa_type=pa_type)
    assert actual == desired


def test_series_apply_udf_argument():
    """Tests `x` in series.apply(lambda x: x) is a valid DataFrame."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[100, 101])

    series_of_dfs = series.apply(lambda x: x)
    assert_frame_equal(
        series_of_dfs.iloc[0], pd.DataFrame({"a": np.array([1.0, 2.0, 3.0]), "b": -np.array([4.0, 5.0, 6.0])})
    )


def test___iter__():
    """Tests iter(series) yields valid DataFrames."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[100, 101])

    # Check last df only
    df = list(series)[-1]
    assert_frame_equal(df, pd.DataFrame({"a": np.array([1.0, 2.0, 1.0]), "b": -np.array([3.0, 4.0, 5.0])}))


def test___eq__():
    """Check it raises NotImplementedError."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[100, 101])

    with pytest.raises(NotImplementedError):
        _ = series == series


def test_field_names():
    """Tests that the extension array field names are correct."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    assert ext_array.field_names == ["a", "b"]


def test_flat_length():
    """Tests that the flat length of the extension array is correct."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    assert ext_array.flat_length == 7


def test_num_chunks():
    """Tests .num_chunks property."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    chunked_array = pa.chunked_array([struct_array] * 7)
    ext_array = NestedExtensionArray(chunked_array)

    assert ext_array.num_chunks == 7


def test_iter_field_lists():
    """Test .iter_field_lists() yields the correct field lists"""
    a = [[1, 2, 3], [1, 2, 3, 4]]
    b = [np.array(["a", "b", "c"]), np.array(["x", "y", "z", "w"])]
    struct_array = pa.StructArray.from_arrays(
        arrays=[a, b],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    for actual, desired in zip(ext_array.iter_field_lists("a"), a):
        assert_array_equal(actual, desired)
    for actual, desired in zip(ext_array.iter_field_lists("b"), b):
        assert_array_equal(actual, desired)


def test_view_fields_with_single_field():
    """Tests ext_array.view("field")"""
    arrays = [
        pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
        pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
    ]
    ext_array = NestedExtensionArray(
        pa.StructArray.from_arrays(
            arrays=arrays,
            names=["a", "b"],
        )
    )

    view = ext_array.view_fields("a")
    assert view.field_names == ["a"]

    desired = NestedExtensionArray(
        pa.StructArray.from_arrays(
            arrays=arrays[:1],
            names=["a"],
        )
    )

    assert_series_equal(pd.Series(view), pd.Series(desired))


def test_view_fields_with_multiple_fields():
    """Tests ext_array.view(["field1", "field2"])"""
    arrays = [
        pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])]),
        pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        pa.array([["x", "y", "z"], ["x1", "x2", "x3", "x4"]]),
    ]
    ext_array = NestedExtensionArray(
        pa.StructArray.from_arrays(
            arrays=arrays,
            names=["a", "b", "c"],
        )
    )

    view = ext_array.view_fields(["b", "a"])
    assert view.field_names == ["b", "a"]

    assert_series_equal(
        pd.Series(view),
        pd.Series(
            NestedExtensionArray(pa.StructArray.from_arrays(arrays=[arrays[1], arrays[0]], names=["b", "a"]))
        ),
    )


def test_view_fields_raises_for_invalid_field():
    """Tests that we raise an error when trying to view a field that does not exist."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    with pytest.raises(ValueError):
        ext_array.view_fields("c")


def test_view_fields_raises_for_non_unique_fields():
    """Tests that we raise an error when trying to view multiple fields with the sama name."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    with pytest.raises(ValueError):
        ext_array.view_fields(["a", "a"])


def test_set_flat_field_new_field_scalar():
    """Tests setting a new field with a struct_scalar value."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    ext_array.set_flat_field("c", "YES")

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
            pa.array([["YES"] * 3, ["YES"] * 4]),
        ],
        names=["a", "b", "c"],
    )
    desired = NestedExtensionArray(desired_struct_array)

    assert_series_equal(pd.Series(ext_array), pd.Series(desired))


def test_set_flat_field_replace_field_array():
    """Tests replacing a field with a new "flat" array."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    ext_array.set_flat_field("b", [True, False, True, False, True, False, True])

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])]),
            pa.array([np.array([True, False, True]), np.array([False, True, False, True])]),
        ],
        names=["a", "b"],
    )
    desired = NestedExtensionArray(desired_struct_array)

    assert_series_equal(pd.Series(ext_array), pd.Series(desired))


def test_set_flat_field_keep_dtype_raises_for_wrong_dtype():
    """Tests that set_flat_field(keep_dtype=True) raises for a wrong input dtype."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    with pytest.raises(TypeError):
        ext_array.set_flat_field("b", ["x", "y", "z", "w", "v", "u", "t"], keep_dtype=True)

    # Do not raise when keep_dtype=False
    ext_array.set_flat_field("b", ["x", "y", "z", "w", "v", "u", "t"], keep_dtype=False)


def test_set_flat_field_keep_dtype_raises_for_new_field():
    """Tests that set_flat_field(keep_dtype=True) raises for a new field."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    with pytest.raises(ValueError):
        ext_array.set_flat_field("c", [True, False, True, False, True, False, True], keep_dtype=True)

    # Do not raise when keep_dtype=False
    ext_array.set_flat_field("c", [True, False, True, False, True, False, True], keep_dtype=False)


def test_set_list_field_new_field():
    """Tests setting a new field with a new "list" array"""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    ext_array.set_list_field("c", [["x", "y", "z"], ["x1", "x2", "x3", "x4"]])

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
            pa.array([np.array(["x", "y", "z"]), np.array(["x1", "x2", "x3", "x4"])]),
        ],
        names=["a", "b", "c"],
    )
    desired = NestedExtensionArray(desired_struct_array)

    assert_series_equal(pd.Series(ext_array), pd.Series(desired))


def test_set_list_field_replace_field():
    """Tests replacing a field with a new "list" array."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    ext_array.set_list_field("b", [["x", "y", "z"], ["x1", "x2", "x3", "x4"]])

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([np.array(["x", "y", "z"]), np.array(["x1", "x2", "x3", "x4"])]),
        ],
        names=["a", "b"],
    )
    desired = NestedExtensionArray(desired_struct_array)

    assert_series_equal(pd.Series(ext_array), pd.Series(desired))


def test_set_list_field_raises_for_non_list_array():
    """Tests that we raise an error when trying to set a field with a non-list array."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    with pytest.raises(ValueError):
        ext_array.set_list_field("b", [1.0, 2.0])


def test_set_list_field_raises_for_wrong_length():
    """Tests that we raise an error when trying to set a field with an array-like of the wrong length."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    longer_array = np.array([[1.0, 2.0, 3.0], [1.0, 2.0]], dtype=object)

    with pytest.raises(ValueError):
        ext_array.set_list_field("b", longer_array)


def test_set_list_field_keep_dtype_raises_for_wrong_dtype():
    """Tests that set_list_field(keep_dtype=True) raises for a wrong input dtype."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    with pytest.raises(TypeError):
        ext_array.set_list_field("b", [["x", "y", "z"]] * 2, keep_dtype=True)

    # Do not raise when keep_dtype=False
    ext_array.set_list_field("b", [["x", "y", "z"]] * 2, keep_dtype=False)


def test_set_list_field_keep_dtype_raises_for_new_field():
    """Tests that set_list_field(keep_dtype=True) raises for a new field."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    with pytest.raises(ValueError):
        ext_array.set_list_field("c", [["x", "y", "z"]], keep_dtype=True)

    # Do not raise when keep_dtype=False
    ext_array.set_list_field("c", [["x", "y", "z"]], keep_dtype=False)


def test_pop_fields():
    """Tests that we can pop a field from the extension array."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
            pa.array([np.array(["x", "y", "z"]), np.array(["x1", "x2", "x3", "x4"])]),
        ],
        names=["a", "b", "c"],
    )
    ext_array = NestedExtensionArray(struct_array)

    ext_array.pop_fields(["c"])

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    desired = NestedExtensionArray(desired_struct_array)

    assert_series_equal(pd.Series(ext_array), pd.Series(desired))


def test_pop_fields_multiple_fields():
    """Tests that we can pop multiple fields from the extension array."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
            pa.array([np.array(["x", "y", "z"]), np.array(["x1", "x2", "x3", "x4"])]),
        ],
        names=["a", "b", "c"],
    )
    ext_array = NestedExtensionArray(struct_array)

    ext_array.pop_fields(["a", "c"])

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])])],
        names=["b"],
    )
    desired = NestedExtensionArray(desired_struct_array)

    assert_series_equal(pd.Series(ext_array), pd.Series(desired))


def test_pop_fields_raises_for_invalid_field():
    """Tests that we raise an error when trying to pop a field that does not exist."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    with pytest.raises(ValueError):
        ext_array.pop_fields(["c"])


def test_pop_fields_raises_for_some_invalid_fields():
    """Tests that we raise an error when trying to pop some fields that do not exist."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    with pytest.raises(ValueError):
        ext_array.pop_fields(["a", "c"])


def test_delete_last_field_raises():
    """Tests that we raise an error when trying to delete the last field left."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
            pa.array([np.array(["x", "y", "z"]), np.array(["x1", "x2", "x3", "x4"])]),
        ],
        names=["a", "b", "c"],
    )
    ext_array = NestedExtensionArray(struct_array)

    ext_array.pop_fields(["c", "a"])
    assert ext_array.field_names == ["b"]

    with pytest.raises(ValueError):
        ext_array.pop_fields(["b"])


def test_from_arrow_ext_array():
    """Tests that we can create a NestedExtensionArray from an ArrowExtensionArray."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1, 2])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = ArrowExtensionArray(struct_array)

    from_arrow = NestedExtensionArray.from_arrow_ext_array(ext_array)
    assert_series_equal(pd.Series(ext_array), pd.Series(from_arrow), check_dtype=False)


def test_to_arrow_ext_array():
    """Tests that we can create an ArrowExtensionArray from a NestedExtensionArray."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1, 2])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    to_arrow = ext_array.to_arrow_ext_array()
    assert_series_equal(pd.Series(ext_array), pd.Series(to_arrow), check_dtype=False)


def test_series_interpolate():
    """We do not support interpolate() on NestedExtensionArray."""
    with pytest.raises(NotImplementedError):
        _series = pd.Series(
            [pd.DataFrame({"a": [1, 2, 3]}), pd.NA], dtype=NestedDtype.from_fields({"a": pa.float64()})
        ).interpolate()


def test__from_sequence_of_strings():
    """We do not support from_sequence_of_strings() which would apply things like pd.read_csv()"""
    with pytest.raises(NotImplementedError):
        NestedExtensionArray._from_sequence_of_strings(["1,2,3", "4,5,6"])


def test__from_factorized():
    """We do not support _from_factorized() which would apply pd.factorize()"""
    with pytest.raises(NotImplementedError):
        NestedExtensionArray._from_factorized(
            [0], NestedExtensionArray.from_sequence([{"a": [1, 2, 3], "b": [4, 5, 6]}])
        )
