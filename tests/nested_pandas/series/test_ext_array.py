import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas import NestedDtype
from nested_pandas.series.ext_array import NestedExtensionArray
from numpy.testing import assert_array_equal
from pandas.core.arrays import ArrowExtensionArray
from pandas.testing import assert_frame_equal, assert_series_equal


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


def test__convert_df_to_pa_scalar():
    """Test that we can convert a DataFrame to a pyarrow scalar."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]})
    pa_scalar = NestedExtensionArray._convert_df_to_pa_scalar(df, type=None)

    assert pa_scalar == pa.scalar(
        {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]},
        type=pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
    )


def test__convert_df_to_pa_from_scalar():
    """Test that we can convert a DataFrame to a pyarrow scalar."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]})
    pa_scalar = NestedExtensionArray._convert_df_to_pa_scalar(df, type=None)

    assert pa_scalar == pa.scalar(
        {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]},
        type=pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
    )


def test__convert_df_to_pa_from_series():
    """Test that we can convert a DataFrame to a pyarrow scalar."""
    series = pd.Series(
        [
            pd.DataFrame({"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}),
            pd.DataFrame({"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]}),
        ]
    )
    list_of_dicts = list(NestedExtensionArray._convert_df_value_to_pa(series, type=None))

    desired_type = pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))])

    assert list_of_dicts == [
        pa.scalar({"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}, type=desired_type),
        pa.scalar({"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]}, type=desired_type),
    ]


def test__convert_df_to_pa_from_list():
    """Test that we can convert a DataFrame to a pyarrow scalar."""
    list_of_dfs = [
        pd.DataFrame({"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}),
        pd.DataFrame({"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]}),
    ]
    list_of_dicts = list(NestedExtensionArray._convert_df_value_to_pa(list_of_dfs, type=None))

    desired_type = pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))])

    assert list_of_dicts == [
        pa.scalar({"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}, type=desired_type),
        pa.scalar({"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]}, type=desired_type),
    ]


def test__from_sequence():
    """Test that we can convert a list of DataFrames to a NestedExtensionArray."""
    list_of_dfs = [
        pd.DataFrame({"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}),
        pd.DataFrame({"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]}),
    ]
    ext_array = NestedExtensionArray._from_sequence(list_of_dfs, dtype=None)

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
    """Tests nested_ext_array[:] = [pd.DataFrame(...), pd.DataFrame(...), ...]"""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    ext_array[:] = [
        pd.DataFrame({"a": [5, 6], "b": [100.0, 200.0]}),
        pd.DataFrame({"a": [7, 8], "b": [300.0, 400.0]}),
    ]

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([5, 6]), np.array([7, 8])]),
            pa.array([np.array([100.0, 200.0]), np.array([300.0, 400.0])]),
        ],
        names=["a", "b"],
    )
    desired = NestedExtensionArray(desired_struct_array)

    assert ext_array.equals(desired)


def test___setitem___series_of_dfs():
    """Tests nested_ext_array[:] = pd.Series([pd.DataFrame(...), pd.DataFrame(...), ...])"""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    ext_array[:] = pd.Series(
        [
            pd.DataFrame({"a": [5, 6], "b": [100.0, 200.0]}),
            pd.DataFrame({"a": [7, 8], "b": [300.0, 400.0]}),
        ]
    )

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([5, 6]), np.array([7, 8])]),
            pa.array([np.array([100.0, 200.0]), np.array([300.0, 400.0])]),
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


def test_list_offsets():
    """Test that the list offsets are correct."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])], type=pa.list_(pa.uint8())),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = NestedExtensionArray(struct_array)

    desired = pa.chunked_array([pa.array([0, 3, 6])])
    assert_array_equal(ext_array.list_offsets, desired)


def test___getitem__():
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
    """Tests setting a new field with a scalar value."""
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


def test_pop_field():
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

    ext_array.pop_field("c")

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    desired = NestedExtensionArray(desired_struct_array)

    assert_series_equal(pd.Series(ext_array), pd.Series(desired))


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

    ext_array.pop_field("a")
    assert ext_array.field_names == ["b", "c"]

    ext_array.pop_field("c")
    assert ext_array.field_names == ["b"]

    with pytest.raises(ValueError):
        ext_array.pop_field("b")


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