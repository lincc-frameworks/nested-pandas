import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas import NestedDtype
from nested_pandas.series.ext_array import NestedExtensionArray
from nested_pandas.series.packer import pack_flat, pack_seq
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal


def test_registered():
    """Test that the series accessor .nest is registered."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1.0, 2.0, 1.0])]),
            pa.array([np.array([4, 5, 6]), np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    _accessor = series.nest


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1, 2, 3]),
        pd.Series([1.0, 2.0, 3.0], dtype=pd.ArrowDtype(pa.float64())),
        pd.Series(
            [{"a": [1, 2]}, {"a": [3, 4]}],
            dtype=pd.ArrowDtype(pa.struct([pa.field("a", pa.list_(pa.int64()))])),
        ),
    ],
)
def test_does_not_work_for_non_nested_series(series):
    """Test that the .nest accessor does not work for non-nested series."""
    with pytest.raises(AttributeError):
        _ = series.nest


def test_to_lists():
    """Test that the .nest.to_lists() method works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), -np.array([1.0, 2.0, 1.0])]),
            pa.array([np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    lists = series.nest.to_lists()

    desired = pd.DataFrame(
        data={
            "a": pd.Series(
                data=[np.array([1.0, 2.0, 3.0]), -np.array([1.0, 2.0, 1.0])],
                dtype=pd.ArrowDtype(pa.list_(pa.float64())),
                index=[0, 1],
            ),
            "b": pd.Series(
                data=[np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])],
                dtype=pd.ArrowDtype(pa.list_(pa.float64())),
                index=[0, 1],
            ),
        },
    )
    assert_frame_equal(lists, desired)


def test_to_lists_for_chunked_array():
    """ ""Test that the .nest.to_lists() when underlying array is chunked"""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            [np.array([1.0, 2.0, 3.0]), -np.array([1.0, 2.0, 1.0])],
            [np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])],
        ],
        names=["a", "b"],
    )
    chunked_array = pa.chunked_array([struct_array] * 3)
    assert chunked_array.length() == 6
    series = pd.Series(chunked_array, dtype=NestedDtype(chunked_array.type), index=[0, 1, 2, 3, 4, 5])
    assert series.array.num_chunks == 3

    lists = series.nest.to_lists()

    desired = pd.DataFrame(
        data={
            "a": pd.Series(
                data=[np.array([1.0, 2.0, 3.0]), -np.array([1.0, 2.0, 1.0])] * 3,
                dtype=pd.ArrowDtype(pa.list_(pa.float64())),
                index=[0, 1, 2, 3, 4, 5],
            ),
            "b": pd.Series(
                data=[np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])] * 3,
                dtype=pd.ArrowDtype(pa.list_(pa.float64())),
                index=[0, 1, 2, 3, 4, 5],
            ),
        },
    )
    assert_frame_equal(lists, desired)


def test_to_lists_with_fields():
    """Test that the .nest.to_lists(fields=...) method works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), -np.array([1.0, 2.0, 1.0])]),
            pa.array([np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    lists = series.nest.to_lists(fields=["a"])

    desired = pd.DataFrame(
        data={
            "a": pd.Series(
                data=[np.array([1.0, 2.0, 3.0]), -np.array([1.0, 2.0, 1.0])],
                dtype=pd.ArrowDtype(pa.list_(pa.float64())),
                index=[0, 1],
            ),
        },
    )
    assert_frame_equal(lists, desired)


def test_to_lists_fails_for_empty_input():
    """Test that the .nest.to_lists([]) fails when no fields are provided."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([]), np.array([])]),
            pa.array([np.array([]), np.array([])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    with pytest.raises(ValueError):
        _ = series.nest.to_lists([])


def test_to_flat():
    """Test that the .nest.to_flat() method works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )

    series = pd.Series(
        struct_array, dtype=NestedDtype(struct_array.type), index=pd.Series([0, 1], name="idx")
    )

    flat = series.nest.to_flat()

    desired = pd.DataFrame(
        data={
            "a": pd.Series(
                data=[1.0, 2.0, 3.0, 1.0, 2.0, 1.0],
                index=[0, 0, 0, 1, 1, 1],
                name="a",
                copy=False,
                dtype=pd.ArrowDtype(pa.float64()),
            ),
            "b": pd.Series(
                data=[-4.0, -5.0, -6.0, -3.0, -4.0, -5.0],
                index=[0, 0, 0, 1, 1, 1],
                name="b",
                copy=False,
                dtype=pd.ArrowDtype(pa.float64()),
            ),
        },
        index=pd.Index([0, 0, 0, 1, 1, 1], name="idx"),
    )

    assert_array_equal(flat.dtypes, desired.dtypes)
    assert_array_equal(flat.index, desired.index)
    assert flat.index.name == desired.index.name

    for column in flat.columns:
        assert_array_equal(flat[column], desired[column])


def test_to_flat_for_chunked_array():
    """Test that the .nest.to_flat() when underlying array is pa.ChunkedArray."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            [np.array([1.0, 2.0, 3.0]), -np.array([1.0, 2.0, 1.0])],
            [np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])],
        ],
        names=["a", "b"],
    )
    chunked_array = pa.chunked_array([struct_array] * 3)
    assert chunked_array.length() == 6
    series = pd.Series(chunked_array, dtype=NestedDtype(chunked_array.type), index=[0, 1, 2, 3, 4, 5])
    assert series.array.num_chunks == 3

    flat = series.nest.to_flat()

    desired = pd.DataFrame(
        data={
            "a": pd.Series(
                data=[1.0, 2.0, 3.0, -1.0, -2.0, -1.0] * 3,
                name="a",
                index=np.repeat([0, 1, 2, 3, 4, 5], 3),
                dtype=pd.ArrowDtype(pa.float64()),
            ),
            "b": pd.Series(
                data=[4.0, 5.0, 6.0, -3.0, -4.0, -5.0] * 3,
                name="b",
                index=np.repeat([0, 1, 2, 3, 4, 5], 3),
                dtype=pd.ArrowDtype(pa.float64()),
            ),
        },
        index=pd.Index(np.repeat([0, 1, 2, 3, 4, 5], 3)),
    )

    assert_frame_equal(flat, desired)


def test_to_flat_with_fields():
    """Test that the .nest.to_flat(fields=...) method works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    flat = series.nest.to_flat(fields=["a"])

    desired = pd.DataFrame(
        data={
            "a": pd.Series(
                data=[1.0, 2.0, 3.0, 1.0, 2.0, 1.0],
                index=[0, 0, 0, 1, 1, 1],
                name="a",
                copy=False,
                dtype=pd.ArrowDtype(pa.float64()),
            ),
        },
    )

    assert_array_equal(flat.dtypes, desired.dtypes)
    assert_array_equal(flat.index, desired.index)

    for column in flat.columns:
        assert_array_equal(flat[column], desired[column])


def test_to_flat_fails_for_empty_input():
    """Test that the .nest.to_flat([]) fails when no fields are provided."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([]), np.array([])]),
            pa.array([np.array([]), np.array([])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    with pytest.raises(ValueError):
        _ = series.nest.to_flat([])


def test_fields():
    """Test that the .nest.fields attribute works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    assert_array_equal(series.nest.fields, ["a", "b"])


def test_flat_length():
    """Test that the .nest.flat_length attribute works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    assert series.nest.flat_length == 6


def test_with_flat_field():
    """Test that the .nest.set_flat_field() method works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    new_series = series.nest.with_flat_field("a", np.array(["a", "b", "c", "d", "e", "f"]))

    assert_series_equal(
        new_series.nest["a"],
        pd.Series(
            data=["a", "b", "c", "d", "e", "f"],
            index=[0, 0, 0, 1, 1, 1],
            name="a",
            dtype=pd.ArrowDtype(pa.string()),
        ),
    )


def test_with_field():
    """Test that .nest.with_field is just an alias to .nest.with_flat_field."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])
    assert_series_equal(
        series.nest.with_field("a", np.array(["a", "b", "c", "d", "e", "f"])),
        series.nest.with_flat_field("a", np.array(["a", "b", "c", "d", "e", "f"])),
    )


def test_with_list_field():
    """Test that the .nest.set_list_field() method works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    new_series = series.nest.with_list_field("c", [["a", "b", "c"], ["d", "e", "f"]])

    assert_series_equal(
        new_series.nest["c"],
        pd.Series(
            data=["a", "b", "c", "d", "e", "f"],
            index=[0, 0, 0, 1, 1, 1],
            name="c",
            dtype=pd.ArrowDtype(pa.string()),
        ),
    )


def test_without_field_single_field():
    """Test .nest.without_field("field")"""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([4, 5, 6])]),
            pa.array([np.array([6, 4, 2]), np.array([1, 2, 3])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[5, 7])

    new_series = series.nest.without_field("a")

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([6, 4, 2]), np.array([1, 2, 3])]),
        ],
        names=["b"],
    )
    desired = pd.Series(desired_struct_array, dtype=NestedDtype(desired_struct_array.type), index=[5, 7])

    assert_series_equal(new_series, desired)


def test_without_field_multiple_fields():
    """Test .nest.without_field(["field1", "field2"])"""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([4, 5, 6])]),
            pa.array([np.array([6, 4, 2]), np.array([1, 2, 3])]),
            pa.array([["a", "b", "c"], ["d", "e", "f"]]),
        ],
        names=["a", "b", "c"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[5, 7])

    new_series = series.nest.without_field(["a", "b"])

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([["a", "b", "c"], ["d", "e", "f"]]),
        ],
        names=["c"],
    )
    desired = pd.Series(desired_struct_array, dtype=NestedDtype(desired_struct_array.type), index=[5, 7])

    assert_series_equal(new_series, desired)


def test_without_field_raises_for_missing_field():
    """Test .nest.without_field("field") raises for missing field."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([4, 5, 6])]),
            pa.array([np.array([6, 4, 2]), np.array([1, 2, 3])]),
            pa.array([["a", "b", "c"], ["d", "e", "f"]]),
        ],
        names=["a", "b", "c"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[5, 7])

    with pytest.raises(ValueError):
        _ = series.nest.without_field("d")


def test_without_field_raises_for_missing_fields():
    """Test .nest.without_field(["field1", "field2"]) raises for missing fields."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([4, 5, 6])]),
            pa.array([np.array([6, 4, 2]), np.array([1, 2, 3])]),
            pa.array([["a", "b", "c"], ["d", "e", "f"]]),
        ],
        names=["a", "b", "c"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[5, 7])

    with pytest.raises(ValueError):
        _ = series.nest.without_field(["a", "d"])


def test_query_flat_1():
    """Test that the .nest.query_flat() method works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]),
            pa.array([np.array([6.0, 4.0, 2.0]), np.array([1.0, 2.0, 3.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[5, 7])

    filtered = series.nest.query_flat("a + b >= 7.0")

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0]), np.array([5.0, 6.0])]),
            pa.array([np.array([6.0]), np.array([2.0, 3.0])]),
        ],
        names=["a", "b"],
    )
    desired = pd.Series(desired_struct_array, dtype=NestedDtype(desired_struct_array.type), index=[5, 7])

    assert_series_equal(filtered, desired)


# Currently we remove empty rows from the output series
def test_query_flat_empty_rows():
    """Test that the .nest.query_flat() method works as expected for empty rows."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]),
            pa.array([np.array([6.0, 4.0, 2.0]), np.array([1.0, 2.0, 3.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[5, 7])

    filtered = series.nest.query_flat("a > 1000.0")
    desired = pd.Series([], dtype=series.dtype)

    assert_series_equal(filtered, desired)


def test_query_flat_with_empty_result():
    """Make sure the index is properly set for empty result cases"""
    base = npd.NestedFrame({"a": []}, index=pd.Index([], dtype=np.float64))
    nested = npd.NestedFrame({"b": []}, index=pd.Index([], dtype=np.float64))

    ndf = base.add_nested(nested, "nested")

    res = ndf.nested.nest.query_flat("b > 2")
    assert res.index.dtype == np.float64


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"a": [1] * 10}, index=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4]),
        pd.DataFrame(
            {"a": [1] * 10},
            index=pd.MultiIndex.from_arrays(([1, 1, 1, 1, 1, 1, 2, 2, 2, 2], [0, 1, 1, 2, 2, 2, 1, 1, 0, 0])),
        ),
        pd.DataFrame({"a": [1] * 10}, index=[1, 0, 0, 3, 3, 3, 0, 0, 0, 0]),
        pd.DataFrame(
            {"a": [1] * 6}, index=pd.MultiIndex.from_arrays(([0, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 2]))
        ),
    ],
)
def test_get_flat_index(df):
    """Test .nest.get_flat_index() returns the index of the original flat df"""
    series = pack_flat(df)
    assert_index_equal(series.nest.get_flat_index(), df.index.sort_values())


def test_get_list_series():
    """Test that the .nest.get_list_series() method works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([4, 5, 6])]),
            pa.array([np.array([6, 4, 2]), np.array([1, 2, 3])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[5, 7])

    lists = series.nest.get_list_series("a")

    assert_series_equal(
        lists,
        pd.Series(
            data=[np.array([1, 2, 3]), np.array([4, 5, 6])],
            dtype=pd.ArrowDtype(pa.list_(pa.int64())),
            index=[5, 7],
            name="a",
        ),
    )


def test_get():
    """Test .nest.get() which is implemented by the base class"""
    series = pack_seq(
        [
            pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 5.0, 6.0]}),
            pd.DataFrame({"a": [1, 2], "b": [None, 0.0]}),
            None,
        ]
    )
    assert_series_equal(series.nest.get("a"), series.nest.to_flat()["a"])
    assert_series_equal(series.nest.get("b"), series.nest.to_flat()["b"])
    assert series.nest.get("c", "default_value") == "default_value"


def test___getitem___single_field():
    """Test that the .nest["field"] works for a single field."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    assert_series_equal(
        series.nest["a"],
        pd.Series(
            np.array([1.0, 2.0, 3.0, 1.0, 2.0, 1.0]),
            dtype=pd.ArrowDtype(pa.float64()),
            index=[0, 0, 0, 1, 1, 1],
            name="a",
        ),
    )
    assert_series_equal(
        series.nest["b"],
        pd.Series(
            -np.array([4.0, 5.0, 6.0, 3.0, 4.0, 5.0]),
            dtype=pd.ArrowDtype(pa.float64()),
            index=[0, 0, 0, 1, 1, 1],
            name="b",
        ),
    )


def test___getitem___multiple_fields():
    """Test that the .nest[["b", "a"]] works for multiple fields."""
    arrays = [
        pa.array([np.array([1.0, 2.0, 3.0]), -np.array([1.0, 2.0, 1.0])]),
        pa.array([np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
    ]
    series = pd.Series(
        NestedExtensionArray(
            pa.StructArray.from_arrays(
                arrays=arrays,
                names=["a", "b"],
            )
        ),
        index=[0, 1],
    )

    assert_series_equal(
        series.nest[["b", "a"]],
        pd.Series(
            NestedExtensionArray(
                pa.StructArray.from_arrays(
                    arrays=arrays[::-1],
                    names=["b", "a"],
                )
            ),
            index=[0, 1],
        ),
    )


def test___setitem__():
    """Test that the .nest["field"] = ... works for a single field."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    series.nest["a"] = np.arange(6, 0, -1)

    assert_series_equal(
        series.nest["a"],
        pd.Series(
            data=[6, 5, 4, 3, 2, 1],
            index=[0, 0, 0, 1, 1, 1],
            name="a",
            dtype=pd.ArrowDtype(pa.float64()),
        ),
    )


def test___setitem___with_series_with_index():
    """Test that the .nest["field"] = pd.Series(...) works for a single field."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    flat_series = pd.Series(
        data=np.arange(6, 0, -1),
        index=[0, 0, 0, 1, 1, 1],
        name="a",
        dtype=pd.ArrowDtype(pa.float32()),
    )

    series.nest["a"] = flat_series

    assert_series_equal(
        series.nest["a"],
        flat_series.astype(pd.ArrowDtype(pa.float64())),
    )
    assert_series_equal(
        series.nest.get_list_series("a"),
        pd.Series(
            data=[np.array([6, 5, 4]), np.array([3, 2, 1])],
            dtype=pd.ArrowDtype(pa.list_(pa.float64())),
            index=[0, 1],
            name="a",
        ),
    )


def test___setitem___empty_series():
    """Test that series.nest["field"] = [] does nothing for empty series."""
    empty_series = pd.Series([], dtype=NestedDtype.from_fields({"a": pa.float64()}))
    empty_series.nest["a"] = []
    assert len(empty_series) == 0


def test___setitem___with_single_value():
    """Test series.nest["field"] = const"""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0])

    series.nest["a"] = -1.0

    assert_series_equal(
        series.nest["a"],
        pd.Series(
            data=[-1.0, -1.0, -1.0],
            index=[0, 0, 0],
            name="a",
            dtype=pd.ArrowDtype(pa.float64()),
        ),
    )


def test___setitem___raises_for_wrong_dtype():
    """Test that the .nest["field"] = ... raises for a wrong dtype."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    with pytest.raises(TypeError):
        series.nest["a"] = np.array(["a", "b", "c", "d", "e", "f"])


def test___setitem___raises_for_wrong_length():
    """Test that the .nest["field"] = ... raises for a wrong length."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    with pytest.raises(ValueError):
        series.nest["a"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def test___setitem___raises_for_wrong_index():
    """Test that the .nest["field"] = ... raises for a wrong index."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    flat_series = pd.Series(
        data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        index=[0, 1, 1, 1, 1, 1],
        name="a",
        dtype=pd.ArrowDtype(pa.float64()),
    )

    with pytest.raises(ValueError):
        series.nest["a"] = flat_series


def test___setitem___raises_for_new_field():
    """Test that series.nest["field"] = ... raises for a new field."""
    series = pack_seq([{"a": [1, 2, 3]}, {"a": [4, None]}])
    with pytest.raises(ValueError):
        series.nest["b"] = series.nest["a"] - 1


def test___delitem___raises():
    """Test that the `del .nest["field"]` is not implemented."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    with pytest.raises(AttributeError):
        del series.nest["a"]


def test___iter__():
    """Test that the iter(.nest) works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    assert_array_equal(list(series.nest), ["a", "b"])


def test___len__():
    """Test that the len(.nest) works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    assert len(series.nest) == 2


def test_to_flat_dropna():
    """Test that to_flat() gives a valid dataframe, based on GH22

    https://github.com/lincc-frameworks/nested-pandas/issues/22
    """

    flat = pd.DataFrame(
        data={"c": [0, 2, 4, 1, np.nan, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    nested = pack_flat(flat, name="nested")

    new_flat = nested.nest.to_flat()
    # .dropna() was failing in the issue report
    filtered = new_flat.dropna(subset="c")

    assert_frame_equal(
        filtered,
        pd.DataFrame(
            data={"c": [0.0, 2, 4, 1, 3, 1, 4, 1], "d": [5, 4, 7, 5, 1, 9, 3, 4]},
            index=[0, 0, 0, 1, 1, 2, 2, 2],
        ),
        check_dtype=False,  # filtered's Series are pd.ArrowDtype
    )


def test___contains__():
    """Test that the `"field" in .nest` works.

    We haven't implemented it, but base class does
    """
    series = pack_seq([pd.DataFrame({"a": [1, 2, 3]})])
    assert "a" in series.nest
    assert "x" not in series.nest


def test___eq__():
    """Test that one.nest == other.nest works."""

    series1 = pack_seq([pd.DataFrame({"a": [1, 2, 3]})])
    series2 = pack_seq([pd.DataFrame({"b": [1, 2, 3]})])
    series3 = pack_seq([pd.DataFrame({"a": [1, 2, 3, 4]})])
    series4 = pack_seq([pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})])

    assert series1.nest == series1.nest

    assert series2.nest == series2.nest
    assert series1.nest != series2.nest

    assert series3.nest == series3.nest
    assert series1.nest != series3.nest

    assert series4.nest == series4.nest
    assert series1.nest != series4.nest


def test___eq___false_for_different_types():
    """Test that one.nest == other.nest is False for different types."""
    seq = [{"a": [1, 2, 3]}, {"a": [4, None]}]
    series = pack_seq(seq)
    assert series.nest != pd.Series(seq, dtype=pd.ArrowDtype(pa.struct([("a", pa.list_(pa.int64()))])))


def test_clear_raises():
    """Test that .nest.clear() raises - we cannot handle nested series with no fields"""
    series = pack_seq([pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}), None])
    with pytest.raises(NotImplementedError):
        series.nest.clear()


def test_popitem_raises():
    """Test .nest.popitem() raises"""
    series = pack_seq(
        [pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}), pd.DataFrame({"a": [1, 2], "b": [2.0, None]}), None]
    )

    with pytest.raises(AttributeError):
        _ = series.nest.popitem()


def test_setdefault_raises():
    """Test .nest.setdefault() is not implemented"""
    series = pack_seq([{"a": [1, 2, 3]}, {"a": [4, None]}])
    with pytest.raises(AttributeError):
        series.nest.setdefault("b", series.nest["a"] * 2.0)


def test_update_raises():
    """test series.nest.update(other.nest) is not implemented"""
    series1 = pack_seq([{"a": [1, 2, 3], "b": [4, 5, 6]}, {"a": [4, None], "b": [7, 8]}])
    series2 = pack_seq(
        [
            {"b": ["x", "y", "z"], "c": [-2.0, -3.0, -4.0]},
            {"b": ["!", "?"], "c": [-5.0, -6.0]},
        ]
    )
    with pytest.raises(AttributeError):
        series1.nest.update(series2.nest)


def test_items():
    """Test series.nest.items() implemented by the base class"""
    series = pack_seq([{"a": [1, 2, 3], "b": [3, 2, 1]}, {"a": [4, None], "b": [7, 8]}])
    for key, value in series.nest.items():
        assert_series_equal(value, series.nest[key])


def test_keys():
    """Test series.nest.keys() implemented by the base class"""
    series = pack_seq([{"a": [1, 2, 3], "b": [3, 2, 1]}, {"a": [4, None], "b": [7, 8]}])
    assert_array_equal(list(series.nest.keys()), ["a", "b"])


def test_values():
    """Test series.nest.values() implemented by the base class"""
    series = pack_seq([{"a": [1, 2, 3], "b": [3, 2, 1]}, {"a": [4, None], "b": [7, 8]}])
    for value in series.nest.values():
        assert_series_equal(value, series.nest[value.name])
