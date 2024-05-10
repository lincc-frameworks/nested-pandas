import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas import NestedDtype
from nested_pandas.series.ext_array import NestedExtensionArray
from nested_pandas.series.packer import pack_flat
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


def test_set_flat_field():
    """Test that the .nest.set_flat_field() method works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    series.nest.set_flat_field("a", np.array(["a", "b", "c", "d", "e", "f"]))

    assert_series_equal(
        series.nest["a"],
        pd.Series(
            data=["a", "b", "c", "d", "e", "f"],
            index=[0, 0, 0, 1, 1, 1],
            name="a",
            dtype=pd.ArrowDtype(pa.string()),
        ),
    )


def test_set_list_field():
    """Test that the .nest.set_list_field() method works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    series.nest.set_list_field("c", [["a", "b", "c"], ["d", "e", "f"]])

    assert_series_equal(
        series.nest["c"],
        pd.Series(
            data=["a", "b", "c", "d", "e", "f"],
            index=[0, 0, 0, 1, 1, 1],
            name="c",
            dtype=pd.ArrowDtype(pa.string()),
        ),
    )


def test_pop_field():
    """Test that the .nest.pop_field() method works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    a = series.nest.pop_field("a")

    assert_array_equal(series.nest.fields, ["b"])
    assert_series_equal(
        a,
        pd.Series(
            [1.0, 2.0, 3.0, 1.0, 2.0, 1.0],
            dtype=pd.ArrowDtype(pa.float64()),
            index=[0, 0, 0, 1, 1, 1],
            name="a",
        ),
    )


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

    series.nest["a"] = np.array(["a", "b", "c", "d", "e", "f"])

    assert_series_equal(
        series.nest["a"],
        pd.Series(
            data=["a", "b", "c", "d", "e", "f"],
            index=[0, 0, 0, 1, 1, 1],
            name="a",
            dtype=pd.ArrowDtype(pa.string()),
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
        data=["a", "b", "c", "d", "e", "f"],
        index=[0, 0, 0, 1, 1, 1],
        name="a",
        dtype=pd.ArrowDtype(pa.string()),
    )

    series.nest["a"] = flat_series

    assert_series_equal(
        series.nest["a"],
        flat_series,
    )
    assert_series_equal(
        series.nest.get_list_series("a"),
        pd.Series(
            data=[np.array(["a", "b", "c"]), np.array(["d", "e", "f"])],
            dtype=pd.ArrowDtype(pa.list_(pa.string())),
            index=[0, 1],
            name="a",
        ),
    )


def test___setitem___empty_series():
    """Test that the series.nest["field"] = [] for empty series."""
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

    series.nest["a"] = 1.0
    assert_series_equal(
        series.nest["a"],
        pd.Series(
            data=[1.0, 1.0, 1.0],
            index=[0, 0, 0],
            name="a",
            dtype=pd.ArrowDtype(pa.float64()),
        ),
    )


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


def test___delitem__():
    """Test that the `del .nest["field"]` works."""
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=NestedDtype(struct_array.type), index=[0, 1])

    del series.nest["a"]

    assert_array_equal(series.nest.fields, ["b"])


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
        data={"c": [0, 2, 4, 1, np.NaN, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
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
