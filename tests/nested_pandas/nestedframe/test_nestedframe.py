import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas import NestedDtype, NestedFrame
from nested_pandas.datasets import generate_data
from nested_pandas.nestedframe.core import _SeriesFromNest
from nested_pandas.series.packer import pack_lists
from pandas.testing import assert_frame_equal


def test_nestedframe_construction():
    """Test NestedFrame construction"""
    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    assert isinstance(base, NestedFrame)
    assert_frame_equal(base, pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2]))

    list_struct_array = pa.array(
        [[{"x": 1, "y": 1.0}], [{"x": 2, "y": 2.0}], [{"x": 3, "y": 3.0}, {"x": 4, "y": 4.0}]]
    )
    list_struct_series = pd.Series(list_struct_array, dtype=pd.ArrowDtype(list_struct_array.type))
    nested_series = pd.Series(list_struct_series, dtype=NestedDtype(list_struct_array.type))

    nf = NestedFrame(base.to_dict(orient="series") | {"list_struct": list_struct_series})
    # Test auto-cast to nested
    assert_frame_equal(nf, base.assign(list_struct=nested_series))


def test_nestedseries_construction():
    """Test NestedSeries construction"""
    series = _SeriesFromNest([1, 2, 3], index=[0, 2, 4])

    assert isinstance(series, _SeriesFromNest)
    assert series[4] == 3

    # Exercise the constructor used during promoting operations
    combine_left = _SeriesFromNest([1, 2, 3], index=[0, 2, 4]) + pd.Series([1, 2, 3], index=[0, 2, 4])
    assert isinstance(combine_left, _SeriesFromNest)
    combine_right = pd.Series([1, 2, 3], index=[0, 2, 4]) + _SeriesFromNest([1, 2, 3], index=[0, 2, 4])
    assert isinstance(combine_right, _SeriesFromNest)

    # Exercising the expanddim constructor
    frame = series.to_frame()
    assert isinstance(frame, NestedFrame)
    assert (frame[0] == [1, 2, 3]).all()


def test_html_repr():
    """Just make sure the html representation code doesn't throw any errors"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    assert list(base.all_columns.keys()) == ["base"]
    assert list(base.all_columns["base"]) == list(base.columns)

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    # Check nested repr
    base._repr_html_()

    # Check repr path without nested cols
    base[["a", "b"]]._repr_html_()

    # Check repr truncation for larger nf
    nf = generate_data(100, 2)
    nf._repr_html_()


def test_all_columns():
    """Test the all_columns function"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    assert list(base.all_columns.keys()) == ["base"]
    assert list(base.all_columns["base"]) == list(base.columns)

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    assert list(base.all_columns.keys()) == ["base", "nested"]
    assert list(base.all_columns["nested"]) == list(nested.columns)


def test_nested_columns():
    """Test that nested_columns correctly retrieves the nested base columns"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    assert base.nested_columns == ["nested"]


def test_is_known_hierarchical_column():
    """Test that hierarchical column labels can be identified"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    assert base._is_known_hierarchical_column("nested.c")
    assert not base._is_known_hierarchical_column("nested.b")
    assert not base._is_known_hierarchical_column("base.a")


def test_is_known_column():
    """
    Test that known (non-hierarchical) columns can be identified.  The key
    point to test is that columns which might look like they are nested,
    but which are already known to not be, are correctly identified.
    """
    base = NestedFrame(data={"R. A.": [1, 2, 3], "nested.b": [2, 4, 6]}, index=[0, 1, 2])
    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    assert base._is_known_column("R. A.")
    assert base._is_known_column("`R. A.`")
    assert base._is_known_column("nested.b")

    # In this context, the "." delimiter matters a lot, so the following, which would be
    # acceptable in an .eval() context, is not acceptable here.
    assert not base._is_known_column("nested . b")
    assert not base._is_known_column("nested. c")
    assert not base._is_known_column("nested  .d")

    # But hierarchical ones should also work
    assert base._is_known_column("nested.c")
    assert base._is_known_column("nested.d")


def test_get_nested_column():
    """Test that __getitem__ can retrieve a nested column"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    base_c = base["nested.c"]

    # check basic properties
    assert isinstance(base_c, pd.Series)
    assert np.array_equal(np.array([0, 2, 4, 1, 4, 3, 1, 4, 1]), base_c.values.to_numpy())


def test_get_nested_columns():
    """Test that __getitem__ can retrieve a list including nested columns"""
    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    df = base[["a", "b", "nested.c"]]
    assert np.all(df.columns == ["a", "b", "nested"])
    assert df.dtypes["nested"].field_names == ["c"]
    assert np.all(df["nested"].iloc[0].columns == ["c"])

    df = base[["a", "b", "nested.c", "nested.d"]]
    assert np.all(df.columns == ["a", "b", "nested"])
    assert df.dtypes["nested"].field_names == ["c", "d"]
    assert np.all(df["nested"].iloc[0].columns == ["c", "d"])

    df = base[["a", "b", "nested.d", "nested.c"]]
    assert np.all(df.columns == ["a", "b", "nested"])
    assert df.dtypes["nested"].field_names == ["d", "c"]
    assert np.all(df["nested"].iloc[0].columns == ["d", "c"])

    df = base[["nested.c"]]
    assert np.all(df.columns == ["nested"])
    assert df.dtypes["nested"].field_names == ["c"]
    assert np.all(df["nested"].iloc[0].columns == ["c"])

    df = base[["a", "b"]]
    assert np.all(df.columns == ["a", "b"])

    df = base[["a", "b", "nested"]]
    assert np.all(df.columns == ["a", "b", "nested"])
    assert df.dtypes["nested"].field_names == ["c", "d"]
    assert np.all(df["nested"].iloc[0].columns == ["c", "d"])


def test_get_nested_columns_errors():
    """Test that __getitem__ errors with missing columns"""
    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    with pytest.raises(KeyError):
        base[["a", "c"]]

    with pytest.raises(KeyError):
        base[["a", "nested.g"]]

    with pytest.raises(KeyError):
        base[["a", "nested.a", "wrong.b"]]


def test_set_or_replace_nested_col():
    """Test that __setitem__ can set or replace a column in an existing nested structure"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])
    c = [0, 2, 4, 1, 4, 3, 1, 4, 1]
    nested = pd.DataFrame(
        data={"c": c, "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    # test direct replacement
    base["nested.c"] = base["nested.c"] + 1
    assert np.array_equal(np.array(c) + 1, base["nested.c"].values.to_numpy())

    # test += syntax
    base["nested.c"] += 1
    assert np.array_equal(
        np.array(c) + 2,  # 2 now, chained from above
        base["nested.c"].values.to_numpy(),
    )

    # test new column assignment
    base["nested.e"] = base["nested.d"] * 2

    assert "e" in base.nested.nest.fields
    assert np.array_equal(base["nested.d"].values.to_numpy() * 2, base["nested.e"].values.to_numpy())

    # test assignment a new column with list-repeated values
    base["nested.a"] = base["a"]

    assert "a" in base.nested.nest.fields
    assert np.array_equal(np.unique(base["a"].to_numpy()), np.unique(base["nested.a"].to_numpy()))

    # rest replacement with a list-repeated column
    base["nested.c"] = base["a"] + base["b"] - 99

    assert np.array_equal(base["a"] + base["b"] - 99, np.unique(base["nested.c"].to_numpy()))


def test_set_new_nested_col():
    """Test that __setitem__ can create a new nested structure"""
    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])
    c = [0, 2, 4, 1, 4, 3, 1, 4, 1]
    nested = pd.DataFrame(
        data={"c": c, "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    base = base.add_nested(nested, "nested")

    # assign column cd in new_nested from c+d in nested
    base["new_nested.cd"] = base["nested.c"] + base["nested.d"]

    assert "new_nested" in base.nested_columns
    assert "cd" in base["new_nested"].nest.fields

    assert np.array_equal(
        base["new_nested.cd"].values.to_numpy(),
        base["nested.c"].values.to_numpy() + base["nested.d"].values.to_numpy(),
    )


def test_set_item_combine_nested():
    """Test that operations like nf['new'] = nf[['c', 'd']] work as expected"""
    list_nf = NestedFrame(
        {
            "a": ["cat", "dog", "bird"],
            "b": [1, 2, 3],
            "c": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "d": [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
        }
    )

    list_nf = list_nf.nest_lists(["c"], "c")
    list_nf = list_nf.nest_lists(["d"], "d")

    list_nf["nested"] = list_nf[["c", "d"]]

    assert "nested" in list_nf.columns
    assert list_nf.nested.nest.fields == ["c", "d"]
    assert len(list_nf.nested.nest.to_flat()) == 9


def test_set_list_struct_col():
    """Test that __setitem__ would cast list-struct columns to nested."""
    nf = generate_data(10, 3)
    nf["a"] = nf["a"].astype(pd.ArrowDtype(pa.float64()))
    nf["b"] = nf["b"].astype(pd.ArrowDtype(pa.float64()))

    list_struct_array = pa.array(nf.nested)
    list_struct_series = pd.Series(list_struct_array, dtype=pd.ArrowDtype(list_struct_array.type))

    nf["nested2"] = list_struct_series
    assert_frame_equal(nf.nested.nest.to_flat(), nf.nested2.nest.to_flat())

    nf = nf.assign(nested3=list_struct_series)
    assert_frame_equal(nf.nested.nest.to_flat(), nf.nested3.nest.to_flat())


def test_get_dot_names():
    """Test the ability to still work with column names with '.' characters outside of nesting"""
    nf = NestedFrame.from_flat(
        NestedFrame({"a": [1, 2, 3, 4], ".b.": [1, 1, 3, 3], "R.A.": [3, None, 6, 5]}, index=[1, 1, 2, 2]),
        base_columns=[".b."],
    )

    assert len(nf[".b."]) == 2
    assert len(nf["nested.R.A."]) == 4


def test_nesting_limit():
    """Test the ability to prevent nesting beyond a depth of 1."""
    nf = NestedFrame.from_flat(
        NestedFrame({"a": [1, 2, 3, 4], ".b.": [1, 1, 3, 3], "R.A.": [3, None, 6, 5]}, index=[1, 1, 2, 2]),
        base_columns=[".b."],
    )
    with pytest.raises(ValueError):
        # The error gets triggered for the attempt to create new nested columns; if the column has
        # already been created, it should be fine.
        nf["nested.c.d.e"] = nf[".b."]
    nf["nested.c"] = nf["nested.R.A."]
    # Test that the above works with backticks, too, even in cases where they are not strictly necessary.
    nf["`nested.d`"] = nf["`.b.`"]


def test_add_nested_with_flat_df():
    """Test that add_nested correctly adds a nested column to the base df"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    assert "nested" in base.columns
    # to_flat() gives pd.ArrowDtype, so we skip dtype check here
    assert_frame_equal(base.nested.nest.to_flat(), nested, check_dtype=False)


def test_add_nested_with_flat_df_and_mismatched_index():
    """Test add_nested when index values of base are missing matches in nested"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6], "new_index": [0, 1, 3]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={
            "c": [0, 2, 4, 1, 4, 3, 1, 4, 1],
            "d": [5, 4, 7, 5, 3, 1, 9, 3, 4],
            # A column we can have as an alternative joining index with 'on'
            "new_index": [1, 1, 1, 1, 2, 2, 5, 5, 5],
        },
        # no data for base index value of "2" and introduces new index value "4"
        index=[0, 0, 0, 1, 1, 1, 1, 4, 4],
    )

    # Add the nested frame in a "left" fashion, where the index of the "left"
    # frame (our base layer) is preserved
    left_res = base.add_nested(nested, "nested", how="left")
    assert "nested" in left_res.columns
    # Check that the index of the base layer is being used
    assert (left_res.index == base.index).all()
    for idx in left_res.index:
        # Check that the nested column is aligned correctly to the base layer
        if idx in nested.index:
            assert left_res.loc[idx]["nested"] is not None
        else:  # idx only in base.index
            assert left_res.loc[idx]["nested"] is None

    # Test that the default behavior is the same as how="left" by comparing the pandas dataframes
    default_res = base.add_nested(nested, "nested")
    assert_frame_equal(left_res, default_res)

    # Test still adding the nested frame in a "left" fashion but on the "new_index" column

    # We currently don't support a list of columns for the 'on' argument
    with pytest.raises(ValueError):
        left_res_on = base.add_nested(nested, "nested", how="left", on=["new_index"])
    # Instead we should pass a single column name, "new_index" which exists in both frames.
    left_res_on = base.add_nested(nested, "nested", how="left", on="new_index")
    assert "nested" in left_res_on.columns
    # Check that the index of the base layer is still being used
    assert (left_res_on.index == base.index).all()
    # Assert that the new_index column we joined on was dropped from the nested layer
    # but is present in the base layer
    assert "new_index" in left_res_on.columns
    assert "new_index" not in left_res_on["nested"].nest.to_flat().columns

    # For each index in the columns we joined on, check that values are aligned correctly
    for i in range(len(left_res_on.new_index)):
        # The actual "index" value we "joined" on.
        join_idx = left_res_on.new_index.iloc[i]
        # Check that the nested column is aligned correctly to the base layer
        if join_idx in nested["new_index"].values:
            assert left_res_on.iloc[i]["nested"] is not None
            # Check that it is present in new the index we constructed for the nested layer
            assert join_idx in left_res_on["nested"].nest.to_flat().index
        else:
            # Use an iloc
            assert left_res_on.iloc[i]["nested"] is None
            assert join_idx not in left_res_on["nested"].nest.to_flat().index

    # Test adding the nested frame in a "right" fashion, where the index of the "right"
    # frame (our nested layer) is preserved
    right_res = base.add_nested(nested, "nested", how="right")
    assert "nested" in right_res.columns
    # Check that the index of the nested layer is being used. Note that separate
    # from a traditional join this will not be the same as our nested layer index
    # and is just dropping values from the base layer that don't have a match in
    # the nested layer.
    assert (right_res.index == nested.index.unique()).all()
    # For each index check that the base layer is aligned correctly to the nested layer
    for idx in right_res.index:
        # Check that the nested column is aligned correctly to the base layer. Here
        # it should never be None
        assert right_res.loc[idx]["nested"] is not None
        # Check the values for each column in our "base" layer
        for col in base.columns:
            assert col in right_res.columns
            if idx not in base.index:
                # We expect a NaN value in the base layer due to the "right" join
                assert pd.isna(right_res.loc[idx][col])
            else:
                assert not pd.isna(right_res.loc[idx][col])

    # Test still adding the nested frame in a "right" fashion but on the "new_index" column
    right_res_on = base.add_nested(nested, "nested", how="right", on="new_index")
    assert "nested" in right_res_on.columns
    # Check that rows were dropped if the base layer's "new_index" value is not present
    # in the "right" nested layer
    assert (right_res_on.new_index.values == np.unique(nested.new_index.values)).all()

    # Check that the new_index column we joined on was dropped from the nested layer
    assert "new_index" not in right_res_on["nested"].nest.to_flat().columns
    # Check that the flattend nested layer has the same index as the original column we joined on
    all(right_res_on.nested.nest.to_flat().index.values == nested.new_index.values)

    # For each index check that the base layer is aligned correctly to the nested layer
    for i in range(len(right_res_on)):
        # The actual "index" value we "joined" on. Since it was a right join, guaranteed to
        # be in the "new_index" column of the orignal frame we wanted to nest
        join_idx = right_res_on.new_index.iloc[i]
        assert join_idx in nested["new_index"].values

        # Check the values for each column in our "base" layer
        for col in base.columns:
            if col != "new_index":
                assert col in right_res_on.columns
                if join_idx not in base.new_index.values:
                    # We expect a NaN value in the base layer due to the "right" join
                    assert pd.isna(right_res_on.iloc[i][col])
                else:
                    assert not pd.isna(right_res_on.iloc[i][col])

    # Test the "outer" behavior
    outer_res = base.add_nested(nested, "nested", how="outer")
    assert "nested" in outer_res.columns
    # We expect the new index to be the union of the base and nested indices
    assert set(outer_res.index) == set(base.index).union(set(nested.index))
    for idx in outer_res.index:
        # Check that the nested column is aligned correctly to the base layer
        if idx in nested.index:
            assert outer_res.loc[idx]["nested"] is not None
        else:  # idx only in base.index
            assert outer_res.loc[idx]["nested"] is None
        # Check the values for each column in our "base" layer
        for col in base.columns:
            assert col in outer_res.columns
            if idx not in base.index:
                # We expect a NaN value in the base layer due to the "outer" join
                assert pd.isna(outer_res.loc[idx][col])
            else:
                assert not pd.isna(outer_res.loc[idx][col])

    # Test still adding the nested frame in an "outer" fashion but with on the "new_index" column
    outer_res_on = base.add_nested(nested, "nested", how="outer", on="new_index")
    assert "nested" in outer_res_on.columns
    # We expect the result's new_index column to be the set union of the values of that column
    # in the base and nested frames
    assert set(outer_res_on.new_index) == set(base.new_index).union(set(nested.new_index))

    # Check that the new_index column we joined on was dropped from the nested layer
    assert "new_index" not in outer_res_on["nested"].nest.to_flat().columns
    # Check that the flattend nested layer has the same index as the original column we joined on
    # Note that it does not have index values only present in the base layer since those empty rows
    # are dropped when we flatten the nested frame.
    all(outer_res_on.nested.nest.to_flat().index.values == nested.new_index.values)

    for i in range(len(outer_res_on)):
        # The actual "index" value we "joined" on.
        join_idx = outer_res_on.new_index.iloc[i]
        # Check that the nested column is aligned correctly to the base layer
        if join_idx not in nested["new_index"].values:
            assert outer_res_on.iloc[i]["nested"] is None
        else:
            assert outer_res_on.iloc[i]["nested"] is not None
        # Check the values for each column in our "base" layer
        for col in base.columns:
            if col != "new_index":
                assert col in outer_res_on.columns
                if join_idx in base.new_index.values:
                    # We expect a NaN value in the base layer due to the "outer" join
                    assert not pd.isna(outer_res_on.iloc[i][col])
                else:
                    assert pd.isna(outer_res_on.iloc[i][col])

    # Test the "inner" behavior
    inner_res = base.add_nested(nested, "nested", how="inner")
    assert "nested" in inner_res.columns
    # We expect the new index to be the set intersection of the base and nested indices
    assert set(inner_res.index) == set(base.index).intersection(set(nested.index))
    for idx in inner_res.index:
        # None of our nested values should be None
        assert inner_res.loc[idx]["nested"] is not None
        # Check the values for each column in our "base" layer
        for col in base.columns:
            assert col in inner_res.columns
            assert not pd.isna(inner_res.loc[idx][col])

    # Test still adding the nested frame in a "inner" fashion but on the "new_index" column
    inner_res_on = base.add_nested(nested, "nested", how="inner", on="new_index")
    assert "nested" in inner_res_on.columns
    # We expect the new index to be the set intersection of the base and nested column we used
    # for the 'on' argument
    assert set(inner_res_on.new_index) == set(base.new_index).intersection(set(nested.new_index))
    # Check that the new_index column we joined on was dropped from the nested layer
    assert "new_index" not in right_res_on["nested"].nest.to_flat().columns

    # Since we have confirmed that the "nex_index" column was the intersection that we expected
    # we know that none of the joined values should be none
    assert not inner_res_on.isnull().values.any()


def test_add_nested_with_series():
    """Test that add_nested correctly adds a nested column to the base df"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    nested = pd.Series(
        data=[pd.DataFrame({"c": [0, 1]}), pd.DataFrame({"c": [1, 2]}), pd.DataFrame({"c": [2, 3]})],
        index=[0, 1, 2],
        name="c",
    )

    base = base.add_nested(nested, "nested")

    assert "nested" in base.columns
    for i in range(3):
        assert_frame_equal(base.iloc[i]["nested"], nested[i])


def test_add_nested_with_series_and_mismatched_index():
    """Test add_nested when index values of base are missing matches in nested"""
    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    nested = pd.Series(
        data=[pd.DataFrame({"c": [0, 1]}), pd.DataFrame({"c": [2, 3]})],
        index=[0, 2],  # no data for index value of "1"
        name="c",
    )

    base = base.add_nested(nested, "nested")

    assert "nested" in base.columns
    assert pd.isna(base.loc[1]["nested"])


def test_add_nested_for_empty_df():
    """Test that .add_nested() works for empty frame and empty input"""
    base = NestedFrame(data={"a": [], "b": []}, index=[])
    nested = pd.DataFrame(data={"c": []}, index=[])
    new_base = base.add_nested(nested, "nested")

    # Check original frame is unchanged
    assert_frame_equal(base, NestedFrame(data={"a": [], "b": []}, index=[]))

    assert "nested" in new_base.columns
    assert_frame_equal(new_base.nested.nest.to_flat(), nested.astype(pd.ArrowDtype(pa.float64())))


@pytest.mark.parametrize("pandas", [False, True])
@pytest.mark.parametrize("on", [None, "a", "c"])
def test_from_flat(on, pandas):
    """Test the NestedFrame.from_flat functionality"""

    if pandas:
        nf = pd.DataFrame(
            {"a": [1, 1, 1, 2, 2], "b": [2, 2, 2, 4, 4], "c": [1, 2, 3, 4, 5], "d": [2, 4, 6, 8, 10]},
            index=[0, 0, 0, 1, 1],
        )
    else:
        nf = NestedFrame(
            {"a": [1, 1, 1, 2, 2], "b": [2, 2, 2, 4, 4], "c": [1, 2, 3, 4, 5], "d": [2, 4, 6, 8, 10]},
            index=[0, 0, 0, 1, 1],
        )

    out_nf = NestedFrame.from_flat(nf, base_columns=["a", "b"], on=on, name="new_nested")

    if on is None:
        assert list(out_nf.columns) == ["a", "b", "new_nested"]
        assert list(out_nf.new_nested.nest.fields) == ["c", "d"]
        assert len(out_nf) == 2
    elif on == "a":
        assert list(out_nf.columns) == ["b", "new_nested"]
        assert list(out_nf.new_nested.nest.fields) == ["c", "d"]
        assert len(out_nf) == 2
    elif on == "c":  # not what a user likely wants, but should still work
        assert list(out_nf.columns) == ["a", "b", "new_nested"]
        assert list(out_nf.new_nested.nest.fields) == ["d"]
        assert len(out_nf) == 5


def test_recover_from_flat():
    """test that going to_flat and then from_flat recovers the same df"""
    nf = generate_data(5, 10, seed=1)

    flat = nf["nested"].nest.to_flat()

    nf2 = NestedFrame.from_flat(nf[["a", "b"]].join(flat), base_columns=["a", "b"], name="nested")

    assert nf2.equals(nf)


def test_from_flat_omitting_columns():
    """test that from_flat successfully produces subsets"""
    flat = NestedFrame(
        {"a": [1, 1, 1, 2, 2], "b": [2, 2, 2, 4, 4], "c": [1, 2, 3, 4, 5], "d": [2, 4, 6, 8, 10]},
        index=[0, 0, 0, 1, 1],
    )

    # omit a base column
    nf = NestedFrame.from_flat(flat, base_columns=["b"], nested_columns=["c", "d"])
    assert list(nf.columns) == ["b", "nested"]
    assert list(nf.nested.nest.fields) == ["c", "d"]

    # omit a nested column
    nf = NestedFrame.from_flat(flat, base_columns=["a", "b"], nested_columns=["c"])
    assert list(nf.columns) == ["a", "b", "nested"]
    assert list(nf.nested.nest.fields) == ["c"]


def test_from_lists():
    """Test NestedFrame.from_lists behavior"""
    # Test a dataframe with no rows
    empty_nf = NestedFrame({"c": [], "d": [], "e": []}, index=[], columns=["c", "d", "e"])
    res = NestedFrame.from_lists(empty_nf, base_columns=["c", "d"], list_columns=["e"])
    assert list(res.columns) == ["c", "d", "nested"]
    assert list(res.nested_columns) == ["nested"]
    assert res.shape == (0, 3)

    nf = NestedFrame(
        {"c": [1, 2, 3], "d": [2, 4, 6], "e": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}, index=[0, 1, 2]
    )

    # Test a few combinations
    res = NestedFrame.from_lists(nf, base_columns=["c", "d"], name="nested_e")
    assert list(res.columns) == ["c", "d", "nested_e"]
    assert list(res.nested_columns) == ["nested_e"]

    res = NestedFrame.from_lists(nf, base_columns=["c", "d"], list_columns=["e"])
    assert list(res.columns) == ["c", "d", "nested"]
    assert list(res.nested_columns) == ["nested"]

    res = NestedFrame.from_lists(nf, list_columns=["e"])
    assert list(res.columns) == ["c", "d", "nested"]
    assert list(res.nested_columns) == ["nested"]

    # Test a using a non-iterable base column as a list column
    with pytest.raises(ValueError):
        res = NestedFrame.from_lists(nf, base_columns=["d"], list_columns=["e", "c"])

    # Check for the no list columns error
    with pytest.raises(ValueError):
        res = NestedFrame.from_lists(nf, base_columns=["c", "d", "e"])

    # Multiple list columns (of uneven length)
    nf2 = NestedFrame(
        {
            "c": [1, 2, 3],
            "d": [2, 4, 6],
            "e": [[1, 2, 3], [4, 5, 6, 7], [8, 9]],
            "f": [[10, 20, 30], [40, 50, 60, 70], [80, 90]],
        },
        index=[0, 1, 2],
    )

    res = NestedFrame.from_lists(nf2, list_columns=["e", "f"])
    assert list(res.columns) == ["c", "d", "nested"]
    assert list(res.nested_columns) == ["nested"]
    assert list(res.nested.nest.fields) == ["e", "f"]

    # Check for subsetting
    res = NestedFrame.from_lists(nf, base_columns=["c"], list_columns=["e"])
    assert list(res.columns) == ["c", "nested"]
    assert list(res.nested_columns) == ["nested"]

    res = NestedFrame.from_lists(nf, base_columns=[], list_columns=["e"])
    assert list(res.columns) == ["nested"]
    assert list(res.nested_columns) == ["nested"]

    res = NestedFrame.from_lists(nf[["e"]], base_columns=None, list_columns=None)
    assert list(res.columns) == ["nested"]
    assert list(res.nested_columns) == ["nested"]


def test_query():
    """Test that NestedFrame.query handles nested queries correctly"""

    base = NestedFrame(data={"a": [1, 2, 2, 3], "b": [2, 3, 4, 6]}, index=[0, 1, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    # Test vanilla queries
    base = base.add_nested(nested, "nested")
    assert len(base.query("a > 2")) == 1

    # Check for the multi-layer error
    with pytest.raises(ValueError):
        base.query("a > 2 & nested.c > 1")
    # Create another nest in order to further test the multi-layer error
    base_2 = base.eval("nest2.c = nested.c + 1")
    assert len(base_2.nested_columns) == 2
    with pytest.raises(ValueError):
        base_2.query("nested.c > 1 & nest2.c > 2")

    # Test nested queries
    nest_queried = base.query("nested.c > 1")
    assert len(nest_queried.nested.nest.to_flat()) == 7

    nest_queried = base.query("(nested.c > 1) and (nested.d>2)")
    assert len(nest_queried.nested.nest.to_flat()) == 5

    # Check edge conditions
    with pytest.raises(ValueError):
        # Expression must be a string
        base.query(3 + 4)

    # Verify that inplace queries will change the shape of the instance.
    base.query("(a % 2) == 1", inplace=True)
    assert base.shape == (2, 3)
    # A chunk of the nested rows will be gone, too.
    assert base["nested.c"].shape == (6,)
    assert base["nested.d"].shape == (6,)

    # Now query into the nest, throwing away most rows.  First, check that
    # without inplace=True, the original is not affected.
    assert base.query("nested.c + nested.d > 9")["nested.c"].shape == (2,)
    assert base.query("nested.c + nested.d > 9")["nested.d"].shape == (2,)
    # and verify the original:
    assert base["nested.c"].shape == (6,)
    assert base["nested.d"].shape == (6,)

    # Then, with inplace=True, 'base' should be changed in-place.
    base.query("nested.c + nested.d > 9", inplace=True)
    assert base["nested.c"].shape == (2,)
    assert base["nested.d"].shape == (2,)


def test_query_on_non_identifier_columns():
    """
    Column names very often follow the same rules as Python identifiers, but
    they are not required to.  Test that query() can handle such names.
    """
    # Taken from GH#174
    nf = NestedFrame(data={"dog": [1, 2, 3], "good dog": [2, 4, 6]}, index=[0, 1, 2])
    nested = pd.DataFrame(
        data={"a": [0, 2, 4, 1, 4, 3, 1, 4, 1], "b": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    nf = nf.add_nested(nested, "bad dog")
    nf2 = nf.query("`good dog` > 3")
    assert nf.shape == (3, 3)
    assert nf2.shape == (2, 3)
    nf3 = nf.query("`bad dog`.a > 2")
    assert nf3["bad dog"].nest["a"].size == 4

    # And also for fields within the nested columns.
    # Taken from GH#176
    nf = NestedFrame(data={"dog": [1, 2, 3], "good dog": [2, 4, 6]}, index=[0, 1, 2])
    nested = pd.DataFrame(
        data={"n/a": [0, 2, 4, 1, 4, 3, 1, 4, 1], "n/b": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    nf = nf.add_nested(nested, "bad dog")
    nf4 = nf.query("`bad dog`.`n/a` > 2")
    assert nf4["bad dog"].nest["n/a"].size == 4


def test_dropna():
    """Test that dropna works on all layers"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, np.nan, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, np.nan, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    # Test basic functionality
    dn_base = base.dropna(subset=["b"])
    assert len(dn_base) == 2
    assert len(dn_base["nested"].nest.to_flat() == 6)

    # Test on_nested kwarg
    dn_on_nested = base.dropna(on_nested="nested")
    assert len(dn_on_nested) == 3
    assert len(dn_on_nested["nested"].nest.to_flat() == 8)

    # Test hierarchical column subset
    dn_hierarchical = base.dropna(subset="nested.c")
    assert len(dn_hierarchical) == 3
    assert len(dn_hierarchical["nested"].nest.to_flat() == 8)

    # Test hierarchical column subset and on_nested
    dn_hierarchical = base.dropna(on_nested="nested", subset="nested.c")
    assert len(dn_hierarchical) == 3
    assert len(dn_hierarchical["nested"].nest.to_flat() == 8)


def test_dropna_layer_as_base_column():
    """Test that a nested column still works as a top level column for dropna"""
    nf = generate_data(10, 100, seed=1).query("nested.t>19.75")
    nf = nf.dropna(subset=["nested"])

    # make sure rows have been dropped as expected
    assert len(nf) == 6


def test_dropna_inplace_base():
    """Test in-place behavior of dropna"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [np.nan, 4, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, np.nan, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    # Test inplace=False with base layer
    dn_base = base.dropna(subset=["b"], inplace=False)
    assert not dn_base.equals(base)

    # Test inplace=True with base layer
    base.dropna(subset=["b"], inplace=True)
    assert dn_base.equals(base)


def test_dropna_inplace_nested():
    """Test in-place behavior of dropna"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [np.nan, 4, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, np.nan, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    # Test inplace=False with nested layer
    dn_base = base.dropna(on_nested="nested", inplace=False)
    assert not dn_base.nested.nest.to_flat().equals(base.nested.nest.to_flat())

    # Test inplace=True with nested layer
    base.dropna(on_nested="nested", inplace=True)
    assert dn_base.equals(base)


def test_dropna_errors():
    """Test that the various dropna exceptions trigger"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, np.nan, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, np.nan, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    # Test multi-target
    with pytest.raises(ValueError):
        base.dropna(subset=["b", "nested.c"])

    # Test no-target
    with pytest.raises(ValueError):
        base.dropna(subset=["not_nested.c"])

    # Test bad on-nested value
    with pytest.raises(ValueError):
        base.dropna(on_nested="not_nested")

    # Test on-nested + subset disagreement
    with pytest.raises(ValueError):
        base.dropna(on_nested="nested", subset=["b"])


def test_sort_values():
    """Test that sort_values works on all layers"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 3, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    # Test basic functionality
    sv_base = base.sort_values("b")
    assert list(sv_base.index) == [0, 1, 2]

    # Test on nested column
    sv_base = base.sort_values(["nested.d"])
    assert list(sv_base.iloc[0]["nested"]["d"]) == [4, 5, 7]

    # Test multi-layer error trigger
    with pytest.raises(ValueError):
        base.sort_values(["a", "nested.c"])

    # Test inplace=True
    base.sort_values("nested.d", inplace=True)
    assert list(base.iloc[0]["nested"]["d"]) == [4, 5, 7]


def test_sort_values_ascension():
    """Test that sort_values works with various ascending settings"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 3, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    # Test ascending=False
    sv_base = base.sort_values("nested.d", ascending=False)
    assert list(sv_base.iloc[0]["nested"]["d"]) == [7, 5, 4]

    # Test list ascending
    sv_base = base.sort_values("nested.d", ascending=[False])
    assert list(sv_base.iloc[0]["nested"]["d"]) == [7, 5, 4]

    # Test multi-by multi-ascending
    sv_base = base.sort_values(["nested.d", "nested.c"], ascending=[False, True])
    assert list(sv_base.iloc[0]["nested"]["d"]) == [7, 5, 4]


def test_reduce():
    """Tests that we can call reduce on a NestedFrame with a custom function."""
    nf = NestedFrame(
        data={"a": [1, 2, 3], "b": [2, 4, 6]},
        index=pd.Index([0, 1, 2], name="idx"),
    )

    to_pack = pd.DataFrame(
        data={
            "time": [1, 2, 3, 1, 2, 4, 2, 1, 4],
            "c": [0, 2, 4, 10, 4, 3, 1, 4, 1],
            "d": [5, 4, 7, 5, 3, 1, 9, 3, 4],
        },
        index=pd.Index([0, 0, 0, 1, 1, 1, 2, 2, 2], name="idx"),
    )

    to_pack2 = pd.DataFrame(
        data={
            "time2": [
                1,
                2,
                3,
                1,
                2,
                3,
                1,
                2,
                4,
            ],  # TODO: fix duplicate name in join once to_list subset bug fixed
            "e": [2, 9, 4, 1, 23, 3, 1, 4, 1],
            "f": [5, 4, 7, 5, 3, 25, 9, 3, 4],
        },
        index=pd.Index([0, 0, 0, 1, 1, 1, 2, 2, 2], name="idx"),
    )

    # Add two nested layers to pack into our dataframe
    nf = nf.add_nested(to_pack, "packed").add_nested(to_pack2, "packed2")

    # Define a simple custom function to apply to the nested data
    def get_max(col1, col2):
        # returns the max value within each specified colun
        return pd.Series([col1.max(), col2.max()], index=["max_col1", "max_col2"])

    # The expected max values for of our nested columns
    expected_max_c = [4, 10, 4]
    expected_max_d = [7, 5, 9]
    expected_max_e = [9, 23, 4]

    # Test that we raise an error when no arguments are provided
    with pytest.raises(ValueError):
        nf.reduce(get_max)

    # Batch only on columns in the first packed layer
    result = nf.reduce(get_max, "packed.c", "packed.d")
    assert len(result) == len(nf)
    assert isinstance(result, NestedFrame)
    assert result.index.name == "idx"
    for i in range(len(result)):
        assert result["max_col1"].values[i] == expected_max_c[i]
        assert result["max_col2"].values[i] == expected_max_d[i]

    # Batch on columns in the first and second packed layers
    result = nf.reduce(get_max, "packed.c", "packed2.e")
    assert len(result) == len(nf)
    assert isinstance(result, NestedFrame)
    assert result.index.name == "idx"
    for i in range(len(result)):
        assert result["max_col1"].values[i] == expected_max_c[i]
        assert result["max_col2"].values[i] == expected_max_e[i]

    # Test that we can pass a scalar from the base layer to the reduce function and that
    # the user can also provide non-column arguments (in this case, the list of column names)
    def offset_avg(offset, col_to_avg, column_names):
        # A simple function which adds a scalar 'offset' to a column which is then averaged.
        return pd.Series([(offset + col_to_avg).mean()], index=column_names)

    expected_offset_avg = [
        sum([2, 4, 6]) / 3.0,
        sum([14, 8, 7]) / 3.0,
        sum([7, 10, 7]) / 3.0,
    ]

    result = nf.reduce(offset_avg, "b", "packed.c", column_names=["offset_avg"])
    assert len(result) == len(nf)
    assert isinstance(result, NestedFrame)
    assert result.index.name == "idx"
    for i in range(len(result)):
        assert result["offset_avg"].values[i] == expected_offset_avg[i]

    # Verify that we can understand a string argument to the reduce function,
    # so long as it isn't a column name.
    def make_id(col1, prefix_str):
        return f"{prefix_str}{col1}"

    result = nf.reduce(make_id, "b", prefix_str="some_id_")
    assert result[0][1] == "some_id_4"

    # Verify that append_columns=True works as expected.
    # Ensure that even with non-unique indexes, the final result retains
    # the original index (nested-pandas#301)
    nf.index = pd.Index([0, 1, 1], name="non-unique")
    result = nf.reduce(get_max, "packed.c", "packed.d", append_columns=True)
    assert len(result) == len(nf)
    assert isinstance(result, NestedFrame)
    result_c = list(result.columns)
    nf_c = list(nf.columns)
    # The result should have the original columns plus the new max columns
    assert result_c[: len(nf_c)] == nf_c
    assert result_c[len(nf_c) :] == ["max_col1", "max_col2"]
    assert result.index.name == "non-unique"
    assert list(result.index) == [0, 1, 1]
    for i in range(len(result)):
        assert result["max_col1"].values[i] == expected_max_c[i]
        assert result["max_col2"].values[i] == expected_max_d[i]
        # The original columns should still be present
        assert result["packed.c"].values[i] == to_pack["c"].values[i]
        assert result["packed.d"].values[i] == to_pack["d"].values[i]


def test_reduce_duplicated_cols():
    """Tests nf.reduce() to correctly handle duplicated column names."""
    nf = NestedFrame(
        data={"a": [1, 2, 3], "b": [2, 4, 6]},
        index=pd.Index([0, 1, 2], name="idx"),
    )

    to_pack = pd.DataFrame(
        data={
            "time": [1, 2, 3, 1, 2, 4, 2, 1, 4],
            "c": [0, 2, 4, 10, 4, 3, 1, 4, 1],
            "d": [5, 4, 7, 5, 3, 1, 9, 3, 4],
        },
        index=pd.Index([0, 0, 0, 1, 1, 1, 2, 2, 2], name="idx"),
    )

    to_pack2 = pd.DataFrame(
        data={
            "time2": [
                1,
                2,
                3,
                1,
                2,
                3,
                1,
                2,
                4,
            ],  # TODO: fix duplicate name in join once to_list subset bug fixed
            "e": [2, 9, 4, 1, 23, 3, 1, 4, 1],
            "f": [5, 4, 7, 5, 3, 25, 9, 3, 4],
        },
        index=pd.Index([0, 0, 0, 1, 1, 1, 2, 2, 2], name="idx"),
    )

    # Add two nested layers to pack into our dataframe
    nf = nf.add_nested(to_pack, "packed").add_nested(to_pack2, "packed2")

    def cols_allclose(col1, col2):
        return pd.Series([np.allclose(col1, col2)], index=["allclose"])

    result = nf.reduce(cols_allclose, "packed.time", "packed2.f")
    assert_frame_equal(
        result, pd.DataFrame({"allclose": [False, False, False]}, index=pd.Index([0, 1, 2], name="idx"))
    )

    result = nf.reduce(cols_allclose, "packed.c", "packed.c")
    assert_frame_equal(
        result, pd.DataFrame({"allclose": [True, True, True]}, index=pd.Index([0, 1, 2], name="idx"))
    )


def test_reduce_infer_nesting():
    """Test that nesting inference works in reduce"""

    ndf = generate_data(3, 20, seed=1)

    # Test simple case
    def complex_output(flux):
        return {
            "max_flux": np.max(flux),
            "lc.flux_quantiles": np.quantile(flux, [0.1, 0.2, 0.3, 0.4, 0.5]),
        }

    result = ndf.reduce(complex_output, "nested.flux")
    assert list(result.columns) == ["max_flux", "lc"]
    assert list(result.lc.nest.fields) == ["flux_quantiles"]

    # Test multi-column nested output
    def complex_output(flux):
        return {
            "max_flux": np.max(flux),
            "lc.flux_quantiles": np.quantile(flux, [0.1, 0.2, 0.3, 0.4, 0.5]),
            "lc.labels": [0.1, 0.2, 0.3, 0.4, 0.5],
        }

    result = ndf.reduce(complex_output, "nested.flux")
    assert list(result.columns) == ["max_flux", "lc"]
    assert list(result.lc.nest.fields) == ["flux_quantiles", "labels"]

    # Test integer names
    def complex_output(flux):
        return np.max(flux), np.quantile(flux, [0.1, 0.2, 0.3, 0.4, 0.5]), [0.1, 0.2, 0.3, 0.4, 0.5]

    result = ndf.reduce(complex_output, "nested.flux")
    assert list(result.columns) == [0, 1, 2]

    # Test multiple nested structures output
    def complex_output(flux):
        return {
            "max_flux": np.max(flux),
            "lc.flux_quantiles": np.quantile(flux, [0.1, 0.2, 0.3, 0.4, 0.5]),
            "lc.labels": [0.1, 0.2, 0.3, 0.4, 0.5],
            "meta.colors": ["green", "red", "blue"],
        }

    result = ndf.reduce(complex_output, "nested.flux")
    assert list(result.columns) == ["max_flux", "lc", "meta"]
    assert list(result.lc.nest.fields) == ["flux_quantiles", "labels"]
    assert list(result.meta.nest.fields) == ["colors"]

    # Test only nested structure output
    def complex_output(flux):
        return {
            "lc.flux_quantiles": np.quantile(flux, [0.1, 0.2, 0.3, 0.4, 0.5]),
            "lc.labels": [0.1, 0.2, 0.3, 0.4, 0.5],
        }

    result = ndf.reduce(complex_output, "nested.flux")
    assert list(result.columns) == ["lc"]
    assert list(result.lc.nest.fields) == ["flux_quantiles", "labels"]


def test_reduce_arg_errors():
    """Test that reduce errors based on non-column args trigger as expected"""

    ndf = generate_data(10, 10, seed=1)

    def func(a, flux, add):
        """a function that takes a scalar, a column, and a boolean"""
        if add:
            return {"nested2.flux": flux + a}
        return {"nested2.flux": flux + a}

    with pytest.raises(TypeError):
        ndf.reduce(func, "a", "nested.flux", True)

    with pytest.raises(ValueError):
        ndf.reduce(func, "ab", "nested.flux", add=True)

    # this should work
    ndf.reduce(func, "a", "nested.flux", add=True)


def test_scientific_notation():
    """
    Test that NestedFrame.query handles constants that are written in scientific notation.
    """
    # https://github.com/lincc-frameworks/nested-pandas/issues/59
    base = NestedFrame({"a": [1, 1e-2, 3]}, index=[0, 1, 2])
    selected = base.query("a > 1e-1")
    assert list(selected.index) == [0, 2]


def test_drop():
    """Test that we can drop nested columns from a NestedFrame"""
    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])
    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    nested2 = pd.DataFrame(
        data={"e": [0, 2, 4, 1, 4, 3, 1, 4, 1], "f": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    base = base.add_nested(nested, "nested").add_nested(nested2, "nested2")

    # test axis=0 drop
    dropped_base = base.drop(0, axis=0)
    assert len(dropped_base) == len(base) - 1

    # Test dropping a base column
    dropped_base = base.drop("a", axis=1)
    assert len(dropped_base.columns) == len(base.columns) - 1
    assert "a" not in dropped_base.columns

    # Test dropping a nested column
    dropped_nested = base.drop("nested.c", axis=1)
    assert len(dropped_nested.columns) == len(base.columns)
    assert "c" not in dropped_nested.nested.nest.fields

    # Test dropping a non-existent column
    with pytest.raises(KeyError):
        base.drop("not_a_column", axis=1)

    # Test dropping multiple columns
    dropped_multiple = base.drop(["a", "nested.c"], axis=1)
    assert len(dropped_multiple.columns) == len(base.columns) - 1
    assert "a" not in dropped_multiple.columns
    assert "c" not in dropped_multiple.nested.nest.fields

    # Test multiple nested structures
    dropped_multiple = base.drop(["nested.c", "nested2.f"], axis=1)
    assert len(dropped_multiple.columns) == len(base.columns)
    assert "c" not in dropped_multiple.nested.nest.fields
    assert "f" not in dropped_multiple.nested2.nest.fields

    # Test inplace=True for both base and nested columns
    base2 = base.copy()
    base2.drop(["a", "nested.c"], axis=1, inplace=True)
    assert "a" not in base2.columns
    assert "c" not in base2["nested"].nest.fields
    assert "b" in base2.columns
    assert "d" in base2["nested"].nest.fields

    # Test inplace=False for both base and nested columns
    base3 = base.copy()
    dropped = base3.drop(["a", "nested.c"], axis=1, inplace=False)
    assert "a" not in dropped.columns
    assert "c" not in dropped["nested"].nest.fields
    assert "b" in dropped.columns
    assert "d" in dropped["nested"].nest.fields
    # Original is unchanged
    assert "a" in base3.columns
    assert "c" in base3["nested"].nest.fields

    # Test error for missing columns in multi-drop
    with pytest.raises(KeyError):
        base.drop(["not_a_column", "nested.c"], axis=1)
    with pytest.raises(KeyError):
        base.drop(["a", "nested.not_a_field"], axis=1)


def test_min():
    """Test min function return correct result with and without the nested columns"""
    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6], "c": ["x", "y", "z"]}, index=[0, 1, 2])
    nested = pd.DataFrame(
        data={"d": [10, 11, 20, 21, 3, 31, 32], "y": [1, 10, 20, 30, 40, 50, 60]}, index=[0, 0, 1, 1, 1, 2, 2]
    )
    nested2 = pd.DataFrame(
        data={"e": [0, 2, 4, 1, 4, 1, 4, 1], "f": [5, 4, 7, 5, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 2, 2, 2],
    )

    # only base columns
    r0 = base.min(numeric_only=True)
    assert (r0 == pd.Series({"a": 1, "b": 2})).all()
    r1 = base.min()
    assert (r1 == pd.Series({"a": 1, "b": 2, "c": "x"})).all()

    # nan tests
    nested_clean = pd.DataFrame(
        data={"g": [1, 0, 3, 4, 5, 6], "h": [1, 2, 3, 4, 5, 6]}, index=[0, 0, 1, 1, 2, 2]
    )
    base_clean = base.add_nested(nested_clean, "nested_clean")
    min_clean = base_clean.min()
    expected_clean = pd.Series({"a": 1, "b": 2, "c": "x", "nested_clean.g": 0, "nested_clean.h": 1})
    assert (min_clean == expected_clean).all()

    nested_nan = pd.DataFrame(
        data={"g": [1, np.nan, 3, 4, 5, 6], "h": [np.nan, np.nan, 3, 4, np.nan, 6]}, index=[0, 0, 1, 1, 2, 2]
    )
    base_nan = base.add_nested(nested_nan, "nested_nan")
    min_nan = base_nan.min()
    assert isinstance(min_nan, pd.Series)
    expected_nan = pd.Series({"a": 1, "b": 2, "c": "x", "nested_nan.g": 1, "nested_nan.h": 3})
    assert (min_nan == expected_nan).all()

    # 1 nested column
    base = base.add_nested(nested, "nested")
    r2 = base.min(exclude_nest=True, numeric_only=True)
    assert (r2 == pd.Series({"a": 1, "b": 2})).all()
    r3 = base.min(exclude_nest=True)
    assert (r3 == pd.Series({"a": 1, "b": 2, "c": "x"})).all()
    r4 = base.min()
    expected4 = pd.Series({"a": 1, "b": 2, "c": "x", "nested.d": 3, "nested.y": 1})
    assert (r4 == expected4).all()

    # 2 nested columns
    base = base.add_nested(nested2, "nested2")
    r5 = base.min(exclude_nest=True, numeric_only=True)
    assert (r5 == pd.Series({"a": 1, "b": 2})).all()
    r6 = base.min(exclude_nest=True)
    assert (r6 == pd.Series({"a": 1, "b": 2, "c": "x"})).all()
    r7 = base.min()
    expected7 = pd.Series(
        {
            "a": 1,
            "b": 2,
            "c": "x",
            "nested.d": 3,
            "nested.y": 1,
            "nested2.e": 0,
            "nested2.f": 1,
        }
    )
    assert (r7 == expected7).all()

    # only nested column
    base2 = NestedFrame(data={"x": [0, 1, 2]}, index=[0, 1, 2])
    nested3 = NestedFrame(data={"a": [1, 2, 3, 4, 5, 6], "b": [2, 4, 6, 8, 9, 0]}, index=[0, 0, 1, 1, 1, 2])
    base2 = base2.add_nested(nested3, "nested3")
    base2 = base2.drop(["x"], axis=1)
    r8 = base2.min(exclude_nest=True)
    assert isinstance(r8, pd.Series)
    assert r8.empty
    r9 = base2.min()
    assert (r9 == pd.Series({"nested3.a": 1, "nested3.b": 0})).all()


def test_max():
    """Test max function return correct result with an without the nested columns"""
    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6], "c": ["x", "y", "z"]}, index=[0, 1, 2])
    nested = pd.DataFrame(
        data={"d": [10, 11, 20, 21, 3, 31, 32], "y": [1, 10, 20, 30, 40, 50, 60]}, index=[0, 0, 1, 1, 1, 2, 2]
    )
    nested2 = pd.DataFrame(
        data={"e": [0, 2, 4, 1, 4, 1, 4, 1], "f": [5, 4, 7, 5, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 2, 2, 2],
    )

    # only base columns
    r0 = base.max(numeric_only=True)
    assert (r0 == pd.Series({"a": 3, "b": 6})).all()
    r1 = base.max()
    assert (r1 == pd.Series({"a": 3, "b": 6, "c": "z"})).all()

    # nan tests
    nested_clean = pd.DataFrame(
        data={"g": [1, 0, 3, 4, 5, 6], "h": [1, 2, 3, 4, 5, 6]}, index=[0, 0, 1, 1, 2, 2]
    )
    base_clean = base.add_nested(nested_clean, "nested_clean")
    max_clean = base_clean.max()
    expected_clean = pd.Series({"a": 3, "b": 6, "c": "z", "nested_clean.g": 6, "nested_clean.h": 6})
    assert (max_clean == expected_clean).all()

    nested_nan = pd.DataFrame(
        data={"g": [1, np.nan, 3, 4, np.nan, np.nan], "h": [np.nan, np.nan, 3, 4, 5, np.nan]},
        index=[0, 0, 1, 1, 2, 2],
    )
    base_nan = base.add_nested(nested_nan, "nested_nan")
    max_nan = base_nan.max()
    assert isinstance(max_nan, pd.Series)
    expected_nan = pd.Series({"a": 3, "b": 6, "c": "z", "nested_nan.g": 4, "nested_nan.h": 5})
    assert (max_nan == expected_nan).all()

    # 1 nested column
    base = base.add_nested(nested, "nested")
    r2 = base.max(exclude_nest=True, numeric_only=True)
    assert (r2 == pd.Series({"a": 3, "b": 6})).all()
    r3 = base.max(exclude_nest=True)
    assert (r3 == pd.Series({"a": 3, "b": 6, "c": "z"})).all()
    r4 = base.max()
    expected4 = pd.Series({"a": 3, "b": 6, "c": "z", "nested.d": 32, "nested.y": 60})
    assert (r4 == expected4).all()

    # 2 nested columns
    base = base.add_nested(nested2, "nested2")
    r5 = base.max(exclude_nest=True, numeric_only=True)
    assert (r5 == pd.Series({"a": 3, "b": 6})).all()
    r6 = base.max(exclude_nest=True)
    assert (r6 == pd.Series({"a": 3, "b": 6, "c": "z"})).all()
    r7 = base.max()
    expected7 = pd.Series(
        {
            "a": 3,
            "b": 6,
            "c": "z",
            "nested.d": 32,
            "nested.y": 60,
            "nested2.e": 4,
            "nested2.f": 9,
        }
    )
    assert (r7 == expected7).all()

    # only nested column
    base2 = NestedFrame(data={"x": [0, 1, 2]}, index=[0, 1, 2])
    nested3 = NestedFrame(data={"a": [1, 2, 3, 4, 5, 6], "b": [2, 4, 6, 8, 9, 0]}, index=[0, 0, 1, 1, 1, 2])
    base2 = base2.add_nested(nested3, "nested3")
    base2 = base2.drop(["x"], axis=1)
    r8 = base2.max(exclude_nest=True)
    assert isinstance(r8, pd.Series)
    assert r8.empty
    r9 = base2.max()
    assert (r9 == pd.Series({"nested3.a": 6, "nested3.b": 9})).all()


def test_eval():
    """
    Test basic behavior of NestedFrame.eval, and that it can handle nested references
    the same as the nest accessor.
    """
    nf = NestedFrame(
        data={"a": [1, 2, 3], "b": [2, 4, 6]},
        index=pd.Index([0, 1, 2], name="idx"),
    )

    to_pack = pd.DataFrame(
        data={
            "time": [1, 2, 3, 1, 2, 4, 2, 1, 4],
            "c": [0, 2, 4, 10, 4, 3, 1, 4, 1],
            "d": [5, 4, 7, 5, 3, 1, 9, 3, 4],
        },
        index=pd.Index([0, 0, 0, 1, 1, 1, 2, 2, 2], name="idx"),
    )

    nf = nf.add_nested(to_pack, "packed")
    p5 = nf.eval("packed.d > 5")
    assert isinstance(p5, _SeriesFromNest)
    assert p5.any()
    assert not p5.all()
    assert list(p5.loc[p5].index) == [0, 2]

    r1 = nf.eval("packed.c + packed.d")
    r2 = nf["packed"].nest["c"] + nf["packed"].nest["d"]
    r3 = nf["packed.c"] + nf["packed.d"]
    assert (r1 == r2).all()
    assert (r2 == r3).all()


def test_eval_funcs():
    """
    Test the ability to use expected methods and functions within eval(),
    on nested columns.
    """
    # Verifies https://github.com/lincc-frameworks/nested-pandas/issues/146
    nf = NestedFrame.from_flat(NestedFrame({"a": [1, 2], "b": [3, None]}, index=[1, 1]), base_columns=[])
    assert nf["nested.b"].shape == (2,)
    assert nf.query("nested.b.isna()")["nested.b"].shape == (1,)

    assert nf["nested.a"].max() == nf.eval("nested.a.max()") == 2
    assert nf["nested.a"].min() == nf.eval("nested.a.min()") == 1


def test_mixed_eval_funcs():
    """
    Test operations across base and nested.  Whether these evaluations
    work is data-dependent, since the dimensions of the base and
    nested columns are not guaranteed to be compatible, but when they
    are, it should work as expected.
    """
    nf = NestedFrame(
        data={"a": [1, 2, 3], "b": [2, 4, 6]},
        index=pd.Index([0, 1, 2], name="idx"),
    )

    to_pack = pd.DataFrame(
        data={
            "time": [1, 2, 3, 1, 2, 4, 2, 1, 4],
            "c": [0, 2, 4, 10, 4, 3, 1, 4, 1],
            "d": [5, 4, 7, 5, 3, 1, 9, 3, 4],
        },
        index=pd.Index([0, 0, 0, 1, 1, 1, 2, 2, 2], name="idx"),
    )
    # Reduction
    nf = nf.add_nested(to_pack, "packed")
    assert (nf.eval("a + packed.c.median()") == pd.Series([4, 5, 6])).all()

    # Across the nest: each base column element applies to each of its indexes
    assert (nf.eval("a + packed.c") == nf["a"] + nf["packed.c"]).all()


def test_eval_assignment():
    """
    Test eval strings that perform assignment, within base columns, nested columns,
    and across base and nested.
    """
    nf = NestedFrame(
        data={"a": [1, 2, 3], "b": [2, 4, 6]},
        index=pd.Index([0, 1, 2], name="idx"),
    )
    to_pack = pd.DataFrame(
        data={
            "time": [1, 2, 3, 1, 2, 4, 2, 1, 4],
            "c": [0, 2, 4, 10, 4, 3, 1, 4, 1],
            "d": [5, 4, 7, 5, 3, 1, 9, 3, 4],
        },
        index=pd.Index([0, 0, 0, 1, 1, 1, 2, 2, 2], name="idx"),
    )
    nf = nf.add_nested(to_pack, "packed")
    # Assigning to new base columns from old base columns
    nf_b = nf.eval("c = a + 1")
    assert len(nf_b.columns) == len(nf.columns) + 1
    assert (nf_b["c"] == nf["a"] + 1).all()

    # Assigning to new nested columns from old nested columns
    nf_nc = nf.eval("packed.e = packed.c + 1")
    assert len(nf_nc.packed.nest.fields) == len(nf["packed"].nest.fields) + 1
    assert (nf_nc["packed.e"] == nf["packed.c"] + 1).all()

    # Verify that overwriting a nested column works
    nf_nc_2 = nf_nc.eval("packed.e = packed.c * 2")
    assert len(nf_nc_2.packed.nest.fields) == len(nf_nc["packed"].nest.fields)
    assert (nf_nc_2["packed.e"] == nf["packed.c"] * 2).all()

    # Assigning to new nested columns from a combo of base and nested
    nf_nx = nf.eval("packed.f = a + packed.c")
    assert len(nf_nx.packed.nest.fields) == len(nf["packed"].nest.fields) + 1
    assert (nf_nx["packed.f"] == nf["a"] + nf["packed.c"]).all()
    assert (nf_nx["packed.f"] == pd.Series([1, 3, 5, 12, 6, 5, 4, 7, 4], index=to_pack.index)).all()

    # Only supporting one level of nesting at present.
    with pytest.raises(ValueError):
        nf.eval("packed.c.inner = packed.c * 2 + packed.d")

    # Assigning to new base columns from nested columns.  This can't be done because
    # it would attempt to create base column values that were "between indexes", or as
    # Pandas puts, duplicate index labels.
    with pytest.raises(ValueError):
        nf.eval("g = packed.c * 2")

    # Create new nests via eval()
    nf_n2 = nf.eval("p2.c2 = packed.c * 2")
    assert len(nf_n2.p2.nest.fields) == 1
    assert (nf_n2["p2.c2"] == nf["packed.c"] * 2).all()
    assert (nf_n2["p2.c2"] == pd.Series([0, 4, 8, 20, 8, 6, 2, 8, 2], index=to_pack.index)).all()
    assert len(nf_n2.columns) == len(nf.columns) + 1  # new packed column
    assert len(nf_n2.p2.nest.fields) == 1

    # Assigning to new columns across two different nests
    nf_n3 = nf_n2.eval("p2.d = p2.c2 + packed.d * 2 + b")
    assert len(nf_n3.p2.nest.fields) == 2
    assert (nf_n3["p2.d"] == nf_n2["p2.c2"] + nf["packed.d"] * 2 + nf["b"]).all()

    # Now test multiline and inplace=True
    # Verify the resolution of GH#159, where a nested column created in
    # an existing nest during a multi-line eval was not being recognized
    # in a subsequent line.
    nf.eval(
        """
        c = a + b
        packed.e = packed.d * 2
        p2.e = packed.e + c
        p2.f = p2.e + b
        """,
        inplace=True,
    )
    assert set(nf.nested_columns) == {"packed", "p2"}
    assert set(nf.packed.nest.fields) == {"c", "d", "e", "time"}
    assert set(nf.p2.nest.fields) == {"e", "f"}
    assert (nf["p2.e"] == nf["packed.d"] * 2 + nf.c).all()
    assert (nf["p2.f"] == nf["p2.e"] + nf.b).all()

    # Verify that assignment can be done to nested columns and fields
    # having names which are not valid Python identifiers, and must
    # be quoted with backticks.
    nf = NestedFrame(data={"dog": [1, 2, 3], "good dog": [2, 4, 6]}, index=[0, 1, 2])
    nested = pd.DataFrame(
        data={"n/a": [0, 2, 4, 1, 4, 3, 1, 4, 1], "n/b": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    nf = nf.add_nested(nested, "bad dog")
    nfx = nf.eval("`bad dog`.`n/c` = `bad dog`.`n/b` + 2.5")
    # The number of columns at the top should not have changed
    assert len(nfx.columns) == len(nf.columns)
    assert (nfx["bad dog"].nest["n/c"] == nf["bad dog"].nest["n/b"] + 2.5).all()


def test_access_non_existing_column():
    """Test that accessing a non-existing column raises a KeyError"""
    nf = NestedFrame()
    with pytest.raises(KeyError):
        _ = nf["non_existing_column"]


def test_issue193():
    """https://github.com/lincc-frameworks/nested-pandas/issues/193"""
    ndf = generate_data(3, 3)
    ndf.query("nested.flux / nested.t > 0")
    # This failed with numpy 1 with:
    # TypeError: Cannot interpret 'double[pyarrow]' as a data type


def test_issue235():
    """https://github.com/lincc-frameworks/nested-pandas/issues/235"""
    nf = generate_data(3, 10).iloc[:0]
    nf["nested.x"] = []


def test_nest_lists():
    """
    Test that we can take columns with list values and pack them into nested columns.
    """
    # Test that an empty dataframe we still produce a nested column, even if it has no values.
    empty_ndf = NestedFrame({"a": [], "b": [], "c": []})
    empty_ndf = empty_ndf.nest_lists(columns=["b", "c"], name="nested")
    assert len(empty_ndf) == 0
    assert empty_ndf.nested.nest.to_flat().shape == (0, 2)
    assert empty_ndf.nested.nest.fields == ["b", "c"]
    assert set(empty_ndf.columns) == set(["a", "nested"])

    # Test packing empty lists as columns.
    empty_list_ndf = NestedFrame({"a": [1], "b": [[]], "c": [[]]})
    empty_list_ndf = empty_list_ndf.nest_lists(columns=["b", "c"], name="nested")
    assert empty_list_ndf.nested.nest.to_flat().shape == (0, 2)
    assert empty_list_ndf.nested.nest.fields == ["b", "c"]
    assert set(empty_list_ndf.columns) == {"a", "nested"}

    # Test that we raise an error if the columns are not lists
    with pytest.raises(ValueError):
        empty_list_ndf.nest_lists(columns=["a", "c"], name="nested")

    expected = NestedFrame(
        {
            "a": [1, 2, 3],
            "b": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "c": [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            "d": ["dog", "cat", "mouse"],
        }
    )
    ndf = expected.copy()

    # Generate our expected dataframe to compare to by packing
    # the lists and dropping the original columns.
    packed_expected = pack_lists(ndf[["b", "c"]], name="nested")
    expected = expected.join(packed_expected, how="left")
    expected.drop(columns=["b", "c"], inplace=True)

    # Test that we can pack the lists into a nested column
    res = ndf.nest_lists(columns=["b", "c"], name="nested")
    assert res.equals(expected)

    # Test that we will not pack in a string column as a list
    # and that we raise an error if we try to do so.
    with pytest.raises(ValueError):
        ndf.nest_lists(columns=["c", "d"], name="nested")

    # Test nest_lists ordering deprecation warning
    with pytest.warns(DeprecationWarning):
        res = ndf.nest_lists("nested", ["c", "b"])


def test_delitem_base_and_nested():
    """Test that __delitem__ works for both base and nested columns."""
    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])
    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    base = base.add_nested(nested, "nested")

    # Delete a nested field
    del base["nested.c"]
    assert "c" not in base["nested"].nest.fields
    # Delete a base column
    del base["a"]
    assert "a" not in base.columns
    # Deleting a missing column should raise KeyError
    with pytest.raises(KeyError):
        del base["not_a_column"]
    with pytest.raises(KeyError):
        del base["nested.not_a_field"]


def test_auto_nest_on_dataframe_assignment():
    """Test that assigning a DataFrame via __setitem__ to a new column auto-nests it."""
    nested_data = {"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]}
    nested_index = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    # Create pandas and nested frames which we will nest to our base layer
    pd_nested = pd.DataFrame(data=nested_data, index=nested_index)
    nf_nested = NestedFrame(data=nested_data, index=nested_index)
    for nested in [pd_nested, nf_nested]:
        # Create a base NestedFrame and assign the nested DataFrame using __setitem__
        base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])
        base["nested"] = nested

        # Validate we added the nested column
        assert "nested" in base.nested_columns

        # The flat representation should match the original DataFrame (ignoring dtype)
        flat = base["nested"].nest.to_flat()
        assert (flat.values == nested.values).all()
        assert list(flat.columns) == list(nested.columns)
        assert list(flat.index) == list(nested.index)


def test_issue294():
    """https://github.com/lincc-frameworks/nested-pandas/issues/294"""
    nf1 = generate_data(3, 5)
    nf2 = generate_data(4, 6)
    nf = pd.concat([nf1, nf2])
    nf["c"] = range(7)
    # Check if we did concatenation right
    assert nf.shape[0] == 7
    # We need multiple chunk_lens in the nested columns for the test setup
    assert nf.nested.array.list_array.num_chunks == 2
    # And no chunk_lens in the base column
    c_pa_array = pa.array(nf["c"])
    assert isinstance(c_pa_array, pa.Array) or (
        isinstance(c_pa_array, pa.ChunkedArray) and c_pa_array.num_chunks == 1
    )

    # Failed with a ValueError in the original issue
    nf["nested.c"] = nf["c"]
    nf["nested.mag"] = -2.5 * np.log10(nf["nested.flux"])
