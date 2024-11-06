import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas import NestedFrame
from nested_pandas.datasets import generate_data
from nested_pandas.nestedframe.core import _SeriesFromNest
from pandas.testing import assert_frame_equal


def test_nestedframe_construction():
    """Test NestedFrame construction"""
    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    assert isinstance(base, NestedFrame)


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


def test_set_or_replace_nested_col():
    """Test that __setitem__ can set or replace a column in a existing nested structure"""

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


def test_get_dot_names():
    """Test the ability to still work with column names with '.' characters outside of nesting"""
    nf = NestedFrame.from_flat(
        NestedFrame({"a": [1, 2, 3, 4], ".b.": [1, 1, 3, 3], "R.A.": [3, None, 6, 5]}, index=[1, 1, 2, 2]),
        base_columns=[".b."],
    )

    assert len(nf[".b."]) == 2
    assert len(nf["nested.R.A."]) == 4


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

    result = nf.reduce(offset_avg, "b", "packed.c", ["offset_avg"])
    assert len(result) == len(nf)
    assert isinstance(result, NestedFrame)
    assert result.index.name == "idx"
    for i in range(len(result)):
        assert result["offset_avg"].values[i] == expected_offset_avg[i]


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


def test_scientific_notation():
    """
    Test that NestedFrame.query handles constants that are written in scientific notation.
    """
    # https://github.com/lincc-frameworks/nested-pandas/issues/59
    base = NestedFrame({"a": [1, 1e-2, 3]}, index=[0, 1, 2])
    selected = base.query("a > 1e-1")
    assert list(selected.index) == [0, 2]


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


def test_access_non_existing_column():
    """Test that accessing a non-existing column raises a KeyError"""
    nf = NestedFrame()
    with pytest.raises(KeyError):
        _ = nf["non_existing_column"]
