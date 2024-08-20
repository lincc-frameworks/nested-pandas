import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas import NestedFrame
from nested_pandas.datasets import generate_data
from pandas.testing import assert_frame_equal


def test_nestedframe_construction():
    """Test NestedFrame construction"""
    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    assert isinstance(base, NestedFrame)


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

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
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


@pytest.mark.parametrize("index", [None, "a", "c"])
def test_from_flat(index):
    """Test the NestedFrame.from_flat functionality"""
    nf = NestedFrame(
        {"a": [1, 1, 1, 2, 2], "b": [2, 2, 2, 4, 4], "c": [1, 2, 3, 4, 5], "d": [2, 4, 6, 8, 10]},
        index=[0, 0, 0, 1, 1],
    )

    out_nf = NestedFrame.from_flat(nf, base_columns=["a", "b"], index=index, name="new_nested")

    if index is None:
        assert list(out_nf.columns) == ["a", "b", "new_nested"]
        assert list(out_nf.new_nested.nest.fields) == ["c", "d"]
        assert len(out_nf) == 2
    elif index == "a":
        assert list(out_nf.columns) == ["b", "new_nested"]
        assert list(out_nf.new_nested.nest.fields) == ["c", "d"]
        assert len(out_nf) == 2
    elif index == "c":  # not what a user likely wants, but should still work
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

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

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

    # Test nested queries
    nest_queried = base.query("nested.c > 1")
    assert len(nest_queried.nested.nest.to_flat()) == 5

    nest_queried = base.query("(nested.c > 1) and (nested.d>2)")
    assert len(nest_queried.nested.nest.to_flat()) == 4


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
