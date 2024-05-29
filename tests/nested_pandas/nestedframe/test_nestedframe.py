import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas import NestedFrame
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
        index=[0, 0, 0, 1, 1, 1, 1, 1, 1],  # no data for index value of "2"
    )

    base = base.add_nested(nested, "nested")

    assert "nested" in base.columns
    assert pd.isna(base.loc[2]["nested"])


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

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, np.NaN, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, np.NaN, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
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


def test_dropna_inplace_base():
    """Test in-place behavior of dropna"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [np.NaN, 4, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, np.NaN, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
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

    base = NestedFrame(data={"a": [1, 2, 3], "b": [np.NaN, 4, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, np.NaN, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
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

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, np.NaN, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, np.NaN, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
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
