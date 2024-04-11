import numpy as np
import pandas as pd
import pytest
from nested_pandas import NestedFrame


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


def test_add_nested():
    """Test that add_nested correctly adds a nested column to the base df"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    nested = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    base = base.add_nested(nested, "nested")

    assert "nested" in base.columns
    assert base.nested.nest.to_flat().equals(nested)


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


def test_dropna_inplace():
    """Test in-place behavior of dropna"""

    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, np.NaN, 6]}, index=[0, 1, 2])

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
