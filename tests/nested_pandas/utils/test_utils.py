import numpy as np
import pandas as pd
import pytest
from nested_pandas import NestedFrame
from nested_pandas.utils import count_nested
from numpy.testing import assert_array_equal
from pandas.testing import assert_index_equal


@pytest.mark.parametrize("join", [True, False])
def test_count_nested(join):
    """Test the functionality of count nested"""

    # Initialize test data
    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, np.nan, 6]}, index=[100, 101, 102])
    nested = pd.DataFrame(
        data={
            "c": [0, 2, 4, 1, np.nan, 3, 1, 4, 1],
            "d": [5, 4, 7, 5, 3, 1, 9, 3, 4],
            "label": ["b", "a", "b", "b", "a", "a", "b", "a", "b"],
        },
        index=[100, 100, 100, 101, 101, 101, 102, 102, 102],
    )
    base = base.add_nested(nested, "nested")

    # Test general count
    total_counts = count_nested(base, "nested", join=join)
    assert_array_equal(total_counts["n_nested"].values, 3)
    assert_index_equal(total_counts.index, base.index)

    # Test count by
    label_counts = count_nested(base, "nested", by="label", join=join)

    assert_array_equal(label_counts["n_nested_a"].values, [1, 2, 1])
    assert_array_equal(label_counts["n_nested_b"].values, [2, 1, 2])
    assert_index_equal(label_counts.index, base.index)

    # Make sure the ordering is alphabetical
    # https://github.com/lincc-frameworks/nested-pandas/issues/109
    assert label_counts.columns[-1] == "n_nested_b"
    assert label_counts.columns[-2] == "n_nested_a"

    # Test join behavior
    if join:
        assert total_counts.columns.tolist() == base.columns.tolist() + ["n_nested"]
        assert label_counts.columns.tolist() == base.columns.tolist() + ["n_nested_a", "n_nested_b"]
    else:
        assert total_counts.columns.tolist() == ["n_nested"]
        assert label_counts.columns.tolist() == ["n_nested_a", "n_nested_b"]


def test_check_expr_nesting():
    """
    Test the correctness of the evaluation expression pre-flight checks, which are
    used to ensure that an expression-based query does not try to combine base and nested
    sub-expressions.
    """
    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, np.nan, 6]}, index=[0, 1, 2])
    nested = pd.DataFrame(
        data={
            "c": [0, 2, 4, 1, np.nan, 3, 1, 4, 1],
            "d": [5, 4, 7, 5, 3, 1, 9, 3, 4],
            "label": ["b", "a", "b", "b", "a", "a", "b", "a", "b"],
        },
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    b1 = base.add_nested(nested, "nested")
    assert b1.extract_nest_names("a > 2 & nested.c > 1") == {"", "nested"}
    assert b1.extract_nest_names("(nested.c > 1) and (nested.d>2)") == {"nested"}
    assert b1.extract_nest_names("-1.52e-5 < b < 35.2e2") == {""}

    b2 = base.add_nested(nested.copy(), "n")
    assert b2.extract_nest_names("(n.c > 1) and ((b + a) > (b - 1e-8)) or n.d > a") == {"n", ""}

    abc = pd.DataFrame(
        data={
            "c": [3, 1, 4, 1, 5, 9, 2, 6, 5],
            "d": [1, 4, 1, 2, 1, 3, 5, 6, 2],
            "g": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        },
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    b3 = base.add_nested(abc, "abc").add_nested(abc, "c")
    assert b3.extract_nest_names("abc.c > 2 & c.d < 5") == {"abc", "c"}

    assert b3.extract_nest_names("(abc.d > 3) & (abc.c == [2, 5])") == {"abc"}
    assert b3.extract_nest_names("(abc.d > 3)&(abc.g == 'f')") == {"abc"}
    assert b3.extract_nest_names("(abc.d > 3) & (abc.g == 'f')") == {"abc"}

    assert b1.extract_nest_names("a>3") == {""}
    assert b1.extract_nest_names("a > 3") == {""}

    b4 = base.add_nested(nested, "test")
    assert b4.extract_nest_names("test.c>5&b==2") == {"test", ""}
    assert b4.extract_nest_names("test.c > 5 & b == 2") == {"test", ""}
