import numpy as np
import pandas as pd
import pytest
from nested_pandas import NestedFrame
from nested_pandas.utils import count_nested


@pytest.mark.parametrize("join", [True, False])
def test_count_nested(join):
    """Test the functionality of count nested"""

    # Initialize test data
    base = NestedFrame(data={"a": [1, 2, 3], "b": [2, np.nan, 6]}, index=[0, 1, 2])
    nested = pd.DataFrame(
        data={
            "c": [0, 2, 4, 1, np.nan, 3, 1, 4, 1],
            "d": [5, 4, 7, 5, 3, 1, 9, 3, 4],
            "label": ["b", "a", "b", "b", "a", "a", "b", "a", "b"],
        },
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    base = base.add_nested(nested, "nested")

    # Test general count
    total_counts = count_nested(base, "nested", join=join)
    assert all(total_counts["n_nested"].values == 3)

    # Test count by
    label_counts = count_nested(base, "nested", by="label", join=join)

    assert all(label_counts["n_nested_a"].values == [1, 2, 1])
    assert all(label_counts["n_nested_b"].values == [2, 1, 2])

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
