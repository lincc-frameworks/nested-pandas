import os
import tempfile

import pandas as pd
import pytest
from nested_pandas import NestedFrame, read_parquet
from pandas.testing import assert_frame_equal


@pytest.mark.parametrize("columns", [["a"], None])
def test_read_parquet(tmp_path, columns):
    """Test nested parquet loading"""
    # Setup a temporary directory for files
    save_path = os.path.join(tmp_path, ".")

    # Generate some test data
    base = pd.DataFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    nested1 = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    nested2 = pd.DataFrame(
        data={"e": [0, 2, 4, 1, 4, 3, 1, 4, 1], "f": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    # Save to parquet
    base.to_parquet(os.path.join(save_path, "base.parquet"))
    nested1.to_parquet(os.path.join(save_path, "nested1.parquet"))
    nested2.to_parquet(os.path.join(save_path, "nested2.parquet"))

    # Read from parquet
    nf = read_parquet(
        data=os.path.join(save_path, "base.parquet"),
        columns=columns,
    )

    nest1 = read_parquet(os.path.join(save_path, "nested1.parquet"))
    nest2 = read_parquet(os.path.join(save_path, "nested1.parquet"))

    nf = nf.add_nested(nest1, name="nested1").add_nested(nest2, name="nested2")

    # Check Base Columns
    if columns is not None:
        assert nf.columns.tolist() == columns + ["nested1", "nested2"]
    else:
        assert nf.columns.tolist() == base.columns.tolist() + ["nested1", "nested2"]


def test_write_parquet():
    """Tests writing a nested frame to a single parquet file."""
    # Generate some test data
    base = pd.DataFrame(data={"a": [1, 2, 3], "b": [2, 4, 6]}, index=[0, 1, 2])

    nested1 = pd.DataFrame(
        data={"c": [0, 2, 4, 1, 4, 3, 1, 4, 1], "d": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    nested2 = pd.DataFrame(
        data={"e": [0, 2, 4, 1, 4, 3, 1, 4, 1], "f": [5, 4, 7, 5, 3, 1, 9, 3, 4]},
        index=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )

    # Construct the NestedFrame
    nf = NestedFrame(base).add_nested(nested1, name="nested1").add_nested(nested2, name="nested2")

    # Write to parquet using a named temporary file
    temp = tempfile.NamedTemporaryFile(suffix=".parquet")
    nf.to_parquet(temp.name)

    # Read from parquet
    nf2 = read_parquet(temp.name)
    assert_frame_equal(nf, nf2)
