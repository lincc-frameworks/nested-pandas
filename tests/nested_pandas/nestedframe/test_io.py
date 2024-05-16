import os
import tempfile

import pandas as pd
import pytest
from nested_pandas import NestedFrame, read_parquet
from pandas.testing import assert_frame_equal


@pytest.mark.parametrize("columns", [["a"], None])
@pytest.mark.parametrize("pack_columns", [{"nested1": ["c"], "nested2": ["e"]}, {"nested1": ["d"]}, None])
def test_read_parquet(tmp_path, columns, pack_columns):
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
        to_pack={
            "nested1": os.path.join(save_path, "nested1.parquet"),
            "nested2": os.path.join(save_path, "nested2.parquet"),
        },
        columns=columns,
        pack_columns=pack_columns,
    )

    # Check Base Columns
    if columns is not None:
        assert nf.columns.tolist() == columns + ["nested1", "nested2"]
    else:
        assert nf.columns.tolist() == base.columns.tolist() + ["nested1", "nested2"]

    # Check Nested Columns
    if pack_columns is not None:
        for nested_col in pack_columns:
            assert nf[nested_col].nest.fields == pack_columns[nested_col]
    else:
        for nested_col in nf.nested_columns:
            if nested_col == "nested1":
                assert nf[nested_col].nest.fields == nested1.columns.tolist()
            elif nested_col == "nested2":
                assert nf[nested_col].nest.fields == nested2.columns.tolist()


def test_write_packed_parquet():
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


def test_write_parquet_by_layer():
    """Tests writing a nested frame to multiple parquet files."""
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

    # Asser that a temporary file path must be a directory when by_layer is True
    with pytest.raises(ValueError):
        nf.to_parquet(tempfile.NamedTemporaryFile(suffix=".parquet").name, by_layer=True)

    # Write to parquet using a named temporary file
    tmp_dir = tempfile.TemporaryDirectory()
    nf.to_parquet(tmp_dir.name, by_layer=True)

    # Validate the individual layers were correctly saved as their own parquet files
    read_base_frame = read_parquet(os.path.join(tmp_dir.name, "base.parquet"), to_pack=None)
    assert_frame_equal(read_base_frame, nf.drop(columns=["nested1", "nested2"]))

    read_nested1 = read_parquet(os.path.join(tmp_dir.name, "nested1.parquet"), to_pack=None)
    assert_frame_equal(read_nested1, nf["nested1"].nest.to_flat())

    read_nested2 = read_parquet(os.path.join(tmp_dir.name, "nested2.parquet"), to_pack=None)
    assert_frame_equal(read_nested2, nf["nested2"].nest.to_flat())

    # Validate the entire NestedFrame can be read
    entire_nf = read_parquet(
        data=os.path.join(tmp_dir.name, "base.parquet"),
        to_pack={
            "nested1": os.path.join(tmp_dir.name, "nested1.parquet"),
            "nested2": os.path.join(tmp_dir.name, "nested2.parquet"),
        },
    )

    assert_frame_equal(nf, entire_nf)
