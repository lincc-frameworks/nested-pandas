import os
import tempfile

import pandas as pd
import pytest
from nested_pandas import NestedFrame, read_parquet
from pandas.testing import assert_frame_equal

import pyarrow as pa


def test_read_parquet():
    """Test reading a parquet file with no columns specified"""
    # Load in the example file
    nf = read_parquet("tests/test_data/nested.parquet")

    # Check the columns
    assert nf.columns.tolist() == ["a", "flux", "nested", "lincc"]

    # Make sure nested columns were recognized
    assert nf.nested_columns == ["nested", "lincc"]

    # Check the nested columns
    assert nf.nested.nest.fields == ["t", "flux", "band"]
    assert nf.lincc.nest.fields == ["band", "frameworks"]


@pytest.mark.parametrize("columns", [["a", "flux"],
                                     ["flux", "nested", "lincc"],
                                     ["nested.flux", "nested.t"],
                                     ["flux", "nested.flux"],
                                     ["nested.band", "lincc.band"],])
def test_read_parquet_column_selection(columns):
    """Test reading a parquet file with column selection"""
    # Load in the example file
    nf = read_parquet("tests/test_data/nested.parquet", columns=columns)

    # Output expectations
    if columns == ["a", "flux"]:
        expected_columns = ["a", "flux"]
    elif columns == ["flux", "nested", "lincc"]:
        expected_columns = ["flux", "nested", "lincc"]
    elif columns == ["nested.flux", "nested.t"]:
        expected_columns = ["nested"]
    elif columns == ["flux", "nested.flux"]:
        expected_columns = ["flux", "nested"]
    elif columns == ["nested.band", "lincc.band"]:
        expected_columns = ["nested", "lincc"]
    # Check the columns
    assert nf.columns.tolist() == expected_columns

    # Check nested columns
    if columns == ["nested.flux", "nested.t"]:
        assert nf.nested.nest.fields == ["flux", "t"]
    elif columns == ["nested.band", "lincc.band"]:
        assert nf.nested.nest.fields == ["band"]
        assert nf.lincc.nest.fields == ["band"]


def test_read_parquet_reject_nesting():
    """Test reading a parquet file with column selection"""
    # Load in the example file
    nf = read_parquet("tests/test_data/nested.parquet",
                      columns=["a", "nested"],
                      reject_nesting=["nested"])

    # Check the columns
    assert nf.columns.tolist() == ["a", "nested"]

    # Make sure "nested" was not recognized as a nested column
    assert nf.nested_columns == []

    assert pa.types.is_struct(nf["nested"].dtype.pyarrow_dtype)


def test_read_parquet_reject_nesting_partial_loading():
    """Test reading a parquet file with column selection"""
    # Load in the example file
    nf = read_parquet("tests/test_data/nested.parquet",
                      columns=["a", "nested.t"],
                      reject_nesting=["nested"])

    # Check the columns
    assert nf.columns.tolist() == ["a", "t"]


def test_read_parquet_catch_full_and_partial():
    """Test reading a parquet file with column selection"""
    # Load in the example file
    with pytest.raises(ValueError):
        read_parquet("tests/test_data/nested.parquet", columns=["a", "nested.t", "nested"])


def test_to_parquet():
    """Test writing a parquet file with no columns specified"""
    # Load in the example file
    nf = read_parquet("tests/test_data/nested.parquet")

    # Write to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        nf.to_parquet(os.path.join(tmpdir, "nested.parquet"))

        # Read the file back in
        nf2 = read_parquet(os.path.join(tmpdir, "nested.parquet"))

        # Check the columns
        assert nf.columns.tolist() == nf2.columns.tolist()

        # Check the nested columns
        assert nf.nested_columns == nf2.nested_columns

        # Check the data
        assert_frame_equal(nf, nf2)


def test_pandas_read_parquet():
    """Test that pandas can read our serialized files"""
    # Load in the example file
    df = pd.read_parquet("tests/test_data/nested.parquet")

    # Check the columns
    assert df.columns.tolist() == ["a", "flux", "nested", "lincc"]
