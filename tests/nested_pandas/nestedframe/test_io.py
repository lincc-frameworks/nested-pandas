import os
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nested_pandas import NestedFrame, read_parquet
from nested_pandas.datasets import generate_data
from nested_pandas.nestedframe.io import from_pyarrow
from pandas.testing import assert_frame_equal
from upath import UPath


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


def test_read_parquet_list():
    """Test reading a parquet file with no columns specified"""
    # Load in the example files
    single_file_nf = read_parquet("tests/test_data/nested.parquet")
    nf = read_parquet(["tests/test_data/nested.parquet", "tests/test_data/nested.parquet"])

    # Check the columns
    assert nf.columns.tolist() == ["a", "flux", "nested", "lincc"]

    # Make sure nested columns were recognized
    assert nf.nested_columns == ["nested", "lincc"]

    # Check the nested columns
    assert nf.nested.nest.fields == ["t", "flux", "band"]
    assert nf.lincc.nest.fields == ["band", "frameworks"]

    # Check loading list works correctly
    assert len(nf) == 2 * len(single_file_nf)


def test_read_parquet_directory():
    """Test reading a parquet file with no columns specified"""
    # Load in the example file
    nf = read_parquet("tests/test_data")

    # Check the columns
    assert nf.columns.tolist() == ["a", "flux", "nested", "lincc"]

    # Make sure nested columns were recognized
    assert nf.nested_columns == ["nested", "lincc"]

    # Check the nested columns
    assert nf.nested.nest.fields == ["t", "flux", "band"]
    assert nf.lincc.nest.fields == ["band", "frameworks"]


def test_read_parquet_directory_with_filesystem():
    """Test reading a parquet file with no columns specified"""
    # Load in the example file
    path = UPath("tests/test_data")
    nf = read_parquet(path.path, filesystem=path.fs)

    # Check the columns
    assert nf.columns.tolist() == ["a", "flux", "nested", "lincc"]

    # Make sure nested columns were recognized
    assert nf.nested_columns == ["nested", "lincc"]

    # Check the nested columns
    assert nf.nested.nest.fields == ["t", "flux", "band"]
    assert nf.lincc.nest.fields == ["band", "frameworks"]


def test_file_object_read_parquet():
    """Test reading parquet from a file-object"""
    with open("tests/test_data/nested.parquet", "rb") as f:
        nf = read_parquet(f)
    # Check the columns
    assert nf.columns.tolist() == ["a", "flux", "nested", "lincc"]
    # Make sure nested columns were recognized
    assert nf.nested_columns == ["nested", "lincc"]
    # Check the nested columns
    assert nf.nested.nest.fields == ["t", "flux", "band"]
    assert nf.lincc.nest.fields == ["band", "frameworks"]


@pytest.mark.parametrize(
    "columns",
    [
        ["a", "flux"],
        ["flux", "nested", "lincc"],
        ["nested.flux", "nested.band"],
        ["flux", "nested.flux"],
        ["nested.band", "lincc.band"],
    ],
)
def test_read_parquet_column_selection(columns):
    """Test reading a parquet file with column selection"""
    # Load in the example file
    nf = read_parquet("tests/test_data/nested.parquet", columns=columns)

    # Output expectations
    if columns == ["a", "flux"]:
        expected_columns = ["a", "flux"]
    elif columns == ["flux", "nested", "lincc"]:
        expected_columns = ["flux", "nested", "lincc"]
    elif columns == ["nested.flux", "nested.band"]:
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


@pytest.mark.parametrize("reject", [["nested"], "nested"])
def test_read_parquet_reject_nesting(reject):
    """Test reading a parquet file with column selection"""
    # Load in the example file
    nf = read_parquet("tests/test_data/nested.parquet", columns=["a", "nested"], reject_nesting=reject)

    # Check the columns
    assert nf.columns.tolist() == ["a", "nested"]

    # Make sure "nested" was not recognized as a nested column
    assert nf.nested_columns == []

    assert pa.types.is_struct(nf["nested"].dtype.pyarrow_dtype)


def test_read_parquet_reject_nesting_partial_loading():
    """Test reading a parquet file with column selection"""
    # Load in the example file
    nf = read_parquet("tests/test_data/nested.parquet", columns=["a", "nested.t"], reject_nesting=["nested"])

    # Check the columns
    assert nf.columns.tolist() == ["a", "t"]


def test_read_parquet_catch_full_and_partial():
    """Test reading a parquet file with column selection"""
    # Load in the example file
    with pytest.raises(ValueError):
        read_parquet("tests/test_data/nested.parquet", columns=["a", "nested.t", "nested"])


def test_read_parquet_catch_failed_cast():
    """Test reading a parquet file with column selection"""
    # Load in the example file
    with pytest.raises(ValueError):
        read_parquet("tests/test_data/not_nestable.parquet")


def test_read_parquet_test_mixed_struct():
    """Test reading a parquet file with mixed struct types"""
    # Create the pure-list StructArray
    field1 = pa.array([[1, 2], [3, 4], [5, 6]])
    field2 = pa.array([["a", "b"], ["b", "c"], ["c", "d"]])
    field3 = pa.array([[True, False], [True, False], [True, False]])
    struct_array_list = pa.StructArray.from_arrays([field1, field2, field3], ["list1", "list2", "list3"])

    # Create the value StructArray
    field1 = pa.array([1, 2, 3])
    field2 = pa.array(["a", "b", "c"])
    field3 = pa.array([True, False, True])
    struct_array_val = pa.StructArray.from_arrays([field1, field2, field3], ["val1", "va12", "val3"])

    # Create the mixed-list StructArray
    field1 = pa.array([1, 2, 3])
    field2 = pa.array(["a", "b", "c"])
    field3 = pa.array([[True, False], [True, False], [True, False]])
    struct_array_mix = pa.StructArray.from_arrays([field1, field2, field3], ["val1", "va12", "list3"])

    # Create a PyArrow Table with the StructArray as one of the columns
    table = pa.table(
        {
            "id": pa.array([100, 101, 102]),  # Another column
            "struct_list": struct_array_list,  # Struct column
            "struct_value": struct_array_val,
            "struct_mix": struct_array_mix,
        }
    )

    # Write to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        pq.write_table(table, os.path.join(tmpdir, "structs.parquet"))

        # Test full read
        nf = read_parquet(os.path.join(tmpdir, "structs.parquet"))
        assert nf.columns.tolist() == ["id", "struct_list", "struct_value", "struct_mix"]
        assert nf.nested_columns == ["struct_list"]

        # Test partial read
        nf = read_parquet(os.path.join(tmpdir, "structs.parquet"), columns=["id", "struct_mix.list3"])
        assert nf.columns.tolist() == ["id", "struct_mix"]
        assert nf.nested_columns == ["struct_mix"]

        # Test partial read with ordering to force reject pops
        nf = read_parquet(
            os.path.join(tmpdir, "structs.parquet"), columns=["id", "struct_mix.list3", "struct_mix.val1"]
        )
        assert nf.columns.tolist() == ["id", "list3", "val1"]
        assert len(nf.nested_columns) == 0


def test_from_pyarrow_test_mixed_struct():
    """Test reading a pyarrow table with mixed struct types"""
    # Create the pure-list StructArray
    field1 = pa.array([[1, 2], [3, 4], [5, 6]])
    field2 = pa.array([["a", "b"], ["b", "c"], ["c", "d"]])
    field3 = pa.array([[True, False], [True, False], [True, False]])
    struct_array_list = pa.StructArray.from_arrays([field1, field2, field3], ["list1", "list2", "list3"])

    # Create the value StructArray
    field1 = pa.array([1, 2, 3])
    field2 = pa.array(["a", "b", "c"])
    field3 = pa.array([True, False, True])
    struct_array_val = pa.StructArray.from_arrays([field1, field2, field3], ["val1", "va12", "val3"])

    # Create the mixed-list StructArray
    field1 = pa.array([1, 2, 3])
    field2 = pa.array(["a", "b", "c"])
    field3 = pa.array([[True, False], [True, False], [True, False]])
    struct_array_mix = pa.StructArray.from_arrays([field1, field2, field3], ["val1", "va12", "list3"])

    # Create a PyArrow Table with the StructArray as one of the columns
    table = pa.table(
        {
            "id": pa.array([100, 101, 102]),  # Another column
            "struct_list": struct_array_list,  # Struct column
            "struct_value": struct_array_val,
            "struct_mix": struct_array_mix,
        }
    )

    # Test full read
    nf = from_pyarrow(table)
    assert nf.columns.tolist() == ["id", "struct_list", "struct_value", "struct_mix"]
    assert nf.nested_columns == ["struct_list"]


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

    nf = generate_data(10, 100, seed=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        nf.to_parquet(os.path.join(tmpdir, "nested_for_pd.parquet"))
        # Load in the example file
        df = pd.read_parquet(os.path.join(tmpdir, "nested_for_pd.parquet"))

        # Check the columns
        assert df.columns.tolist() == ["a", "b", "nested"]


def test_read_empty_parquet():
    """Test that we can read empty parquet files"""
    orig_nf = generate_data(1, 2).iloc[:0]

    with tempfile.NamedTemporaryFile("wb", suffix="parquet") as tmpfile:
        orig_nf.to_parquet(tmpfile.name)
        # All columns
        # Do not check dtype because of:
        # https://github.com/lincc-frameworks/nested-pandas/issues/252
        assert_frame_equal(read_parquet(tmpfile.name), orig_nf, check_dtype=False)
        # Few columns
        assert_frame_equal(
            read_parquet(
                tmpfile.name,
                columns=[
                    "a",
                    "nested.flux",
                    "nested.band",
                ],
            ),
            orig_nf.drop(["b", "nested.t"], axis=1),
            check_dtype=False,
        )


def test_read_parquet_list_autocast():
    """Test reading a parquet file with list autocasting"""
    list_nf = NestedFrame(
        {
            "a": ["cat", "dog", "bird"],
            "b": [1, 2, 3],
            "c": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "d": [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
        }
    )
    with tempfile.NamedTemporaryFile("wb", suffix="parquet") as tmpfile:
        list_nf.to_parquet(tmpfile.name)

        nf = read_parquet(tmpfile.name, autocast_list=True)

        assert nf.nested_columns == ["c", "d"]
        assert nf["c"].nest.fields == ["c"]
        assert len(nf["c"].nest.to_flat()) == 9
        assert nf["d"].nest.fields == ["d"]
        assert len(nf["d"].nest.to_flat()) == 9
