import io
import os
import tempfile
from pathlib import Path

import fsspec.implementations.http
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs
import pyarrow.parquet as pq
import pytest
from pandas.testing import assert_frame_equal
from upath import UPath

from nested_pandas import NestedFrame, read_parquet
from nested_pandas.datasets import generate_data
from nested_pandas.nestedframe.io import (
    FSSPEC_BLOCK_SIZE,
    _check_datafusion_support,
    _columns_to_load,
    _datafusion_filters_to_expr,
    _datafusion_object_store,
    _datafusion_read_table,
    _get_storage_options,
    _pyarrow_read_table,
    _transform_read_parquet_data_arg,
    from_pyarrow,
)


def test_read_parquet():
    """Test reading a parquet file with no columns specified"""
    # Load in the example file
    nf = read_parquet("tests/test_data/nested.parquet")

    # Check the columns
    assert nf.columns.tolist() == ["a", "flux", "nested", "lincc"]

    # Make sure nested columns were recognized
    assert nf.nested_columns == ["nested", "lincc"]

    # Check the nested columns
    assert nf.nested.nest.columns == ["t", "flux", "band"]
    assert nf.lincc.nest.columns == ["band", "frameworks"]


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
    assert nf.nested.nest.columns == ["t", "flux", "band"]
    assert nf.lincc.nest.columns == ["band", "frameworks"]

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
    assert nf.nested.nest.columns == ["t", "flux", "band"]
    assert nf.lincc.nest.columns == ["band", "frameworks"]


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
    assert nf.nested.nest.columns == ["t", "flux", "band"]
    assert nf.lincc.nest.columns == ["band", "frameworks"]


def test_file_object_read_parquet():
    """Test reading parquet from a file-object"""
    with open("tests/test_data/nested.parquet", "rb") as f:
        nf = read_parquet(f)
    # Check the columns
    assert nf.columns.tolist() == ["a", "flux", "nested", "lincc"]
    # Make sure nested columns were recognized
    assert nf.nested_columns == ["nested", "lincc"]
    # Check the nested columns
    assert nf.nested.nest.columns == ["t", "flux", "band"]
    assert nf.lincc.nest.columns == ["band", "frameworks"]


@pytest.mark.parametrize(
    "columns, expected_columns",
    [
        (["a", "flux"], ["a", "flux"]),
        (["flux", "nested", "lincc"], ["flux", "nested", "lincc"]),
        (["nested.flux", "nested.band"], ["nested"]),
        (["flux", "nested.flux"], ["flux", "nested"]),
        (["nested.band", "lincc.band"], ["nested", "lincc"]),
    ],
)
def test_read_parquet_column_selection(columns, expected_columns):
    """Test reading a parquet file with column selection"""
    # Load in the example file
    nf = read_parquet("tests/test_data/nested.parquet", columns=columns)

    # Check the column expectations
    assert nf.columns.tolist() == expected_columns

    # Check nested columns
    if columns == ["nested.flux", "nested.t"]:
        assert nf.nested.nest.columns == ["flux", "t"]
    elif columns == ["nested.band", "lincc.band"]:
        assert nf.nested.nest.columns == ["band"]
        assert nf.lincc.nest.columns == ["band"]


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


@pytest.mark.parametrize("list_struct", [False, True])
def test_to_pyarrow_list_struct_roundtrip(list_struct):
    """to_pyarrow / to_parquet support both nested layouts and round-trip."""
    # Source is already ArrowDtype-backed so flat-column dtypes round-trip too.
    nf = read_parquet("tests/test_data/nested.parquet")
    nested_col = nf.nested_columns[0]

    # The requested layout is reflected in the table schema.
    table = nf.to_pyarrow(list_struct=list_struct)
    table.validate(full=True)  # the produced table must be structurally sound
    nested_type = table.schema.field(nested_col).type
    if list_struct:
        assert pa.types.is_list(nested_type) or pa.types.is_large_list(nested_type)
    else:
        assert pa.types.is_struct(nested_type)

    # In-memory and on-disk round-trips both reconstruct the NestedFrame.
    assert_frame_equal(nf, from_pyarrow(table))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "layout.parquet")
        nf.to_parquet(path, list_struct=list_struct)
        pq.read_table(path).validate(full=True)  # the written file must be sound
        assert_frame_equal(nf, read_parquet(path))


def test_to_parquet_roundtrip_null_nested_field():
    """Round-trip a nested column with an all-null (``null``-typed) field.

    Regression test for https://github.com/lincc-frameworks/nested-pandas/issues/507:
    ``list<null>`` fields tripped an upstream pyarrow bug both on write (via the
    schema-rebuilding ``table.cast``) and on read (via ``Table.to_pandas``), which
    corrupted the list offsets of the all-null field.
    """
    # Struct-of-lists with an all-null field ``n`` and uneven list lengths [1, 1, 2].
    offsets = pa.array([0, 1, 2, 4], type=pa.int32())
    struct = pa.StructArray.from_arrays(
        [
            pa.ListArray.from_arrays(offsets, pa.nulls(4, pa.null())),
            pa.ListArray.from_arrays(offsets, pa.array([10, 20, 30, 40], pa.int64())),
        ],
        names=["n", "v"],
    )
    table = pa.table({"id": pa.array([1, 2, 3]), "nested": struct})
    nf = from_pyarrow(table)

    def assert_roundtrips(other):
        assert nf.nested_columns == other.nested_columns == ["nested"]
        assert nf["nested"].dtype == other["nested"].dtype
        assert_frame_equal(nf.drop(columns="nested"), other.drop(columns="nested"))
        # Compare the flattened nested data, including the all-null ``n`` field.
        assert_frame_equal(
            nf["nested"].nest.to_flat().reset_index(drop=True),
            other["nested"].nest.to_flat().reset_index(drop=True),
        )

    # In-memory round-trip through the public pyarrow API (no schema metadata).
    # ``validate(full=True)`` guards against silent corruption of the all-null
    # field: the pyarrow bug produces an invalid array without raising, so a
    # round-trip comparison alone would not reliably catch it.
    round_tripped = nf.to_pyarrow()
    round_tripped.validate(full=True)
    assert round_tripped.schema.metadata is None
    assert_roundtrips(from_pyarrow(round_tripped))

    # Round-trip through a parquet file on disk.
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "null_field.parquet")
        nf.to_parquet(path)
        pq.read_table(path).validate(full=True)  # the written file must be sound
        assert_roundtrips(read_parquet(path))


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

    with tempfile.NamedTemporaryFile("wb", suffix=".parquet") as tmpfile:
        tmpfile.close()
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
    with tempfile.NamedTemporaryFile("wb", suffix=".parquet") as tmpfile:
        tmpfile.close()
        list_nf.to_parquet(tmpfile.name)

        nf = read_parquet(tmpfile.name, autocast_list=True)

        assert nf.nested_columns == ["c", "d"]
        assert nf["c"].nest.columns == ["c"]
        assert len(nf["c"].nest.to_flat()) == 9
        assert nf["d"].nest.columns == ["d"]
        assert len(nf["d"].nest.to_flat()) == 9


def test__transform_read_parquet_data_arg():
    """Testing _transform_read_parquet_data_arg"""
    with open("tests/test_data/nested.parquet", "rb") as f:
        bytes = f.read()
    io_bytes = io.BytesIO(bytes)
    assert _transform_read_parquet_data_arg(io_bytes) == (io_bytes, None)

    local_path = "tests/test_data/nested.parquet"
    with open(local_path, "rb") as f:
        assert _transform_read_parquet_data_arg(f) == (f, None)
    with open(Path(local_path), "rb") as f:
        assert _transform_read_parquet_data_arg(f) == (f, None)
    with Path(local_path).open("rb") as f:
        assert _transform_read_parquet_data_arg(f) == (f, None)
    with UPath(local_path).open("rb") as f:
        assert _transform_read_parquet_data_arg(f) == (f, None)

    assert _transform_read_parquet_data_arg(local_path) == (local_path, None)

    assert _transform_read_parquet_data_arg(Path(local_path)) == (Path(local_path), None)

    local_upath = UPath(local_path)
    assert _transform_read_parquet_data_arg(local_upath) == (local_path, None)

    s3_path = "s3://nasa-irsa-euclid-q1/contributed/q1/merged_objects/hats/euclid_q1_merged_objects-hats/dataset/Norder=3/Dir=0/Npix=334/part0.snappy.parquet"
    path, fs = _transform_read_parquet_data_arg(s3_path)
    assert f"s3://{path}" == s3_path
    assert isinstance(fs, pa.fs.S3FileSystem)

    https_path = "https://data.lsdb.io/hats/gaia_dr3/gaia/dataset/Norder=2/Dir=0/Npix=0.parquet"
    path, fs = _transform_read_parquet_data_arg(https_path)
    assert path == https_path
    assert isinstance(fs, fsspec.implementations.http.HTTPFileSystem)

    with pytest.raises(TypeError):
        _transform_read_parquet_data_arg(123)

    local_paths = list(Path("tests/test_data").glob("*.parquet"))
    assert _transform_read_parquet_data_arg(local_paths) == (local_paths, None)

    local_upaths = list(UPath("tests/test_data").glob("*.parquet"))
    paths, fs = _transform_read_parquet_data_arg(local_upaths)
    assert paths == [up.path for up in local_upaths]
    assert fs is None

    with pytest.raises(ValueError):
        _transform_read_parquet_data_arg(
            [
                "tests/test_data",
                "https://data.lsdb.io/hats/gaia_dr3/gaia/dataset/Norder=2/Dir=0/Npix=0.parquet",
            ]
        )


def test_read_parquet_with_fsspec_optimization():
    """Test that read_parquet automatically uses fsspec optimization for remote files."""
    # Test with local file (should not use fsspec optimization)
    local_path = "tests/test_data/nested.parquet"

    # Test basic reading - local files should work as before
    nf1 = read_parquet(local_path)

    # Test with additional kwargs
    nf2 = read_parquet(local_path, columns=["a", "nested.flux"], use_threads=True)

    assert len(nf2) <= len(nf1)  # filtered columns
    assert "a" in nf2.columns
    assert "nested" in nf2.columns


def test_docstring_includes_fsspec_notes():
    """Test that the docstring mentions the automatic fsspec optimization."""
    docstring = read_parquet.__doc__
    assert "fsspec" in docstring
    assert "remote" in docstring.lower()


def test__get_storage_options():
    """Test _get_storage_options function with various input types."""
    local_path = "tests/test_data/nested.parquet"

    # Test with UPath objects (local files)
    local_upath = UPath(local_path)
    storage_opts = _get_storage_options(local_upath)
    assert storage_opts is None  # Local UPath should have no storage options

    # Test with UPath objects (HTTP)
    http_url = "http://example.com/data.parquet"
    http_upath = UPath(http_url)
    storage_opts = _get_storage_options(http_upath)
    assert storage_opts is not None
    assert storage_opts.get("block_size") == FSSPEC_BLOCK_SIZE

    # Test with UPath objects (HTTPS)
    https_url = "https://example.com/data.parquet"
    https_upath = UPath(https_url)
    storage_opts = _get_storage_options(https_upath)
    assert storage_opts is not None
    assert storage_opts.get("block_size") == FSSPEC_BLOCK_SIZE

    # Test with UPath objects (S3)
    s3_url = "s3://bucket/path/data.parquet"
    s3_upath = UPath(s3_url)
    storage_opts = _get_storage_options(s3_upath)
    assert storage_opts is not None
    # S3 should NOT have the block_size override (only HTTP/HTTPS)
    assert storage_opts.get("block_size") != FSSPEC_BLOCK_SIZE


def test__is_local_path():
    """Test the _is_local_path function with various scenarios."""
    from nested_pandas.nestedframe.io import _is_local_path

    assert _is_local_path(UPath("tests/test_data")) is True
    assert _is_local_path(UPath("tests/test_data/nested.parquet")) is True
    assert _is_local_path(UPath("https://example.com/data.parquet")) is False


def test__is_remote_dir():
    """Test the _is_remote_dir function with various scenarios."""
    from nested_pandas.nestedframe.io import _is_remote_dir

    # Local path that is a directory
    local_dir = UPath("tests/test_data")
    assert _is_remote_dir("tests/test_data", local_dir, is_dir=True) is True
    assert _is_remote_dir("tests/test_data", local_dir, is_dir=False) is False
    assert _is_remote_dir("tests/test_data", local_dir, is_dir=None) is True

    # Local path that is a file
    local_file = UPath("tests/test_data/nested.parquet")
    assert _is_remote_dir("tests/test_data/nested.parquet", local_file, is_dir=True) is True
    assert _is_remote_dir("tests/test_data/nested.parquet", local_file, is_dir=False) is False
    assert _is_remote_dir("tests/test_data/nested.parquet", local_file, is_dir=None) is False

    # Remote file path
    remote_path = UPath("https://example.com/data.parquet")
    # In this case, the override is overruled by a protocol check
    assert _is_remote_dir("https://example.com/data.parquet", remote_path, is_dir=True) is False
    assert _is_remote_dir("https://example.com/data.parquet", remote_path, is_dir=False) is False
    assert _is_remote_dir("https://example.com/data.parquet", remote_path, is_dir=None) is False

    # Remote directory path
    remote_dir_path = UPath("https://example.com/data/")
    # Also overruled by protocol check not supporting https
    assert _is_remote_dir("https://example.com/data/", remote_dir_path, is_dir=True) is False
    assert _is_remote_dir("https://example.com/data/", remote_dir_path, is_dir=False) is False
    assert _is_remote_dir("https://example.com/data/", remote_dir_path, is_dir=None) is False


def test_list_struct_partial_loading_error():
    """Test that attempting to partially load a list-struct raises an error."""
    # Load in the example file
    with pytest.raises(ValueError):
        read_parquet("tests/list_struct_data/list_struct.parquet", columns=["lightcurve.hmjd"])


def test_normal_loading_error():
    """Test that making a normal naming mistake raises the normal pyarrow error."""
    # Load in the example file
    with pytest.raises(ValueError, match="No match for*"):
        read_parquet("tests/test_data/nested.parquet", columns=["not_a_column"])


def test_read_parquet_with_fixed_length_struct_list():
    """Test reading a parquet file with fixed-length struct-list columns"""
    nf = read_parquet("tests/fixed_size_list_data/mmu-desi.parquet")
    assert nf.shape == (2, 18)
    assert nf.nested_columns == ["spectrum"]


def test_read_parquet_with_fixed_length_list_struct():
    """Test reading a parquet file with fixed-length list-struct columns"""
    nf = read_parquet("tests/fixed_size_list_data/fixed-size-list-struct.parquet")
    assert nf.shape == (5, 3)
    assert nf.nested_columns == ["fixed_nested"]


@pytest.mark.parametrize("size", [5000, 500_000, 5_000_000])
def test_issue_428(size):
    """Partial loading fsspec issue: https://github.com/lincc-frameworks/nested-pandas/issues/428"""

    # Initialize a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "tmp.parquet")

        # Generate and write the data
        generate_data(size, 3).to_parquet(file_path)
        nf = read_parquet(file_path, columns=["nested.t"])
        assert nf.columns == ["nested"]
        assert nf.nested.nest.columns == ["t"]


def test_use_pandas_metadata():
    """Test use_pandas_metadata parameter in read_parquet.
    Regression test for https://github.com/lincc-frameworks/nested-pandas/issues/460
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "tmp.parquet")

        # Write a parquet file with a custom index stored in pandas metadata
        df = pd.DataFrame({"a": [1, 2, 3], "custom_idx": [10, 20, 30]})
        df = df.set_index("custom_idx")
        df.to_parquet(file_path)

        # Default (use_pandas_metadata=True): index IS restored from metadata
        nf = read_parquet(file_path)
        assert nf.index.name == "custom_idx"

        # Explicit False: index is NOT restored from pandas metadata
        nf_no_meta = read_parquet(file_path, use_pandas_metadata=False)
        assert nf_no_meta.index.name != "custom_idx"


def test_issue_492():
    """Loading with filters issue: https://github.com/lincc-frameworks/nested-pandas/issues/492"""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "tmp.parquet")

        size = 500_000
        generate_data(size, 1).assign(z=np.random.random(size)).to_parquet(file_path)

        nf1 = read_parquet(file_path, columns=["a"], filters=[("z", "<", 0.5)])
        assert nf1.shape[1] == 1

        nf2 = read_parquet(file_path, columns=["a"], filters=[[("z", "<", 0.75), ("z", ">", 0.25)]])
        assert nf2.shape[1] == 1


def test__columns_to_load():
    """Test _columns_to_load for all paths and raises."""
    cols = ["a", "b"]

    # columns=None always returns None regardless of filters
    assert _columns_to_load(None, None) is None
    assert _columns_to_load(None, [("z", "<", 0.5)]) is None
    assert _columns_to_load(None, pc.field("z") < 0.5) is None

    # filters=None returns columns unchanged
    assert _columns_to_load(cols, None) == cols
    assert _columns_to_load([], None) == []

    # PyArrow Expression filter → return None (can't inspect columns needed)
    with pytest.warns(UserWarning, match="list-of-tuples"):
        assert _columns_to_load(cols, pc.field("z") < 0.5) is None

    # filters is not a list/Expression/None → ValueError
    with pytest.raises(ValueError, match="filters must be"):
        _columns_to_load(cols, "z < 0.5")
    with pytest.raises(ValueError, match="filters must be"):
        _columns_to_load(cols, 42)

    # Empty list → IndexError caught → ValueError
    with pytest.raises(ValueError, match="filters format must be"):
        _columns_to_load(cols, [])

    # list[tuple] (flat) format
    result = _columns_to_load(["a"], [("z", "<", 0.5)])
    assert result == ["a", "z"]

    # list[tuple] with multiple filter tuples (flat)
    result = _columns_to_load(["a"], [("z", "<", 0.5), ("w", ">", 0.1)])
    assert result == ["a", "w", "z"]

    # list[list[tuple]] (nested/DNF) format
    result = _columns_to_load(["a"], [[("z", "<", 0.5)]])
    assert result == ["a", "z"]

    # list[list[tuple]] with multiple conjunctions
    result = _columns_to_load(["a"], [[("z", "<", 0.75), ("z", ">", 0.25)], [("w", "=", 1)]])
    assert result == ["a", "w", "z"]

    # filter columns already in columns → no duplicates, sorted
    result = _columns_to_load(["a", "z"], [("z", "<", 0.5)])
    assert result == ["a", "z"]

    # inner element is not a list → ValueError
    with pytest.raises(ValueError, match="filters format must be"):
        _columns_to_load(cols, [("z", "<", 0.5), "not_a_tuple"])

    # tuple in nested list that can't be unpacked into (col, op, val) → ValueError
    with pytest.raises(ValueError, match="filters format must be"):
        _columns_to_load(cols, [[("z", "<", 0.5, "extra")]])


# ------------------------------------------------------------------------------
# "datafusion" engine
# ------------------------------------------------------------------------------

# (columns, filters) pairs the "datafusion" and "pyarrow" engines must agree on
DATAFUSION_READ_CASES = [
    (None, None),
    (["a"], None),
    (["a", "flux"], None),
    (["nested.flux"], None),
    (["nested.flux", "nested.t"], None),
    # The same leaf name in two nested columns, pyarrow returns two "band" columns
    (["nested.band", "lincc.band"], None),
    # The order of the requested columns must be preserved
    (["nested.t", "a", "lincc.frameworks"], None),
    (None, [("a", ">", 0.5)]),
    (["a"], [("a", ">", 0.5)]),
    # The filter column is not among the loaded ones
    (["nested.flux"], [("flux", "<", 50.0)]),
    # A filter on a column loaded as a nested sub-column
    (["a", "nested.t"], [("a", "<", 0.5)]),
    # Conjunction
    (["a"], [("a", ">", 0.1), ("a", "<", 0.9)]),
    # Disjunction of conjunctions
    (["a"], [[("a", "<", 0.1)], [("a", ">", 0.5), ("flux", "<", 50.0)]]),
    # Every comparison operator
    (["a"], [("a", "=", 0.417022004702574)]),
    (["a"], [("a", "==", 0.417022004702574)]),
    (["a"], [("a", "!=", 0.417022004702574)]),
    (["a"], [("a", ">=", 0.5)]),
    (["a"], [("a", "<=", 0.5)]),
    # in / not in
    (["a", "flux"], [("a", "in", [0.417022004702574, 0.7203244934421581])]),
    (["a", "flux"], [("a", "not in", [0.417022004702574])]),
    # A filter matching nothing
    (["a"], [("a", ">", 1e9)]),
]


@pytest.mark.parametrize("columns,filters", DATAFUSION_READ_CASES)
def test_read_parquet_datafusion_matches_pyarrow(columns, filters):
    """The "datafusion" engine returns exactly what the "pyarrow" engine returns."""
    path = "tests/test_data/nested.parquet"

    pyarrow_nf = read_parquet(path, columns=columns, filters=filters, engine="pyarrow")
    datafusion_nf = read_parquet(path, columns=columns, filters=filters, engine="datafusion")

    assert_frame_equal(datafusion_nf, pyarrow_nf)


def test_read_parquet_datafusion_pandas_metadata():
    """The pandas metadata survives the datafusion read, so the index is restored."""
    # pandas writes the "pandas" schema metadata, nested-pandas' to_parquet does not
    df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 11, 12], name="object_id"))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "meta.parquet")
        df.to_parquet(path)

        assert b"pandas" in _datafusion_read_table(path).schema.metadata

        for use_pandas_metadata in [True, False]:
            assert_frame_equal(
                read_parquet(path, engine="datafusion", use_pandas_metadata=use_pandas_metadata),
                read_parquet(path, engine="pyarrow", use_pandas_metadata=use_pandas_metadata),
            )

        # The index is restored from the metadata, rather than left as a column
        assert read_parquet(path, engine="datafusion").index.name == "object_id"
        assert read_parquet(path, engine="datafusion", use_pandas_metadata=False).index.name is None


def test_read_parquet_datafusion_directory():
    """The "datafusion" engine reads a directory of parquet files."""
    nf = generate_data(10, 5, seed=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(2):
            nf.iloc[5 * i : 5 * (i + 1)].to_parquet(os.path.join(tmpdir, f"part{i}.parquet"))

        pyarrow_nf = read_parquet(tmpdir, columns=["a", "nested.t"], engine="pyarrow")
        datafusion_nf = read_parquet(tmpdir, columns=["a", "nested.t"], engine="datafusion")

    # datafusion makes no promise about the order it reads the files in
    assert_frame_equal(
        datafusion_nf.sort_values("a", ignore_index=True),
        pyarrow_nf.sort_values("a", ignore_index=True),
    )


def test_read_parquet_datafusion_row_order():
    """Rows come back in file order, as they do from pyarrow.

    In parallel datafusion splits the row groups of even a single file across
    partitions, and the output order is then not even reproducible run to run.
    """
    # The scan is only split when each partition would be worth it, so the file has
    # to be bigger than roughly `target_partitions` (the CPU count) megabytes. On a
    # machine with many more cores than this file has megabytes the split does not
    # happen and this test passes whatever the settings are; the directory test
    # below has no such threshold.
    n_rows = 2_000_000
    expected = np.arange(n_rows)
    table = pa.table({"i": pa.array(expected)})
    with tempfile.TemporaryDirectory() as tmpdir:
        # Several row groups, so there is something for datafusion to split
        path = os.path.join(tmpdir, "ordered.parquet")
        pq.write_table(table, path, row_group_size=50_000, compression="none")

        for _ in range(3):
            actual = _datafusion_read_table(path)
            np.testing.assert_array_equal(actual.column("i").to_numpy(), expected)

        # A filtered read keeps the order of the rows it selects, and those come
        # from row groups far enough apart to land in different partitions
        selected = [7, n_rows // 2, n_rows - 1]
        actual = _datafusion_read_table(path, filters=[("i", "in", selected)])
        assert actual.column("i").to_pylist() == selected


def test_read_parquet_datafusion_directory_row_order():
    """A directory is read in the same order pyarrow reads it, file by file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(4):
            pq.write_table(pa.table({"i": pa.array([10 * i, 10 * i + 1])}), f"{tmpdir}/part{i}.parquet")

        expected = _pyarrow_read_table(tmpdir).column("i").to_pylist()
        for _ in range(3):
            assert _datafusion_read_table(tmpdir).column("i").to_pylist() == expected


def test_read_parquet_datafusion_quoted_column_names():
    """Names datafusion would otherwise lower-case or split on the dot."""
    table = pa.table(
        {
            "CamelCase": pa.array([1, 2]),
            "nested": pa.array([{"Sub": [1.0]}, {"Sub": [2.0]}]),
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "names.parquet")
        pq.write_table(table, path)

        nf = read_parquet(path, columns=["CamelCase", "nested.Sub"], engine="datafusion")

    assert nf.columns.tolist() == ["CamelCase", "nested"]
    assert nf.nested.nest.columns == ["Sub"]


def test_read_parquet_datafusion_list_struct_error():
    """Partially loading a list-of-structs column gives the pyarrow engine's error."""
    path = "tests/list_struct_data/list_struct.parquet"
    with pytest.raises(ValueError, match="not a struct"):
        read_parquet(path, columns=["lightcurve.hmjd"], engine="datafusion")


def test_read_parquet_datafusion_missing_column_error():
    """A plain naming mistake still raises."""
    with pytest.raises(Exception, match="not_a_column"):
        read_parquet("tests/test_data/nested.parquet", columns=["not_a_column"], engine="datafusion")


class _PointExtensionType(pa.ExtensionType):
    """A custom extension type, to check datafusion doesn't drop the unknown ones."""

    def __init__(self):
        super().__init__(pa.list_(pa.float64(), 2), "nested_pandas.test.point")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls()


@pytest.fixture(name="point_extension_type")
def fixture_point_extension_type():
    """Register the custom extension type for the duration of a single test."""
    point_type = _PointExtensionType()
    pa.register_extension_type(point_type)
    yield point_type
    pa.unregister_extension_type(point_type.extension_name)


@pytest.mark.parametrize("pass_schema", [True, False])
@pytest.mark.parametrize("store_schema", [True, False])
def test_datafusion_read_table_extension_types(point_extension_type, store_schema, pass_schema):
    """Extension types come back from a datafusion read, not their storage types.

    datafusion has no notion of an extension type, it carries the storage type and
    the "ARROW:extension:*" field metadata, and pyarrow rebuilds the type from its
    own registry. The metadata comes either from the file's own arrow schema, or
    from the `schema` argument. With neither, only the storage type is left.
    """
    tensor_type = pa.fixed_shape_tensor(pa.float64(), [2, 2])
    table = pa.table(
        {
            "tensor": pa.ExtensionArray.from_storage(
                tensor_type,
                pa.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], pa.list_(pa.float64(), 4)),
            ),
            "point": pa.ExtensionArray.from_storage(
                point_extension_type,
                pa.array([[1.0, 2.0], [3.0, 4.0]], pa.list_(pa.float64(), 2)),
            ),
            "a": pa.array([1, 2]),
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "extension.parquet")
        writer = pq.ParquetWriter(path, table.schema, store_schema=store_schema)
        writer.write_table(table)
        writer.close()

        schema = table.schema if pass_schema else None
        actual = _datafusion_read_table(path, schema=schema)
        projected = _datafusion_read_table(path, columns=["tensor", "point"], schema=schema)

    if store_schema or pass_schema:
        assert actual.schema.types == [tensor_type, point_extension_type, pa.int64()]
        # The types survive the projection too
        assert projected.schema.types == [tensor_type, point_extension_type]

        # The tensor is readable as a tensor: to_pylist() would only give the flat
        # storage lists, the [2, 2] shape is what the extension type is for
        for tensor_column in [actual.column("tensor"), projected.column("tensor")]:
            ndarray = tensor_column.combine_chunks().to_numpy_ndarray()
            assert ndarray.shape == (2, 2, 2)
            np.testing.assert_array_equal(ndarray, [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

        assert projected.column("point").to_pylist() == [[1.0, 2.0], [3.0, 4.0]]
    else:
        # There is nothing to rebuild them from, only the physical parquet types are
        # left. They are not even the storage types: parquet does not record the
        # fixed length of a list, so both come back as a variable-length list.
        assert not any(isinstance(type_, pa.ExtensionType) for type_ in actual.schema.types)
        assert actual.schema.types == [
            pa.list_(pa.float64()),
            pa.list_(pa.float64()),
            pa.int64(),
        ]


def test_datafusion_read_table_unregistered_extension_type():
    """An unregistered extension type keeps its metadata, exactly as with pyarrow."""
    point_type = _PointExtensionType()
    pa.register_extension_type(point_type)
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "point.parquet")
    try:
        table = pa.table(
            {
                "point": pa.ExtensionArray.from_storage(
                    point_type, pa.array([[1.0, 2.0]], pa.list_(pa.float64(), 2))
                )
            }
        )
        pq.write_table(table, path)
    finally:
        pa.unregister_extension_type(point_type.extension_name)

    # The type is no longer registered, so pyarrow cannot rebuild it for either engine
    for read_table in [_pyarrow_read_table, _datafusion_read_table]:
        actual = read_table(path)
        assert actual.schema.field("point").type == point_type.storage_type
        assert actual.schema.field("point").metadata[b"ARROW:extension:name"] == b"nested_pandas.test.point"


@pytest.mark.parametrize(
    "data",
    [
        # Remote paths we can build an object store for
        "https://example.com/nested.parquet",
        UPath("https://example.com/nested.parquet"),
        UPath("s3://bucket/nested.parquet", key="AK", secret="SK"),
    ],
)
def test_datafusion_supported_remote(data):
    """A remote path datafusion can build an object store for is not an error."""
    _check_datafusion_support(data, columns=["nested.flux"])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        # Remote paths we cannot build an object store for
        ({"data": "http://example.com/nested.parquet"}, "cannot build an object store"),
        ({"data": "gs://bucket/nested.parquet"}, "cannot build an object store"),
        ({"data": UPath("s3://bucket/nested.parquet")}, "needs both 'key' and 'secret'"),
        ({"data": UPath("s3://bucket/nested.parquet", anon=True)}, "needs both 'key' and 'secret'"),
        ({"data": ["a.parquet", "b.parquet"]}, "single path only"),
        ({"data": io.BytesIO(b"")}, "single path only"),
        ({"filesystem": fsspec.implementations.http.HTTPFileSystem()}, "not supported"),
        ({"filters": pc.field("a") > 0.5}, "PyArrow Expression"),
        ({"read_dictionary": ["a"]}, "not supported"),
        ({"use_threads": False}, "use_threads=False"),
    ],
)
def test_datafusion_read_table_unsupported(kwargs, match):
    """What datafusion cannot serve raises, and names the engine to use instead."""
    kwargs = {"data": "tests/test_data/nested.parquet", **kwargs}
    with pytest.raises(ValueError, match=match):
        _datafusion_read_table(**kwargs)


@pytest.mark.parametrize(
    "data,expected_base,expected_store",
    [
        # HTTPS needs no credentials, and registers under scheme://host/
        ("https://data.example.com/a/b.parquet", "https://data.example.com/", "Http"),
        (UPath("https://data.example.com/a/b.parquet"), "https://data.example.com/", "Http"),
        # S3 with a key and a secret we can hand over directly
        (UPath("s3://bucket/a.parquet", key="AK", secret="SK"), "s3://bucket/", "AmazonS3"),
        (
            UPath("s3://bucket/a.parquet", key="AK", secret="SK", endpoint_url="http://e"),
            "s3://bucket/",
            "AmazonS3",
        ),
        # A region is the one client_kwarg we know how to pass on
        (
            UPath("s3://bucket/a.parquet", key="AK", secret="SK", client_kwargs={"region_name": "us-1"}),
            "s3://bucket/",
            "AmazonS3",
        ),
    ],
)
def test_datafusion_object_store(data, expected_base, expected_store):
    """We build a store only from what the path itself carries, never the environment."""
    base_url, store = _datafusion_object_store(UPath(data))
    assert base_url == expected_base
    assert type(store).__name__ == expected_store


@pytest.mark.parametrize(
    "data,match",
    [
        # Plain HTTP: the object store needs allow_http, which Http() does not take
        ("http://data.example.com/a.parquet", "cannot build an object store"),
        # Protocols we have not wired up
        (UPath("gs://bucket/a.parquet"), "cannot build an object store"),
        (UPath("az://container/a.parquet"), "cannot build an object store"),
        # No credentials at all, and datafusion cannot request anonymously
        (UPath("s3://bucket/a.parquet"), "needs both 'key' and 'secret'"),
        (UPath("s3://bucket/a.parquet", anon=True), "needs both 'key' and 'secret'"),
        # Half a credential pair is not enough
        (UPath("s3://bucket/a.parquet", key="AK"), "needs both 'key' and 'secret'"),
        # Options we would silently drop, named in the message
        (
            UPath("https://data.example.com/a.parquet", headers={"key": "value"}),
            r"HTTPS storage options on to its object store: \['headers'\]",
        ),
        (
            UPath("s3://bucket/a.parquet", key="AK", secret="SK", use_ssl=False),
            r"S3 storage options on to its object store: \['use_ssl'\]",
        ),
        (
            UPath("s3://bucket/a.parquet", key="AK", secret="SK", client_kwargs={"verify": False}),
            r"S3 storage options on to its object store: \['verify'\]",
        ),
        # A session token: AmazonS3 cannot take one on our oldest datafusion
        (
            UPath("s3://bucket/a.parquet", key="AK", secret="SK", token="T"),
            r"S3 storage options on to its object store: \['token'\]",
        ),
    ],
)
def test_datafusion_object_store_unsupported(data, match):
    """A path we cannot build a store for raises, and says which part we choked on."""
    with pytest.raises(ValueError, match=match):
        _datafusion_object_store(UPath(data))


def test_datafusion_object_store_is_reused():
    """Building a store stands up an HTTPS client, so equivalent paths share one."""
    https = UPath("https://data.example.com/a.parquet")
    other = UPath("https://data.example.com/b/c.parquet")
    # Same host, so the same base URL and the same store object
    assert _datafusion_object_store(https)[1] is _datafusion_object_store(other)[1]
    # A different host gets its own
    assert (
        _datafusion_object_store(https)[1]
        is not _datafusion_object_store(UPath("https://other.example.com/a.parquet"))[1]
    )

    s3 = UPath("s3://bucket/a.parquet", key="AK", secret="SK")
    assert (
        _datafusion_object_store(s3)[1]
        is _datafusion_object_store(UPath("s3://bucket/b.parquet", key="AK", secret="SK"))[1]
    )
    # Different credentials must never share a client
    assert (
        _datafusion_object_store(s3)[1]
        is not _datafusion_object_store(UPath("s3://bucket/a.parquet", key="AK", secret="OTHER"))[1]
    )


def test_datafusion_object_store_does_not_mutate_storage_options():
    """Reading the options out of a UPath leaves the UPath alone."""
    path = UPath("s3://bucket/a.parquet", key="AK", secret="SK", client_kwargs={"region_name": "us-1"})
    _datafusion_object_store(path)
    assert dict(path.storage_options) == {
        "key": "AK",
        "secret": "SK",
        "client_kwargs": {"region_name": "us-1"},
    }


def test_datafusion_read_table_local_filesystem():
    """A local pyarrow filesystem is accepted, it is what datafusion would use anyway."""
    path = UPath("tests/test_data/nested.parquet")
    actual = _datafusion_read_table(path.path, filesystem=pyarrow.fs.LocalFileSystem())
    assert actual.column_names == ["a", "flux", "nested", "lincc"]


@pytest.mark.parametrize(
    "filters,match",
    [
        ("a > 0.5", "filters must be"),
        (42, "filters must be"),
        ([], "filters format must be"),
        ([[]], "filters format must be"),
        ([("a", ">", 0.5), "not_a_tuple"], "filters format must be"),
        ([[("a", ">", 0.5, "extra")]], "filters format must be"),
        ([("a", "~=", 0.5)], "Unsupported filter operator"),
    ],
)
def test_datafusion_filters_to_expr_errors(filters, match):
    """Malformed filters are rejected with the messages the pyarrow path uses."""
    with pytest.raises(ValueError, match=match):
        _datafusion_filters_to_expr(filters)
