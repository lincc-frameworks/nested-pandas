"""Shared fixtures for the nested-pandas test suite."""

from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
"""Root of the test data files, resolved from this file so tests work from any working directory."""


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Root directory of the test data files, ``tests/data``."""
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def nested_data_dir(test_data_dir) -> Path:
    """Directory of parquet files with nested columns, ``tests/data/nested``."""
    return test_data_dir / "nested"


@pytest.fixture(scope="session")
def list_struct_data_dir(test_data_dir) -> Path:
    """Directory of parquet files with list-of-struct columns, ``tests/data/list_struct``."""
    return test_data_dir / "list_struct"


@pytest.fixture(scope="session")
def fixed_size_list_data_dir(test_data_dir) -> Path:
    """Directory of parquet files with fixed-size-list columns, ``tests/data/fixed_size_list``."""
    return test_data_dir / "fixed_size_list"


@pytest.fixture(scope="session")
def nested_parquet_path(nested_data_dir) -> Path:
    """Small file with two nested columns, ``nested`` and ``lincc``, next to flat ``a`` and ``flux``."""
    return nested_data_dir / "nested.parquet"


@pytest.fixture(scope="session")
def not_nestable_parquet_path(nested_data_dir) -> Path:
    """File with a struct column whose list fields have unequal lengths, so it cannot be nested."""
    return nested_data_dir / "not_nestable.parquet"


@pytest.fixture(scope="session")
def vsx_ztf_parquet_path(nested_data_dir) -> Path:
    """VSX x ZTF DR22 crossmatch with a doubly nested ``ztf.lc`` column."""
    return nested_data_dir / "vsx-x-ztfdr22_lc-m31.parquet"


@pytest.fixture(scope="session")
def list_struct_parquet_path(list_struct_data_dir) -> Path:
    """File with a list-of-struct ``lightcurve`` column."""
    return list_struct_data_dir / "list_struct.parquet"


@pytest.fixture(scope="session")
def mmu_desi_parquet_path(fixed_size_list_data_dir) -> Path:
    """DESI file with a fixed-size struct-list ``spectrum`` column."""
    return fixed_size_list_data_dir / "mmu-desi.parquet"


@pytest.fixture(scope="session")
def fixed_size_list_struct_parquet_path(fixed_size_list_data_dir) -> Path:
    """File with a fixed-size list-of-struct ``fixed_nested`` column."""
    return fixed_size_list_data_dir / "fixed-size-list-struct.parquet"
