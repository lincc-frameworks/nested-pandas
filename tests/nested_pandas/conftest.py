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
