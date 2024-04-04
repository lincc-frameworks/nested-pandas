import pytest
from nested_pandas.series.na import NA


def test_na_is_singleton():
    """Test that NA is a singleton instance"""
    assert NA is NA


def test_na_repr():
    """Test that NA has the correct representation."""
    assert repr(NA) == "<NA>"


def test_na_format():
    """Test that NA has the correct format."""
    assert f"{NA}" == "<NA>"


def test_na_bool():
    """Test that NA raises TypeError when converted to bool."""
    with pytest.raises(TypeError):
        bool(NA)


def test_na_eq():
    """Test that NA is not equal to anything."""
    assert NA != 1
    assert NA != 1.0
    assert NA != "1"
    assert NA != NA


def test_na_neq():
    """Test that NA is not equal to anything."""
    assert NA != 1
    assert NA != 1.0
    assert NA != "1"
    assert [] != NA
    assert {} != NA
    assert NA != ()
    assert set() != NA
    assert NA != NA
    assert object() != NA


def test_hash():
    """Test that hash(NA) is always the same."""
    assert hash(NA) == hash(NA)
    assert {NA, NA} == {NA}
