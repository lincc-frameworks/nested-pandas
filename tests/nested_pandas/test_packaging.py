import nested_pandas


def test_version():
    """Check to see that we can get the package version"""
    assert nested_pandas.__version__ is not None
