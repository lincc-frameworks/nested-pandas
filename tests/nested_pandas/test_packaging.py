import nested_pandas


def test_version():
    """Check to see that the version property returns something"""
    assert nested_pandas.__version__ is not None
