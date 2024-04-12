import pytest
from nested_pandas.nestedframe import utils


@pytest.mark.parametrize(
    "in_out",
    [
        ("a>3", "a > 3"),
        ("test.a>5&b==2", "test.a > 5 & b == 2"),
        ("b > 3", "b > 3"),
        ("(a.b > 3)&(a.c == 'f')", "(a.b > 3) & (a.c == 'f')"),
    ],
)
def test_ensure_spacing(in_out):
    """test a set of input queries to make sure spacing is done correctly"""
    expr, output = in_out
    assert utils._ensure_spacing(expr) == output
