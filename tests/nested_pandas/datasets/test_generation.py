import pytest
from nested_pandas.datasets import generate_data


@pytest.mark.parametrize("n_layers", [10, {"nested_a": 10, "nested_b": 20}])
def test_generate_data(n_layers):
    """test the data generator function"""
    nf = generate_data(10, n_layers, seed=1)

    if isinstance(n_layers, int):
        assert len(nf.nested.nest.to_flat()) == 100

    elif isinstance(n_layers, dict):
        assert "nested_a" in nf.columns
        assert "nested_b" in nf.columns

        assert len(nf.nested_a.nest.to_flat()) == 100
        assert len(nf.nested_b.nest.to_flat()) == 200


def test_generate_data_bad_input():
    """test a poor n_layer input to generate_data"""
    with pytest.raises(TypeError):
        generate_data(10, "nested", seed=1)
