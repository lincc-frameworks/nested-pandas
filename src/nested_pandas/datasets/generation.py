import numpy as np

from nested_pandas import NestedFrame


def generate_data(n_base, n_layer, seed=None) -> NestedFrame:
    """Generates a toy dataset.

    Parameters
    ----------
    n_base : int
        The number of rows to generate for the base layer
    n_layer : int, or dict
        The number of rows per n_base row to generate for a nested layer.
        Alternatively, a dictionary of layer label, layer_size pairs may be
        specified to created multiple nested columns with custom sizing.
    seed : int
        A seed to use for random generation of data

    Returns
    -------
    NestedFrame
        The constructed NestedFrame.

    Examples
    --------
    >>> nested_pandas.datasets.generate_data(10,100)
    >>> nested_pandas.datasets.generate_data(10, {"nested_a": 100, "nested_b": 200})
    """
    # use provided seed, "None" acts as if no seed is provided
    randomstate = np.random.RandomState(seed=seed)

    # Generate base data
    base_data = {"a": randomstate.random(n_base), "b": randomstate.random(n_base) * 2}
    base_nf = NestedFrame(data=base_data)

    # In case of int, create a single nested layer called "nested"
    if isinstance(n_layer, int):
        n_layer = {"nested": n_layer}

    # It should be a dictionary
    if isinstance(n_layer, dict):
        for key in n_layer:
            layer_size = n_layer[key]
            layer_data = {
                "t": randomstate.random(layer_size * n_base) * 20,
                "flux": randomstate.random(layer_size * n_base) * 100,
                "band": randomstate.choice(["r", "g"], size=layer_size * n_base),
                "index": np.arange(layer_size * n_base) % n_base,
            }
            layer_nf = NestedFrame(data=layer_data).set_index("index")
            base_nf = base_nf.add_nested(layer_nf, key)
        return base_nf
    else:
        raise TypeError("Input to n_layer is not an int or dict.")


def generate_parquet_file(n_base, n_layer, path, file_per_layer=False, seed=None):
    """Generates a toy dataset and outputs it to one or more parquet files.

    Parameters
    ----------
    n_base : int
        The number of rows to generate for the base layer
    n_layer : int, or dict
        The number of rows per n_base row to generate for a nested layer.
        Alternatively, a dictionary of layer label, layer_size pairs may be
        specified to created multiple nested columns with custom sizing.
    path : str,
        The path to the parquet file to write to if `file_per_layer` is `False`,
        and otherwise the path to the directory to write the parquet file for
        each layer.
    file_per_layer : bool, default=False
        If True, write each layer to its own parquet file. Otherwise, write
        the generated to a single parquet file representing a nested dataset.
    seed : int, default=None
        A seed to use for random generation of data

    Returns
    -------
    None
    """
    nf = generate_data(n_base, n_layer, seed)
    nf.to_parquet(path, by_layer=file_per_layer)
