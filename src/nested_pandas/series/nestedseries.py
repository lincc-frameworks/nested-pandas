from functools import wraps

import pandas as pd

from nested_pandas.series.dtype import NestedDtype

__all__ = ["NestedSeries"]


def nested_only(func):
    """Decorator to designate certain functions can only be used with NestedDtype."""

    @wraps(func)  # This ensures the original function's metadata is preserved
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].dtype, NestedDtype):
            raise TypeError(f"'{func.__name__}' can only be used with a NestedDtype, not '{args[0].dtype}'.")

        result = func(*args, **kwargs)
        return result

    return wrapper


class NestedSeries(pd.Series):
    """
    A Series that can contain nested data structures, such as lists or data-frames.
    This class extends the functionality of a standard pandas Series to handle nested data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @nested_only
    def fields(self):
        """Returns the fields of the nested series as a list."""
        return self.nest.fields

    @property
    @nested_only
    def flat_length(self):
        """Returns the length of the flattened nested series."""
        return self.nest.flat_length

    @property
    @nested_only
    def list_lengths(self):
        """Returns the lengths of the list-packed nested series."""
        return self.nest.list_lengths

    def __getitem__(self, key):
        """Equip getitem with ability to handle nested data."""

        # Pure pandas Series behavior if not a NestedDtype
        if not isinstance(self.dtype, NestedDtype):
            return super().__getitem__(key)

        # Return a flattened series for a single field
        if isinstance(key, str) and key in self.fields:
            return self.nest[key]

        # For list-like keys, perform sub-column selection
        elif isinstance(key, list | tuple) and all(isinstance(k, str) for k in key):
            return self.nest[key]

        # Handle boolean masking
        if isinstance(key, pd.Series) and pd.api.types.is_bool_dtype(key.dtype):
            return self.nest[key]

        # Otherwise, fall back to the default behavior
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        """Equip setitem with ability to handle nested data."""
        # Pure pandas Series behavior if not a NestedDtype
        if not isinstance(self.dtype, NestedDtype):
            return super().__setitem__(key, value)

        # Use nest setitem when setting on a single field
        if isinstance(key, str) and key in self.fields:
            self.nest[key] = value
            return

        return super().__setitem__(key, value)

    @nested_only
    def to_flat(self, fields: list[str] | None = None) -> pd.DataFrame:
        """Convert nested series into dataframe of flat arrays.

        Parameters
        ----------
        fields : list[str] or None, optional
            Names of the fields to include. Default is None, which means all fields.

        Returns
        -------
        pd.DataFrame
            Dataframe of flat arrays.

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5, 2, seed=1)

        >>> nf["nested"].to_flat()
                   t       flux band
        0    8.38389  80.074457    r
        0   13.40935  89.460666    g
        1   13.70439  96.826158    g
        1   8.346096   8.504421    g
        2   4.089045  31.342418    g
        2  11.173797   3.905478    g
        3  17.562349  69.232262    r
        3   2.807739  16.983042    r
        4   0.547752  87.638915    g
        4    3.96203   87.81425    r

        """
        return self.nest.to_flat(fields=fields)

    @nested_only
    def to_lists(self, fields: list[str] | None = None) -> pd.DataFrame:
        """Convert nested series into dataframe of list-array columns.

        Parameters
        ----------
        fields : list[str] or None, optional
            Names of the fields to include. Default is None, which means all fields.

        Returns
        -------
        pd.DataFrame
            Dataframe of list-arrays.

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5, 2, seed=1)

        >>> nf["nested"].to_lists()
                                   t                       flux       band
        0  [ 8.38389029 13.4093502 ]  [80.07445687 89.46066635]  ['r' 'g']
        1  [13.70439001  8.34609605]  [96.82615757  8.50442114]  ['g' 'g']
        2  [ 4.08904499 11.17379657]  [31.34241782  3.90547832]  ['g' 'g']
        3  [17.56234873  2.80773877]  [69.23226157 16.98304196]  ['r' 'r']
        4    [0.54775186 3.96202978]  [87.63891523 87.81425034]  ['g' 'r']
        """
        return self.nest.to_lists(fields=fields)
