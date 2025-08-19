import pandas as pd

from nested_pandas.series.dtype import NestedDtype


def nested_only(func):
    """Decorator to designate certain functions can only be used with NestedDtype."""

    def wrapper(*args, **kwargs):
        if not isinstance(args[0].dtype, NestedDtype):
            raise TypeError(f"'{func.__name__}' can only be used with a NestedDtype, not '{args[0].dtype}'.")

        result = func(*args, **kwargs)
        return result

    return wrapper


class NestedSeries(pd.Series):
    """
    A Series that can contain nested data structures, such as lists or dictionaries.
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

        if not isinstance(self.dtype, NestedDtype):
            return super().__getitem__(key)
        # Return a flatten series for a single field
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

        return super().__setitem__(key, value)

    @nested_only
    def to_flat(self):
        """Convert to a flat dataframe representation of the nested series."""
        return self.nest.to_flat()

    @nested_only
    def to_lists(self):
        """Convert to a list representation of the nested series."""
        return self.nest.to_lists()
