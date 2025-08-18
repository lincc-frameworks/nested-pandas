import pandas as pd

class NestedSeries(pd.Series):
    """
    A Series that can contain nested data structures, such as lists or dictionaries.
    This class extends the functionality of a standard pandas Series to handle nested data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def fields(self):
        """ Returns the fields of the nested series as a list. """
        return self.nest.fields

    @property
    def flat_length(self):
        """ Returns the length of the flattened nested series."""
        return self.nest.flat_length

    @property
    def list_lengths(self):
        """ Returns the lengths of the list-packed nested series."""
        return self.nest.list_lengths

    def to_flat(self):
        """ Convert to a flat dataframe representation of the nested series. """
        return self.nest.to_flat()

    def to_lists(self):
        """ Convert to a list representation of the nested series. """
        return self.nest.to_lists()