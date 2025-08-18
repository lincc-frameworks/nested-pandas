import pandas as pd

class NestedSeries(pd.Series):
    """
    A Series that can contain nested data structures, such as lists or dictionaries.
    This class extends the functionality of a standard pandas Series to handle nested data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nested = True

    