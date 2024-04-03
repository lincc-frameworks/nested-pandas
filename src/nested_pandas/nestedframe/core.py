# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import pandas as pd

from nested_pandas.series import packer
from nested_pandas.series.dtype import NestedDtype


class NestedFrame(pd.DataFrame):
    """A Pandas Dataframe extension with support for nested structure.

    See https://pandas.pydata.org/docs/development/extending.html#subclassing-pandas-data-structures
    """

    # normal properties
    _metadata = ["added_property"]

    @property
    def _constructor(self) -> Self:  # type: ignore[name-defined] # noqa: F821
        return NestedFrame

    @property
    def _constructor_expanddim(self) -> Self:  # type: ignore[name-defined] # noqa: F821
        return NestedFrame

    @property
    def all_columns(self) -> dict:
        """returns a dictionary of columns for each base/nested dataframe"""
        all_columns = {"base": self.columns}
        for column in self.columns:
            if isinstance(self[column].dtype, NestedDtype):
                nest_cols = self[column].nest.fields
                all_columns[column] = nest_cols
        return all_columns

    @property
    def nested_columns(self) -> list:
        """retrieves the base column names for all nested dataframes"""
        nest_cols = []
        for column in self.columns:
            if isinstance(self[column].dtype, NestedDtype):
                nest_cols.append(column)
        return nest_cols

    def _is_known_hierarchical_column(self, colname) -> bool:
        """Determine whether a string is a known hierarchical column name"""
        if "." in colname:
            left, right = colname.split(".")
            if left in self.nested_columns:
                return right in self.all_columns[left]
            return False
        return False

    def add_nested(self, nested, name) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Packs a dataframe into a nested column"""
        # Add sources to objects
        packed = packer.pack_flat(nested, name=name)
        label = packed.name
        return self.assign(**{f"{label}": packed})
