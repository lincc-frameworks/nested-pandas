# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import pandas as pd

from nested_pandas.series import packer
from nested_pandas.series.dtype import NestedDtype

from .utils import _ensure_spacing


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

    def _split_query(self, expr) -> dict:
        """Splits a pandas query into multiple subqueries for nested and base layers"""
        # Ensure query has needed spacing for upcoming split
        expr = _ensure_spacing(expr)
        nest_exprs = {col: [] for col in self.nested_cols + ["base"]}  # type: dict
        split_expr = expr.split(" ")

        i = 0
        current_focus = "base"
        while i < len(split_expr):
            expr_slice = split_expr[i].strip("()")
            # Check if it's a nested column
            if self._is_known_hierarchical_column(expr_slice):
                nested, colname = split_expr[i].split(".")
                current_focus = nested.strip("()")
                # account for parentheses
                j = 0
                while j < len(nested):
                    if nested[j] == "(":
                        nest_exprs[current_focus].append("(")
                    j += 1
                nest_exprs[current_focus].append(colname)
            # or if it's a top-level column
            elif expr_slice in self.columns:
                current_focus = "base"
                nest_exprs[current_focus].append(split_expr[i])
            else:
                nest_exprs[current_focus].append(split_expr[i])
            i += 1
        return {expr: " ".join(nest_exprs[expr]) for expr in nest_exprs if len(nest_exprs[expr]) > 0}

    def query(self, expr) -> Self:  # type: ignore[name-defined] # noqa: F821
        """overwrite query to first check which columns should be queried"""

        # Rebuild queries for each specified nested/base layer
        exprs_to_use = self._split_query(expr)

        # For now (simplicity), limit query to only operating on one layer
        if len(exprs_to_use.keys()) != 1:
            raise ValueError("Queries cannot target multiple structs/layers, write a separate query for each")

        # Send queries to layers
        # We'll only execute 1 per the Error above, but the loop will be useful
        # for when/if we allow multi-layer queries
        result = self.copy()
        for expr in exprs_to_use:
            if expr == "base":
                result = super().query(exprs_to_use["base"], inplace=False)
            else:
                # TODO: does not work with queries that empty the dataframe
                result[expr] = result[expr].ts.query_flat(exprs_to_use[expr])
        return result
