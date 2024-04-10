# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas._libs import lib
from pandas._typing import AnyAll, Axis, IndexLabel

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
        nest_exprs = {col: [] for col in self.nested_columns + ["base"]}  # type: dict
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
        """
        Query the columns of a NestedFrame with a boolean expression. Specified
        queries can target nested columns in addition to the typical column set

        Parameters
        ----------
        expr : str
            The query string to evaluate.

            Access nested columns using `nested_df.nested_col` (where
            `nested_df` refers to a particular nested dataframe and
            `nested_col` is a column of that nested dataframe).

            You can refer to variables
            in the environment by prefixing them with an '@' character like
            ``@a + b``.

            You can refer to column names that are not valid Python variable names
            by surrounding them in backticks. Thus, column names containing spaces
            or punctuations (besides underscores) or starting with digits must be
            surrounded by backticks. (For example, a column named "Area (cm^2)" would
            be referenced as ```Area (cm^2)```). Column names which are Python keywords
            (like "list", "for", "import", etc) cannot be used.

            For example, if one of your columns is called ``a a`` and you want
            to sum it with ``b``, your query should be ```a a` + b``.

        Returns
        -------
        DataFrame
            DataFrame resulting from the provided query expression.

        Notes
        -----
        Queries that target a particular nested structure return a dataframe
        with rows of that particular nested structure filtered. For example,
        querying the NestedFrame "df" with nested structure "my_nested" as
        below will return all rows of df, but with mynested filtered by the
        condition:

        >>> df.query("mynested.a > 2")
        """

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
                result[expr] = result[expr].nest.query_flat(exprs_to_use[expr])
        return result

    def dropna(
        self,
        *,
        axis: Axis = 0,
        how: AnyAll | lib.NoDefault = lib.no_default,
        thresh: int | lib.NoDefault = lib.no_default,
        on_nested: bool = False,
        subset: IndexLabel | None = None,
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> NestedFrame | None:
        """
        Remove missing values.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Determine if rows or columns which contain missing values are
            removed.

            * 0, or 'index' : Drop rows which contain missing values.
            * 1, or 'columns' : Drop columns which contain missing value.

            Only a single axis is allowed.

        how : {'any', 'all'}, default 'any'
            Determine if row or column is removed from DataFrame, when we have
            at least one NA or all NA.

            * 'any' : If any NA values are present, drop that row or column.
            * 'all' : If all values are NA, drop that row or column.
        thresh : int, optional
            Require that many non-NA values. Cannot be combined with how.
        on_nested : str or bool, optional
            If not False, applies the call to the nested dataframe in the
            column with label equal to the provided string. If specified,
            the nested dataframe should align with any columns given in
            `subset`.
        subset : column label or sequence of labels, optional
            Labels along other axis to consider, e.g. if you are dropping rows
            these would be a list of columns to include.

            Access nested columns using `nested_df.nested_col` (where
            `nested_df` refers to a particular nested dataframe and
            `nested_col` is a column of that nested dataframe).
        inplace : bool, default False
            Whether to modify the DataFrame rather than creating a new one.
        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 2.0.0

        Returns
        -------
        DataFrame or None
            DataFrame with NA entries dropped from it or None if ``inplace=True``.

        Notes
        -----
        Operations that target a particular nested structure return a dataframe
        with rows of that particular nested structure affected.
        """

        # determine target dataframe

        # first check the subset kwarg input
        subset_target = []
        if subset:
            if isinstance(subset, str):
                subset = [subset]
            for col in subset:
                col = col.split(".")[0] if "." in col else col
                if col in self.nested_columns:
                    subset_target.append(col)
                elif col in self.columns:
                    subset_target.append("base")

            # Check for 1 target
            subset_target = np.unique(subset_target)
            if len(subset_target) > 1:  # prohibit multi-target operations
                raise ValueError(
                    f"Targeted multiple nested structures ({subset_target}), write one command per target dataframe"  # noqa
                )
            elif len(subset_target) == 0:
                raise ValueError(
                    "Provided base columns or nested layer did not match any found in the nestedframe"
                )
            subset_target = subset_target[0]

        # Next check the on_nested kwarg input
        if on_nested not in self.nested_columns:
            raise ValueError("Provided nested layer not found in nested dataframes")

        # Resolve target layer
        target = "base"
        if on_nested and subset_target:
            if on_nested != subset_target:
                raise ValueError(
                    f"Provided on_nested={on_nested}, but subset columns are from {subset_target}. Make sure these are aligned or just use subset."  # noqa
                )
            else:
                target = subset_target
        elif on_nested:
            target = str(on_nested)
        elif subset_target:
            target = str(subset_target)

        if target == "base":
            return super().dropna(
                axis=axis, how=how, thresh=thresh, subset=subset, inplace=inplace, ignore_index=ignore_index
            )
        else:
            if subset is not None:
                subset = [col.split(".")[-1] for col in subset]
            self[target] = packer.pack_flat(
                self[target]
                .nest.to_flat()
                .dropna(
                    axis=axis,
                    how=how,
                    thresh=thresh,
                    subset=subset,
                    inplace=inplace,
                    ignore_index=ignore_index,
                )
            )
            return self
