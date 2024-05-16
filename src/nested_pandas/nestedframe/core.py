# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas._libs import lib
from pandas._typing import Any, AnyAll, Axis, IndexLabel
from pandas.api.extensions import no_default

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

    def _is_known_column(self, colname) -> bool:
        """Determine whether a string is a known column name"""
        return colname in self.columns or self._is_known_hierarchical_column(colname)

    def add_nested(
        self, obj, name: str, *, dtype: NestedDtype | pd.ArrowDtype | pa.DataType | None = None
    ) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Packs input object to a nested column and adds it to the NestedFrame

        This method returns a new NestedFrame with the added nested column.

        Parameters
        ----------
        obj : pd.DataFrame or a sequence of items convertible to nested structures
            The object to be packed into nested pd.Series and added to
            the NestedFrame. If a DataFrame is passed, it must have non-unique
            index values, which are used to pack the DataFrame. If a sequence
            of elements is passed, it is packed into a nested pd.Series.
            Sequence elements may be individual pd.DataFrames, dictionaries
            (keys are nested column names, values are arrays of the same
            length), or any other object convertible to pa.StructArray.
            Additionally, None and pd.NA are allowed as elements to represent
            missing values.
        name : str
            The name of the nested column to be added to the NestedFrame.
        dtype : dtype or None
            NestedDtype to use for the nested column; pd.ArrowDtype or
            pa.DataType can also be used to specify the nested dtype. If None,
            the dtype is inferred from the input object.

        Returns
        -------
        NestedFrame
            A new NestedFrame with the added nested column.
        """
        # Add sources to objects
        packed = packer.pack(obj, name=name, dtype=dtype)
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

    def _resolve_dropna_target(self, on_nested, subset):
        """resolves the target layer for a given set of dropna kwargs"""

        nested_cols = self.nested_columns
        columns = self.columns

        # first check the subset kwarg input
        subset_target = []
        if subset:
            if isinstance(subset, str):
                subset = [subset]

            for col in subset:
                col = col.split(".")[0]
                if col in nested_cols:
                    subset_target.append(col)
                elif col in columns:
                    subset_target.append("base")
                else:
                    raise ValueError(f"Column name {col} not found in any base or nested columns")

            # Check for 1 target
            subset_target = np.unique(subset_target)
            if len(subset_target) > 1:  # prohibit multi-target operations
                raise ValueError(
                    f"Targeted multiple nested structures ({subset_target}), write one command per target dataframe"  # noqa
                )
            subset_target = str(subset_target[0])

        # Next check the on_nested kwarg input
        if on_nested and on_nested not in nested_cols:
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
        return target, subset

    def dropna(
        self,
        *,
        axis: Axis = 0,
        how: AnyAll | lib.NoDefault = no_default,
        thresh: int | lib.NoDefault = no_default,
        on_nested: bool = False,
        subset: IndexLabel | None = None,
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> NestedFrame | None:
        """
        Remove missing values for one layer of the NestedFrame.

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

        Values for `on_nested` and `subset` should be consistent in pointing
        to a single layer, multi-layer operations are not supported at this
        time.
        """

        # determine target dataframe
        target, subset = self._resolve_dropna_target(on_nested, subset)

        if target == "base":
            return super().dropna(
                axis=axis, how=how, thresh=thresh, subset=subset, inplace=inplace, ignore_index=ignore_index
            )
        if subset is not None:
            subset = [col.split(".")[-1] for col in subset]
        if inplace:
            target_flat = self[target].nest.to_flat()
            target_flat.dropna(
                axis=axis,
                how=how,
                thresh=thresh,
                subset=subset,
                inplace=inplace,
                ignore_index=ignore_index,
            )
            self[target] = packer.pack_flat(target_flat)
            return self
        # Or if not inplace
        new_df = self.copy()
        new_df[target] = packer.pack_flat(
            new_df[target]
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
        return new_df

    def reduce(self, func, *args, **kwargs) -> NestedFrame:
        """
        Takes a function and applies it to each top-level row of the NestedFrame.

        The user may specify which columns the function is applied to, with
        columns from the 'base' layer being passsed to the function as
        scalars and columns from the nested layers being passed as numpy arrays.

        Parameters
        ----------
        func : callable
            Function to apply to each nested dataframe. The first arguments to `func` should be which
            columns to apply the function to.
        args : positional arguments
            Positional arguments to pass to the function, the first *args should be the names of the
            columns to apply the function to.
        kwargs : keyword arguments, optional
            Keyword arguments to pass to the function.

        Returns
        -------
        `NestedFrame`
            `NestedFrame` with the results of the function applied to the columns of the frame.

        Notes
        -----
        The recommend return value of func should be a `pd.Series` where the indices are the names of the
        output columns in the dataframe returned by `reduce`. Note however that in cases where func
        returns a single value there may be a performance benefit to returning the scalar value
        rather than a `pd.Series`.

        Example User Function:
        ```
        import pandas as pd

        def my_sum(col1, col2):
            return pd.Series(
                [sum(col1), sum(col2)],
                index=["sum_col1", "sum_col2"],
            )

        ```

        """
        # Parse through the initial args to determine the columns to apply the function to
        requested_columns = []
        for arg in args:
            if not isinstance(arg, str) or not self._is_known_column(arg):
                # We've reached an argument that is not a valid column, so we assume
                # the remaining args are extra arguments to the function
                break
            layer = "base" if "." not in arg else arg.split(".")[0]
            col = arg.split(".")[-1]
            requested_columns.append((layer, col))

        # We require the first *args to be the columns to apply the function to
        if not requested_columns:
            raise ValueError("No columns in `*args` specified to apply function to")

        # The remaining args are the extra arguments to the function other than columns
        extra_args: tuple[Any, ...] = ()  # empty tuple to make mypy happy
        if len(requested_columns) < len(args):
            extra_args = args[len(requested_columns) :]

        # find targeted layers
        layers = np.unique([col[0] for col in requested_columns])

        # build a flat dataframe with array columns to apply to the function
        apply_df = NestedFrame()
        for layer in layers:
            if layer == "base":
                columns = [col[1] for col in requested_columns if col[0] == layer]
                apply_df = apply_df.join(self[columns], how="outer")
            else:
                # TODO: It should be faster to pass these columns to to_lists, but its 20x slower
                # columns = [col[1] for col in requested_columns if col[0] == layer]
                apply_df = apply_df.join(self[layer].nest.to_lists(), how="outer")

        # Translates the requested columns into the scalars or arrays we pass to func.
        def translate_cols(frame, layer, col):
            if layer == "base":
                # We pass the "base" column as a scalar
                return frame[col]
            return np.asarray(frame[col])

        # send arrays along to the apply call
        result = apply_df.apply(
            lambda x: func(
                *[translate_cols(x, layer, col) for layer, col in requested_columns], *extra_args, **kwargs
            ),
            axis=1,  # to apply func on each row of our nested frame)
        )
        return result

    def to_parquet(self, path, by_layer=False, **kwargs) -> None:
        """Creates parquet file(s) with the data of a NestedFrame, either
        as a single parquet file where each nested dataset is packed into its
        own column or as an individual parquet file for each layer.

        Note that here we always opt to use the pyarrow engine for writing
        parquet files.

        Parameters
        ----------
        path : str
            The path to the parquet file to be written if 'by_layer' is False.
            If 'by_layer' is True, this should be the path to an existing.
        by_layer : bool, default False
            If False, writes the entire NestedFrame to a single parquet file.

            If True, writes each layer to a separate parquet file within the
            directory specified by path. The filename for each outputted file will
            be named after its layer and then the ".parquet" extension.
            For example for the base layer this is always "base.parquet".
        kwargs : keyword arguments, optional
            Keyword arguments to pass to the function.

        Returns
        -------
        None
        """
        if not by_layer:
            # We just defer to the pandas to_parquet method if we're not writing by layer
            # or there is only one layer in the NestedFrame.
            super().to_parquet(path, engine="pyarrow", **kwargs)
        else:
            # If we're writing by layer, path must be an existing directory
            if not os.path.isdir(path):
                raise ValueError("The provided path must be an existing directory if by_layer=True")

            # Write the base layer to a parquet file
            base_frame = self.drop(columns=self.nested_columns, inplace=False)
            base_frame.to_parquet(os.path.join(path, "base.parquet"), by_layer=False, **kwargs)

            # Write each nested layer to a parquet file
            for layer in self.all_columns:
                if layer != "base":
                    path_layer = os.path.join(path, f"{layer}.parquet")
                    self[layer].nest.to_flat().to_parquet(path_layer, engine="pyarrow", **kwargs)
        return None
