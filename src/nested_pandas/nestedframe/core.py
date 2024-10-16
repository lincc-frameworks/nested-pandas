# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import ast
import os

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas._libs import lib
from pandas._typing import Any, AnyAll, Axis, IndexLabel
from pandas.api.extensions import no_default
from pandas.core.computation.expr import PARSERS, PandasExprVisitor

from nested_pandas.series import packer
from nested_pandas.series.dtype import NestedDtype

from ..series.packer import pack_sorted_df_into_struct
from .utils import NestingType, check_expr_nesting


class NestedPandasExprVisitor(PandasExprVisitor):
    """
    Custom expression visitor for NestedFrame evaluations, which may assign to
    nested columns.
    """

    def visit_Assign(self, node, **kwargs):  # noqa: N802
        """
        Visit an assignment node, which may assign to a nested column.
        """
        if not isinstance(node.targets[0], ast.Attribute):
            # If the target is not an attribute, then it's a simple assignment as usual
            return super().visit_Assign(node)
        target = node.targets[0]
        if not isinstance(target.value, ast.Name):
            raise ValueError("Assignments to nested columns must be of the form `nested.col = ...`")
        # target.value.id will be the name of the nest, target.attr is the column name.
        # Describing the proper target for the assigner is enough for both overwrite and
        # creation of new columns.  The assigner will be a string like "nested.col".
        # This works both for the creation of new nest members and new nests.
        self.assigner = f"{target.value.id}.{target.attr}"
        # Continue visiting.
        return self.visit(node.value, **kwargs)


PARSERS["nested-pandas"] = NestedPandasExprVisitor


class _SeriesFromNest(pd.Series):
    """
    Series that were unpacked from a nest.
    """

    _metadata = ["nest_name", "flat_nest"]

    @property
    def _constructor(self) -> Self:  # type: ignore[name-defined] # noqa: F821
        return _SeriesFromNest

    @property
    def _constructor_expanddim(self) -> Self:  # type: ignore[name-defined] # noqa: F821
        return NestedFrame

    # https://pandas.pydata.org/docs/development/extending.html#arithmetic-with-3rd-party-types
    # The __pandas_priority__ of Series is 3000, so give _SeriesFromNest a
    # higher priority, so that binary operations involving this class and
    # Series produce instances of this class, preserving the type and origin.
    __pandas_priority__ = 3500


class _NestResolver(dict):
    """
    Used by NestedFrame.eval to resolve the names of nests at the top level.
    While the resolver is normally a dictionary, with values that are fixed
    upon entering evaluation, this object needs to be dynamic so that it can
    support multi-line expressions, where new nests may be created during
    evaluation.
    """

    def __init__(self, outer: NestedFrame):
        self._outer = outer
        super().__init__()

    def __contains__(self, item):
        top_nest = item if "." not in item else item.split(".")[0].strip()
        return top_nest in self._outer.nested_columns

    def __getitem__(self, item):
        top_nest = item if "." not in item else item.split(".")[0].strip()
        if not super().__contains__(top_nest):
            if top_nest not in self._outer.nested_columns:
                raise KeyError(f"Unknown nest {top_nest}")
            super().__setitem__(top_nest, _NestedFieldResolver(top_nest, self._outer))
        return super().__getitem__(top_nest)

    def __setitem__(self, key, value):
        # Called to update the resolver with intermediate values.
        # The important point is to intercept the call so that the evaluator
        # does not create any new resolvers on the fly.  Storing the value
        # is not important, since that will have been done already in
        # the NestedFrame.
        pass


class _NestedFieldResolver:
    """
    Used by NestedFrame.eval to resolve the names of fields in nested columns when
    encountered in expressions, interpreting __getattr__ in terms of a
    specific nest.
    """

    def __init__(self, nest_name: str, outer: NestedFrame):
        self._nest_name = nest_name
        # Save the outer frame with an eye toward repacking.
        self._outer = outer
        # Flattened only once for every access of this particular nest
        # within the expression.
        self._flat_nest = outer[nest_name].nest.to_flat()

    def __getattr__(self, item_name: str):
        if item_name in self._flat_nest:
            result = _SeriesFromNest(self._flat_nest[item_name])
            # Assigning these properties directly in order to avoid any complication
            # or interference with the inherited pd.Series constructor.
            result.nest_name = self._nest_name
            result.flat_nest = self._flat_nest
            return result
        raise AttributeError(f"No attribute {item_name}")


class NestedFrame(pd.DataFrame):
    """A Pandas Dataframe extension with support for nested structure.

    See https://pandas.pydata.org/docs/development/extending.html#subclassing-pandas-data-structures
    """

    # https://pandas.pydata.org/docs/development/extending.html#arithmetic-with-3rd-party-types
    # The __pandas_priority__ of DataFrame is 4000, so give NestedFrame a
    # higher priority, so that binary operations involving this class and
    # Series produce instances of this class, preserving the type and origin.
    __pandas_priority__ = 4500

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
            if isinstance(self.dtypes[column], NestedDtype):
                nest_cols = self[column].nest.fields
                all_columns[column] = nest_cols
        return all_columns

    @property
    def nested_columns(self) -> list:
        """retrieves the base column names for all nested dataframes"""
        nest_cols = []
        for column in self.columns:
            if isinstance(self.dtypes[column], NestedDtype):
                nest_cols.append(column)
        return nest_cols

    def _is_known_hierarchical_column(self, colname) -> bool:
        """Determine whether a string is a known hierarchical column name"""
        if "." in colname:
            base_name = colname.split(".")[0]
            if base_name in self.nested_columns:
                # TODO: only handles one level of nesting for now
                nested_name = ".".join(colname.split(".")[1:])
                return nested_name in self.all_columns[base_name]
            return False
        return False

    def _is_known_column(self, colname) -> bool:
        """Determine whether a string is a known column name"""
        return colname in self.columns or self._is_known_hierarchical_column(colname)

    def __getitem__(self, item):
        """Adds custom __getitem__ behavior for nested columns"""

        if isinstance(item, str):
            # Preempt the nested check if the item is a base column
            if item in self.columns:
                return super().__getitem__(item)
            # If a nested column name is passed, return a flat series for that column
            # flat series is chosen over list series for utility
            # e.g. native ability to do something like ndf["nested.a"] + 3
            elif self._is_known_hierarchical_column(item):
                # TODO: only handles one level of nesting for now
                nested = item.split(".")[0]
                col = ".".join(item.split(".")[1:])
                return self[nested].nest.get_flat_series(col)
        else:
            return super().__getitem__(item)

    def __setitem__(self, key, value):
        """Adds custom __setitem__ behavior for nested columns"""

        # Replacing or adding columns to a nested structure
        # Allows statements like ndf["nested.t"] = ndf["nested.t"] - 5
        # Or ndf["nested.base_t"] = ndf["nested.t"] - 5
        # Performance note: This requires building a new nested structure
        # TODO: Support assignment of a new column to an existing nested col from a list series
        if self._is_known_hierarchical_column(key) or (
            "." in key and key.split(".")[0] in self.nested_columns
        ):
            nested, col = key.split(".")
            new_flat = self[nested].nest.to_flat()
            new_flat[col] = value
            packed = packer.pack(new_flat)
            return super().__setitem__(nested, packed)

        # Adding a new nested structure from a column
        # Allows statements like ndf["new_nested.t"] = ndf["nested.t"] - 5
        if "." in key:
            new_nested, col = key.split(".")
            if isinstance(value, pd.Series):
                value.name = col
                value = value.to_frame()
            packed = packer.pack(value)
            return super().__setitem__(new_nested, packed)

        return super().__setitem__(key, value)

    def add_nested(
        self,
        obj,
        name: str,
        *,
        how: str = "left",
        dtype: NestedDtype | pd.ArrowDtype | pa.DataType | None = None,
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
        how : {'left', 'right', 'outer', 'inner'}, default: 'left'
            How to handle the operation of the two objects:

            - left: use calling frame's index.
            - right: use the calling frame's index and order but drop values
              not in the other frame's index.
            - outer: form union of calling frame's index with other frame's
              index, and sort it lexicographically.
            - inner: form intersection of calling frame's index with other
              frame's index, preserving the order of the calling index.
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
        new_df = self.copy()
        return new_df.join(packed, how=how)

    @classmethod
    def from_flat(cls, df, base_columns, nested_columns=None, index=None, name="nested"):
        """Creates a NestedFrame with base and nested columns from a flat
        dataframe.

        Parameters
        ----------
        df: pd.DataFrame or NestedFrame
            A flat dataframe.
        base_columns: list-like
            The columns that should be used as base (flat) columns in the
            output dataframe.
        nested_columns: list-like, or None
            The columns that should be packed into a nested column. All columns
            in the list will attempt to be packed into a single nested column
            with the name provided in `nested_name`. If None, is defined as all
            columns not in `base_columns`.
        index: str, or None
            The name of a column to use as the new index. Typically, the index
            should have a unique value per row for base columns, and should
            repeat for nested columns. For example, a dataframe with two
            columns; a=[1,1,1,2,2,2] and b=[5,10,15,20,25,30] would want an
            index like [0,0,0,1,1,1] if a is chosen as a base column. If not
            provided the current index will be used.
        name:
            The name of the output column the `nested_columns` are packed into.

        Returns
        -------
        NestedFrame
            A NestedFrame with the specified nesting structure.

        Examples
        --------

        >>> nf = NestedFrame({"a":[1,1,1,2,2], "b":[2,2,2,4,4],
        ...                   "c":[1,2,3,4,5], "d":[2,4,6,8,10]},
        ...                   index=[0,0,0,1,1])

        >>> NestedFrame.from_flat(nf, base_columns=["a","b"])
        """

        # Resolve new index
        if index is not None:
            # if a base column is chosen remove it
            if index in base_columns:
                base_columns = [col for col in base_columns if col != index]
            df = df.set_index(index)

        # drop duplicates on index
        out_df = df[base_columns][~df.index.duplicated(keep="first")]

        # Convert df to NestedFrame if needed
        if not isinstance(out_df, NestedFrame):
            out_df = NestedFrame(out_df)

        # add nested
        if nested_columns is None:
            nested_columns = [col for col in df.columns if col not in base_columns]
        return out_df.add_nested(df[nested_columns], name=name)

    @classmethod
    def from_lists(cls, df, base_columns=None, list_columns=None, name="nested"):
        """Creates a NestedFrame with base and nested columns from a flat
        dataframe.

        Parameters
        ----------
        df: pd.DataFrame or NestedFrame
            A dataframe with list columns.
        base_columns: list-like, or None
            Any columns that have non-list values in the input df. These will
            simply be kept as identical columns in the result
        list_columns: list-like, or None
            The list-value columns that should be packed into a nested column.
            All columns in the list will attempt to be packed into a single
            nested column with the name provided in `nested_name`. If None, is
            defined as all columns not in `base_columns`.
        name:
            The name of the output column the `nested_columns` are packed into.

        Returns
        -------
        NestedFrame
            A NestedFrame with the specified nesting structure.

         Examples
        --------

        >>> nf = NestedFrame({"c":[1,2,3], "d":[2,4,6],
        ...                   "e":[[1,2,3], [4,5,6], [7,8,9]]},
        ...                  index=[0,1,2])


        >>> NestedFrame.from_lists(nf, base_columns=["c","d"])
        """

        # Resolve base and list columns
        if base_columns is None:
            if list_columns is None:
                # with no inputs, assume all columns are list-valued
                list_columns = df.columns
            else:
                # if list_columns are defined, assume everything else is base
                base_columns = [col for col in df.columns if col not in list_columns]
        else:
            if list_columns is None:
                # with defined base_columns, assume everything else is list
                list_columns = [col for col in df.columns if col not in base_columns]

        if len(list_columns) == 0:
            raise ValueError("No columns were assigned as list columns.")

        # Pack list columns into a nested column
        packed_df = packer.pack_lists(df[list_columns])
        packed_df.name = name

        # join the nested column to the base_column df
        if base_columns is not None:
            return df[base_columns].join(packed_df)
        # or just return the packed_df as a nestedframe if no base cols
        else:
            return NestedFrame(packed_df.to_frame())

    def eval(self, expr: str, *, inplace: bool = False, **kwargs) -> Any | None:
        """

        Evaluate a string describing operations on NestedFrame columns.

        Operates on columns only, not specific rows or elements.  This allows
        `eval` to run arbitrary code, which can make you vulnerable to code
        injection if you pass user input to this function.

        Works the same way as `pd.DataFrame.eval`, except that this method
        will also automatically unpack nested columns into NestedSeries,
        and the resulting expression will have the dimensions of the unpacked
        series.

        Parameters
        ----------
        expr : str
            The expression string to evaluate.
        inplace : bool, default False
            If the expression contains an assignment, whether to perform the
            operation inplace and mutate the existing NestedFrame. Otherwise,
            a new NestedFrame is returned.
        **kwargs
            See the documentation for :func:`eval` for complete details
            on the keyword arguments accepted by
            :meth:`~pandas.NestedFrame.eval`.

        Returns
        -------
        ndarray, scalar, pandas object, nested-pandas object, or None
            The result of the evaluation or None if ``inplace=True``.

        See Also
        --------
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
        """
        kwargs["resolvers"] = tuple(kwargs.get("resolvers", ())) + (_NestResolver(self),)
        kwargs["inplace"] = inplace
        kwargs["parser"] = "nested-pandas"
        return super().eval(expr, **kwargs)

    def query(self, expr: str, *, inplace: bool = False, **kwargs) -> NestedFrame | None:
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

        inplace : bool
            Whether to modify the DataFrame rather than creating a new one.
        **kwargs
            See the documentation for :func:`eval` for complete details
            on the keyword arguments accepted by :meth:`DataFrame.query`.

        Returns
        -------
        NestedFrame
            NestedFrame resulting from the provided query expression.

        Notes
        -----
        Queries that target a particular nested structure return a dataframe
        with rows of that particular nested structure filtered. For example,
        querying the NestedFrame "df" with nested structure "my_nested" as
        below will return all rows of df, but with mynested filtered by the
        condition:

        >>> df.query("mynested.a > 2")
        """
        if not isinstance(expr, str):
            msg = f"expr must be a string to be evaluated, {type(expr)} given"
            raise ValueError(msg)
        kwargs["level"] = kwargs.pop("level", 0) + 1
        kwargs["target"] = None
        # At present, the query expression must be either entirely within the
        # nested namespace or the base namespace.  Mixed structures are not
        # supported, so preflight the expression.
        nesting_types = check_expr_nesting(expr)
        if NestingType.NESTED in nesting_types and NestingType.BASE in nesting_types:
            raise ValueError("Queries cannot target multiple structs/layers, write a separate query for each")
        result = self.eval(expr, **kwargs)
        # If the result is a _SeriesFromNest, then the evaluation has caused unpacking,
        # which means that a nested attribute was referenced.  Apply this result
        # to the nest and repack.  Otherwise, apply it to this instance as usual,
        # since it operated on the base attributes.
        if isinstance(result, _SeriesFromNest):
            nest_name, flat_nest = result.nest_name, result.flat_nest
            new_flat_nest = flat_nest.loc[result]
            result = self.copy()
            result[nest_name] = pack_sorted_df_into_struct(new_flat_nest)
        else:
            result = self.loc[result]

        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result

    def _resolve_dropna_target(self, on_nested, subset):
        """resolves the target layer for a given set of dropna kwargs"""

        nested_cols = self.nested_columns

        # first check the subset kwarg input
        subset_target = []
        if subset:
            if isinstance(subset, str):
                subset = [subset]

            for col in subset:
                # Without a ".", always assume base layer
                if "." not in col:
                    subset_target.append("base")
                else:
                    layer, col = col.split(".")
                    if layer in nested_cols:
                        subset_target.append(layer)
                    else:
                        raise ValueError(f"layer '{layer}' not found in the base columns")

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

    def reduce(self, func, *args, **kwargs) -> NestedFrame:  # type: ignore[override]
        """
        Takes a function and applies it to each top-level row of the NestedFrame.

        The user may specify which columns the function is applied to, with
        columns from the 'base' layer being passsed to the function as
        scalars and columns from the nested layers being passed as numpy arrays.

        Parameters
        ----------
        func : callable
            Function to apply to each nested dataframe. The first arguments to `func` should be which
            columns to apply the function to. See the Notes for recommendations
            on writing func outputs.
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
        By default, `reduce` will produce a `NestedFrame` with enumerated
        column names for each returned value of the function. For more useful
        naming, it's recommended to have `func` return a dictionary where each
        key is an output column of the dataframe returned by `reduce`.

        Example User Function:

        >>> def my_sum(col1, col2):
        >>>    '''reduce will return a NestedFrame with two columns'''
        >>>    return {"sum_col1": sum(col1), "sum_col2": sum(col2)}

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

        iterators = []
        for layer, col in requested_columns:
            if layer == "base":
                iterators.append(self[col])
            else:
                iterators.append(self[layer].array.iter_field_lists(col))

        results = [func(*cols, *extra_args, **kwargs) for cols in zip(*iterators)]
        return NestedFrame(results, index=self.index)

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
