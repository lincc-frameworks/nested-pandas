# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas._libs import lib
from pandas._typing import Any, AnyAll, Axis, IndexLabel
from pandas.api.extensions import no_default
from pandas.core.computation.eval import Expr, ensure_scope

from nested_pandas.nestedframe.expr import (
    _identify_aliases,
    _NestResolver,
    _SeriesFromNest,
    _subexprs_by_nest,
)
from nested_pandas.series.dtype import NestedDtype
from nested_pandas.series.packer import pack, pack_lists, pack_sorted_df_into_struct

pd.set_option("display.max_rows", 30)
pd.set_option("display.min_rows", 5)


class NestedFrame(pd.DataFrame):
    """A Pandas Dataframe extension with support for nested structure.

    See https://pandas.pydata.org/docs/development/extending.html#subclassing-pandas-data-structures
    """

    # https://pandas.pydata.org/docs/development/extending.html#arithmetic-with-3rd-party-types
    # The __pandas_priority__ of DataFrame is 4000, so give NestedFrame a
    # higher priority, so that binary operations involving this class and
    # Series produce instances of this class, preserving the type and origin.
    __pandas_priority__ = 4500

    # The "_aliases" attribute is usually None or not even present, but when it is present,
    # it indicates that an evaluation is in progress, and that columns and fields with names
    # that are not identifier-like have been aliases to cleaned names, and this attribute
    # contains those aliases, keyed by the cleaned name.
    _metadata = ["_aliases"]

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

    def _repr_html_(self) -> str | None:
        """Override html representation"""

        # Without nested columns (or empty), just do representation as normal
        if len(self.nested_columns) == 0 or len(self) == 0:
            # This mimics pandas behavior
            if self.shape[0] > pd.get_option("display.max_rows"):
                return super().to_html(max_rows=pd.get_option("display.min_rows"), show_dimensions=True)
            else:
                return super().to_html(max_rows=pd.get_option("display.max_rows"), show_dimensions=True)

        # Nested Column Formatting
        # first cell shows the nested df header and a preview row
        def repack_first_cell(chunk):
            # If the chunk is None, just return None
            # Means that the column labels will not be shown
            if chunk is None:
                return None
            # Render header separately to keep data aligned
            output = chunk.head(0).to_html(
                max_rows=0, max_cols=5, show_dimensions=False, index=False, header=True
            )
            # Then add a preview row
            output += repack_row(chunk)
            return output

        # remaining cells show only a preview row
        def repack_row(chunk):
            # If the chunk is None, just return None
            if chunk is None:
                return None
            return chunk.to_html(max_rows=1, max_cols=5, show_dimensions=True, index=False, header=False)

        # replace index to ensure proper behavior for duplicate index values
        index_values = self.index
        html_df = self.reset_index(drop=True)

        # Apply repacking to all nested columns
        repr = html_df.style.format(
            {col: repack_first_cell for col in self.nested_columns}, subset=html_df.index[0]
        )
        if len(self) > 1:
            repr = repr.format(
                {col: repack_row for col in self.nested_columns}, subset=pd.IndexSlice[html_df.index[1] :]
            )

        # Create a mapping function to retrieve original index
        def map_true_index(index):
            return index_values[index]

        repr = repr.format_index(map_true_index, axis=0)

        # Recover some truncation formatting, limited to head truncation
        if pd.get_option("display.max_rows") is None:
            return repr.to_html(max_rows=0)
        elif repr.data.shape[0] > pd.get_option("display.max_rows"):
            html_repr = repr.to_html(max_rows=pd.get_option("display.min_rows"))
        else:
            # when under the max_rows threshold, display all rows (behavior of 0 here)
            html_repr = repr.to_html(max_rows=0)

        # Manually append dimensionality to a styler output
        html_repr += f"{repr.data.shape[0]} rows x {repr.data.shape[1]} columns"

        return html_repr

    def _parse_hierarchical_components(self, delimited_path: str, delimiter: str = ".") -> list[str]:
        """
        Given a string that may be a delimited path, parse it into its components,
        respecting backticks that are used to protect component names that may contain the delimiter.
        """
        aliases = getattr(self, "_aliases", None)
        if aliases is None:
            delimited_path, aliases = _identify_aliases(delimited_path)
        return [aliases.get(x, x) for x in delimited_path.split(delimiter)]

    def _is_known_hierarchical_column(self, components: list[str] | str) -> bool:
        """Determine whether a string is a known hierarchical column name"""
        if isinstance(components, str):
            components = self._parse_hierarchical_components(components)
        if len(components) < 2:
            return False
        base_name = components[0]
        if base_name in self.nested_columns:
            nested_name = ".".join(components[1:])
            return nested_name in self.all_columns[base_name]
        return False

    def _is_known_column(self, components: list[str] | str) -> bool:
        """Determine whether a list of field components describes a known column name"""
        if isinstance(components, str):
            components = self._parse_hierarchical_components(components)
        if ".".join(components) in self.columns:
            return True
        return self._is_known_hierarchical_column(components)

    def __getitem__(self, item):
        """Adds custom __getitem__ behavior for nested columns"""

        if not isinstance(item, str):
            return super().__getitem__(item)

        # Preempt the nested check if the item is a base column, with or without
        # dots and backticks.
        if item in self.columns:
            return super().__getitem__(item)
        components = self._parse_hierarchical_components(item)
        # One more check on the entirety of the item name, in case backticks were used
        # (even if they weren't necessary).
        cleaned_item = ".".join(components)
        if cleaned_item in self.columns:
            return super().__getitem__(cleaned_item)

        # If a nested column name is passed, return a flat series for that column
        # flat series is chosen over list series for utility
        # e.g. native ability to do something like ndf["nested.a"] + 3
        if self._is_known_hierarchical_column(components):
            nested = components[0]
            field = ".".join(components[1:])
            return self[nested].nest.get_flat_series(field)
        else:
            raise KeyError(f"Column '{cleaned_item}' not found in nested columns or base columns")

    def __setitem__(self, key, value):
        """Adds custom __setitem__ behavior for nested columns"""
        components = self._parse_hierarchical_components(key)
        # Replacing or adding columns to a nested structure
        # Allows statements like ndf["nested.t"] = ndf["nested.t"] - 5
        # Or ndf["nested.base_t"] = ndf["nested.t"] - 5
        # Performance note: This requires building a new nested structure
        # TODO: Support assignment of a new column to an existing nested col from a list series
        if self._is_known_hierarchical_column(components) or (
            len(components) > 1 and components[0] in self.nested_columns
        ):
            if len(components) != 2:
                raise ValueError(f"Only one level of nesting is supported; given {key}")
            nested, field = components
            # Support a special case of embedding a base column into a nested column, with values being
            # repeated in each nested list-array.
            if isinstance(value, pd.Series) and self.index.equals(value.index):
                new_nested_series = self[nested].nest.with_filled_field(field, value)
            else:
                new_nested_series = self[nested].nest.with_flat_field(field, value)
            return super().__setitem__(nested, new_nested_series)

        # Adding a new nested structure from a column
        # Allows statements like ndf["new_nested.t"] = ndf["nested.t"] - 5
        if len(components) > 1:
            new_nested, field = components
            if isinstance(value, pd.Series):
                value.name = field
                value = value.to_frame()
            new_df = self.add_nested(value, name=new_nested)
            self._update_inplace(new_df)
            return None

        return super().__setitem__(key, value)

    def add_nested(
        self,
        obj,
        name: str,
        *,
        how: str = "left",
        on: None | str | list[str] = None,
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
        on : str, default: None
            A column in the list
        dtype : dtype or None
            NestedDtype to use for the nested column; pd.ArrowDtype or
            pa.DataType can also be used to specify the nested dtype. If None,
            the dtype is inferred from the input object.

        Returns
        -------
        NestedFrame
            A new NestedFrame with the added nested column.

        Examples
        --------


        >>> nf = npd.NestedFrame({"a": [1, 2, 3], "b": [4, 5, 6]},
        ...            index=[0,1,2])
        >>> nf2 = npd.NestedFrame({"c":[1,2,3,4,5,6,7,8,9]},
        ...             index=[0,0,0,1,1,1,2,2,2])
        >>> # By default, aligns on the index
        >>> nf.add_nested(nf2, "nested")
           a  b                nested
        0  1  4  [{c: 1}; …] (3 rows)
        1  2  5  [{c: 4}; …] (3 rows)
        2  3  6  [{c: 7}; …] (3 rows)
        """
        if on is not None and not isinstance(on, str):
            raise ValueError("Currently we only support a single column for 'on'")
        # Add sources to objects
        packed = pack(obj, name=name, on=on, dtype=dtype)
        new_df = self.copy()
        res = new_df.join(packed, how=how, on=on)
        return res

    @classmethod
    def from_flat(cls, df, base_columns, nested_columns=None, on: str | None = None, name="nested"):
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
        on: str or None
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

        >>> nf = npd.NestedFrame({"a":[1,1,1,2,2], "b":[2,2,2,4,4],
        ...                   "c":[1,2,3,4,5], "d":[2,4,6,8,10]},
        ...                   index=[0,0,0,1,1])

        >>> npd.NestedFrame.from_flat(nf, base_columns=["a","b"])
           a  b                      nested
        0  1  2  [{c: 1, d: 2}; …] (3 rows)
        1  2  4  [{c: 4, d: 8}; …] (2 rows)
        """

        # Resolve new index
        if on is not None:
            # if a base column is chosen remove it
            if on in base_columns:
                base_columns = [col for col in base_columns if col != on]
            df = df.set_index(on)

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

        >>> nf = npd.NestedFrame({"c":[1,2,3], "d":[2,4,6],
        ...                   "e":[[1,2,3], [4,5,6], [7,8,9]]},
        ...                   index=[0,1,2])

        >>> npd.NestedFrame.from_lists(nf, base_columns=["c","d"])
           c  d                nested
        0  1  2  [{e: 1}; …] (3 rows)
        1  2  4  [{e: 4}; …] (3 rows)
        2  3  6  [{e: 7}; …] (3 rows)

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
        packed_df = pack_lists(df[list_columns])
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
        _, aliases = _identify_aliases(expr)
        self._aliases: dict[str, str] | None = aliases

        kwargs["resolvers"] = tuple(kwargs.get("resolvers", ())) + (_NestResolver(self),)
        kwargs["inplace"] = inplace
        kwargs["parser"] = "nested-pandas"
        answer = super().eval(expr, **kwargs)
        self._aliases = None
        return answer

    def extract_nest_names(
        self,
        expr: str,
        local_dict=None,
        global_dict=None,
        resolvers=(),
        level: int = 0,
        target=None,
        **kwargs,
    ) -> set[str]:
        """
        Given a string expression, parse it and visit the resulting expression tree,
        surfacing the nesting types.  The purpose is to identify expressions that attempt
        to mix base and nested columns, or columns from two different nests.
        """
        index_resolvers = self._get_index_resolvers()
        column_resolvers = self._get_cleaned_column_resolvers()
        resolvers = resolvers + (_NestResolver(self), column_resolvers, index_resolvers)
        # Parser needs to be the "nested-pandas" parser.
        # We also need the same variable context that eval() will have, so that
        # backtick-quoted names are substituted as expected.
        env = ensure_scope(
            level + 1,
            global_dict=global_dict,
            local_dict=local_dict,
            resolvers=resolvers,
            target=target,
        )
        parsed_expr = Expr(expr, parser="nested-pandas", env=env)
        expr_tree = parsed_expr.terms
        separable = _subexprs_by_nest([], expr_tree)
        return set(separable.keys())

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

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5,5, seed=1)

        >>> nf = nf.query("nested.t > 10")
        >>> nf
           a         b                                             nested
        0  0.417022  0.184677  [{t: 13.40935, flux: 98.886109, band: 'g'}; …]...
        1  0.720324  0.372520  [{t: 13.70439, flux: 68.650093, band: 'g'}; …]...
        2  0.000114  0.691121  [{t: 11.173797, flux: 28.044399, band: 'r'}; …...
        3  0.302333  0.793535  [{t: 17.562349, flux: 1.828828, band: 'g'}; …]...
        4  0.146756  1.077633  [{t: 17.527783, flux: 13.002857, band: 'r'}; …...


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
        # At present, the query expression must be either entirely within a
        # single nest, or have nothing but base columns.  Mixed structures are not
        # supported, so preflight the expression.
        nest_names = self.extract_nest_names(expr, **kwargs)
        if len(nest_names) > 1:
            raise ValueError("Queries cannot target multiple structs/layers, write a separate query for each")
        result = self.eval(expr, **kwargs)
        # If the result is a _SeriesFromNest, then the evaluation has caused unpacking,
        # which means that a nested attribute was referenced.  Apply this result
        # to the nest and repack.  Otherwise, apply it to this instance as usual,
        # since it operated on the base attributes.
        if isinstance(result, _SeriesFromNest):
            nest_name, flat_nest = result.nest_name, result.flat_nest
            # Reset index to "ordinal" like [0, 0, 0, 1, 1, 2, 2, 2]
            list_index = self[nest_name].array.get_list_index()
            flat_nest = flat_nest.set_index(list_index)
            query_result = result.set_axis(list_index)
            # Selecting flat values matching the query result
            new_flat_nest = flat_nest[query_result]
            new_df = self._set_filtered_flat_df(nest_name, new_flat_nest)
        else:
            new_df = self.loc[result]

        if inplace:
            self._update_inplace(new_df)
            return None
        else:
            return new_df

    def _set_filtered_flat_df(self, nest_name, flat_df):
        """Set a filtered flat dataframe for a nested column

        Here we assume that flat_df has filtered "ordinal" index,
        e.g. flat_df.index == [0, 2, 2, 2], while self.index
        is arbitrary (e.g. ["a", "b", "a"]),
        and self[nest_name].array.list_index is [0, 0, 1, 1, 1, 2, 2, 2, 2].
        """
        new_df = self.reset_index(drop=True)
        new_df[nest_name] = pack_sorted_df_into_struct(flat_df, name=nest_name)
        return new_df.set_index(self.index)

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

        Examples
        --------

        A common usecase for `dropna` is to remove empty nested rows:

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5,5, seed=1)

        >>> # this query empties several of the nested dataframes
        >>> nf = nf.query("nested.t > 19")
        >>> nf
            a         b                                        nested
        0  0.417022  0.184677                                          None
        1  0.720324  0.372520   [{t: 19.365232, flux: 90.85955, band: 'r'}]
        2  0.000114  0.691121  [{t: 19.157791, flux: 14.672857, band: 'r'}]
        3  0.302333  0.793535                                          None
        4  0.146756  1.077633                                          None


        >>> # dropna removes rows with those emptied dataframes
        >>> nf.dropna(subset="nested")
            a         b                                        nested
        1  0.720324  0.372520   [{t: 19.365232, flux: 90.85955, band: 'r'}]
        2  0.000114  0.691121  [{t: 19.157791, flux: 14.672857, band: 'r'}]


        `dropna` can also be used on nested columns:

        >>> nf = generate_data(5,5, seed=1)
        >>> # Either on the whole dataframe
        >>> nf.dropna(on_nested="nested")
        >>> # or on a specific nested column
        >>> nf.dropna(subset="nested.t")


        Notes
        -----
        Operations that target a particular nested structure return a dataframe
        with rows of that particular nested structure affected.

        Values for `on_nested` and `subset` should be consistent in pointing
        to a single layer, multi-layer operations are not supported.
        """

        # determine target dataframe
        target, subset = self._resolve_dropna_target(on_nested, subset)

        if target == "base":
            return super().dropna(
                axis=axis, how=how, thresh=thresh, subset=subset, inplace=inplace, ignore_index=ignore_index
            )
        if ignore_index:
            raise ValueError("ignore_index is not supported for nested columns")
        if subset is not None:
            subset = [col.split(".")[-1] for col in subset]
        target_flat = self[target].nest.to_flat()
        target_flat = target_flat.set_index(self[target].array.get_list_index())
        if inplace:
            target_flat.dropna(
                axis=axis,
                how=how,
                thresh=thresh,
                subset=subset,
                inplace=True,
            )
        else:
            target_flat = target_flat.dropna(
                axis=axis,
                how=how,
                thresh=thresh,
                subset=subset,
                inplace=False,
            )
        new_df = self._set_filtered_flat_df(nest_name=target, flat_df=target_flat)
        if inplace:
            self._update_inplace(new_df)
            return None
        return new_df

    def sort_values(
        self,
        by,
        *,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index=False,
        key=None,
    ):
        """
        Sort by the values along either axis.

        Parameters
        ----------
        by : str or list of str
            Name or list of names to sort by.

            Access nested columns using `nested_df.nested_col` (where
            `nested_df` refers to a particular nested dataframe and
            `nested_col` is a column of that nested dataframe).
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Axis to be sorted.
        ascending : bool or list of bool, default True
            Sort ascending vs. descending. Specify list for multiple sort
            orders. If this is a list of bools, must match the length of the
            by.
        inplace : bool, default False
            If True, perform operation in-place.
        kind : {'quicksort', 'mergesort', 'heapsort'}, default 'quicksort'
            Choice of sorting algorithm. See also ndarray.np.sort for more
            information. mergesort is the only stable algorithm. For DataFrames,
            this option is only applied when sorting on a single column or label.
        na_position : {'first', 'last'}, default 'last'
            Puts NaNs at the beginning if first; last puts NaNs at the end.
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Always False when applied to nested layers.
        key : callable, optional
            Apply the key function to the values before sorting.

        Returns
        -------
        DataFrame or None
            DataFrame with sorted values if inplace=False, None otherwise.

        Examples
        ---------
        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5,5, seed=1)

        >>> # Sort nested values
        >>> nf.sort_values(by="nested.band")
           a         b                                             nested
        0  0.417022  0.184677  [{t: 13.40935, flux: 98.886109, band: 'g'}; …]...
        1  0.720324  0.372520  [{t: 13.70439, flux: 68.650093, band: 'g'}; …]...
        2  0.000114  0.691121  [{t: 4.089045, flux: 83.462567, band: 'g'}; …]...
        3  0.302333  0.793535  [{t: 17.562349, flux: 1.828828, band: 'g'}; …]...
        4  0.146756  1.077633  [{t: 0.547752, flux: 75.014431, band: 'g'}; …]...
        """

        # Resolve target layer
        target = []
        if isinstance(by, str):
            by = [by]
        # Check "by" columns for hierarchical references
        for col in by:
            if self._is_known_hierarchical_column(col):
                target.append(col.split(".")[0])
            else:
                target.append("base")

        # Ensure one target layer, preventing multi-layer operations
        target = np.unique(target)
        if len(target) > 1:
            raise ValueError("Queries cannot target multiple structs/layers, write a separate query for each")
        target = str(target[0])

        # Apply pandas sort_values
        if target == "base":
            return super().sort_values(
                by=by,
                axis=axis,
                ascending=ascending,
                inplace=inplace,
                kind=kind,
                na_position=na_position,
                ignore_index=ignore_index,
                key=key,
            )
        else:  # target is a nested column
            target_flat = self[target].nest.to_flat()
            target_flat = target_flat.set_index(self[target].array.get_list_index())

            if target_flat.index.name is None:  # set name if not present
                target_flat.index.name = "index"
            # Index must always be the first sort key for nested columns
            nested_by = [target_flat.index.name] + [col.split(".")[-1] for col in by]

            # Augment the ascending kwarg to include the index
            if isinstance(ascending, bool):
                ascending = [True] + [ascending] * len(by)
            elif isinstance(ascending, list):
                ascending = [True] + ascending

            target_flat = target_flat.sort_values(
                by=nested_by,
                axis=axis,
                ascending=ascending,
                kind=kind,
                na_position=na_position,
                ignore_index=False,
                key=key,
                inplace=False,
            )

            #  Could be optimized, as number of rows doesn't change
            new_df = self._set_filtered_flat_df(nest_name=target, flat_df=target_flat)

            if inplace:
                self._update_inplace(new_df)
                return None
            return new_df

    def reduce(self, func, *args, infer_nesting=True, **kwargs) -> NestedFrame:  # type: ignore[override]
        """
        Takes a function and applies it to each top-level row of the NestedFrame.

        The user may specify which columns the function is applied to, with
        columns from the 'base' layer being passed to the function as
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
        infer_nesting : bool, default True
            If True, the function will pack output columns into nested
            structures based on column names adhering to a nested naming
            scheme. E.g. "nested.b" and "nested.c" will be packed into a column
            called "nested" with columns "b" and "c". If False, all outputs
            will be returned as base columns.
        kwargs : keyword arguments, optional
            Keyword arguments to pass to the function.

        Returns
        -------
        `NestedFrame`
            `NestedFrame` with the results of the function applied to the columns of the frame.

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> import numpy as np
        >>> nf = generate_data(5,5, seed=1)

        >>> # define a custom user function
        >>> def example_func(base_col, nested_col):
        >>>    '''reduce will return a NestedFrame with two columns'''
        >>>    return {"mean": np.mean(nested_col),
        ...            "mean_minus_base": np.mean(nested_col) - base_col}

        >>> # apply the function
        >>> nf.reduce(example_func, "a", "nested.t")
                mean  mean_minus_base
        0  11.533440        11.116418
        1  10.307751         9.587426
        2   8.294042         8.293928
        3   9.655291         9.352958
        4  10.687591        10.540836

        You may want the result of a `reduce` call to have nested structure,
        we can achieve this by using the `infer_nesting` kwarg:

        >>> # define a custom user function that returns nested structure
        >>> def example_func(base_col1, base_col2, nested_col):
        >>>    '''reduce will return a NestedFrame with nested structure'''
        >>>    return {"offsets.t_a": nested_col - base_col1,
        ...            "offsets.t_b": nested_col - base_col2}

        By giving both output columns the prefix "offsets.", we signal
        to reduce to infer that these should be packed into a nested column
        called "offsets".

        >>> # apply the function with `infer_nesting` (True by default)
        >>> nf.reduce(example_func, "a", "b", "nested.t")
                                                  offsets
        0    [{t_a: 7.966868, t_b: 8.199213}; …] (5 rows)
        1   [{t_a: 12.984066, t_b: 13.33187}; …] (5 rows)
        2    [{t_a: 4.088931, t_b: 3.397924}; …] (5 rows)
        3  [{t_a: 17.260016, t_b: 16.768814}; …] (5 rows)
        4   [{t_a: 0.400996, t_b: -0.529882}; …] (5 rows)

        Notes
        -----
        By default, `reduce` will produce a `NestedFrame` with enumerated
        column names for each returned value of the function. For more useful
        naming, it's recommended to have `func` return a dictionary where each
        key is an output column of the dataframe returned by `reduce` (as
        shown above).
        """
        # Parse through the initial args to determine the columns to apply the function to
        requested_columns = []
        for arg in args:
            # Stop when we reach an argument that is not a valid column, as we assume
            # that the remaining args are extra arguments to the function
            if not isinstance(arg, str):
                break
            components = self._parse_hierarchical_components(arg)
            if not self._is_known_column(components):
                break
            layer = "base" if len(components) < 2 else components[0]
            col = components[-1]
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
        results_nf = NestedFrame(results, index=self.index)

        if infer_nesting:
            # find potential nested structures from columns
            nested_cols = list(
                np.unique(
                    [
                        column.split(".", 1)[0]
                        for column in results_nf.columns
                        if isinstance(column, str) and "." in column
                    ]
                )
            )

            # pack results into nested structures
            for layer in nested_cols:
                layer_cols = [col for col in results_nf.columns if col.startswith(f"{layer}.")]
                rename_df = results_nf[layer_cols].rename(columns=lambda x: x.split(".", 1)[1])
                nested_col = pack_lists(rename_df, name=layer)
                results_nf = results_nf[
                    [col for col in results_nf.columns if not col.startswith(f"{layer}.")]
                ].join(nested_col)

        return results_nf

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
