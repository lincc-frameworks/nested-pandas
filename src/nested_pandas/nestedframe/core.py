# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas._libs import lib
from pandas._typing import Any, AnyAll, Axis, IndexLabel
from pandas.api.extensions import no_default
from pandas.core.computation.eval import Expr, ensure_scope
from pandas.core.dtypes.inference import is_list_like

from nested_pandas.nestedframe.expr import (
    _identify_aliases,
    _NestResolver,
    _SeriesFromNest,
    _subexprs_by_nest,
)
from nested_pandas.series.dtype import NestedDtype
from nested_pandas.series.packer import pack, pack_lists, pack_sorted_df_into_struct
from nested_pandas.series.utils import is_pa_type_a_list

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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cast_cols_to_nested(struct_list=False)

    def _cast_cols_to_nested(self, *, struct_list: bool) -> None:
        """Cast arrow columns to nested.

        Parameters
        ----------
        struct_list : bool
            If `False` cast list-struct columns only. If `True`, also
            try to cast struct-list columns validating if they have
            valid nested structure.
        """
        for column, dtype in self.dtypes.items():
            if not isinstance(dtype, pd.ArrowDtype):
                continue
            pa_type = dtype.pyarrow_dtype
            if not is_pa_type_a_list(pa_type) and not (struct_list and pa.types.is_struct(pa_type)):
                continue
            try:
                nested_dtype = NestedDtype(pa_type)
            except (TypeError, ValueError):
                continue
            self[column] = self[column].astype(nested_dtype)

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
            if pd.get_option("display.max_rows") is None:
                # If max_rows is None, just show the header
                return super().to_html(max_rows=None, show_dimensions=True)
            if self.shape[0] > pd.get_option("display.max_rows"):
                return super().to_html(max_rows=pd.get_option("display.min_rows"), show_dimensions=True)
            else:
                return super().to_html(max_rows=pd.get_option("display.max_rows"), show_dimensions=True)

        # Nested Column Formatting

        # Display nested columns as small html dataframes with a single row
        def repack_row(chunk, header=True):
            # If the chunk is None, just return None
            if chunk is None:
                return None
            # Grab length, then truncate to one row for display
            n_rows = len(chunk)
            chunk = chunk.head(1).round(8)  # only show first row
            chunk.astype({col: object for col in chunk.columns})  # cast to string for info row

            # Add a row that shows the number of additional rows not shown
            len_row = pd.DataFrame(
                {
                    col: [f"<i>+{n_rows-1} rows</i>"] if i == 0 else ["..."]
                    for i, col in enumerate(chunk.columns)
                }
            )
            chunk = pd.concat([chunk, len_row], ignore_index=True)

            # Estimate width and resize
            html_res = chunk.to_html(
                max_rows=2,
                max_cols=5,
                show_dimensions=False,
                index=False,
                header=header,
                escape=False,
            )
            return html_res

        # Handle sizing, trim html dataframe if output will be truncated
        df_shape = self.shape  # grab original shape information for later

        if pd.get_option("display.max_rows") is None:
            html_df = self.copy()
        elif df_shape[0] > pd.get_option("display.max_rows"):
            html_df = self.head(pd.get_option("display.min_rows") + 1)
        else:
            html_df = self.copy()

        # replace index to ensure proper behavior for duplicate index values
        index_values = html_df.index
        html_df = html_df.reset_index(drop=True)
        repr = html_df.style.format({col: repack_row for col in self.nested_columns})

        # Create a mapping function to retrieve original index
        def map_true_index(index):
            return index_values[index]

        repr = repr.format_index(map_true_index, axis=0)

        # Recover some truncation formatting, limited to head truncation
        if pd.get_option("display.max_rows") is None:
            # Just display header
            return repr.to_html(max_rows=0)
        elif df_shape[0] > pd.get_option("display.max_rows"):
            # when over the max_rows threshold, display with truncation ("..." row at the end)
            html_repr = repr.to_html(max_rows=pd.get_option("display.min_rows"))
        else:
            # when under the max_rows threshold, display all rows (behavior of 0 here)
            html_repr = repr.to_html(max_rows=0)

        # Manually append dimensionality to a styler output
        html_repr += f"{df_shape[0]} rows x {df_shape[1]} columns"

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
        if isinstance(item, str):
            return self._getitem_str(item)
        elif self._is_key_list(item):
            return self._getitem_list(item)

        return super().__getitem__(item)

    def _getitem_str(self, item):
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

    def _is_key_list(self, item):
        if not is_list_like(item):
            return False
        for k in item:
            if not isinstance(k, str):
                return False
            if not self._is_known_column(k):
                return False
        return True

    def _getitem_list(self, item):
        non_nested_keys = [k for k in item if k in self.columns]
        result = super().__getitem__(non_nested_keys)
        components = [self._parse_hierarchical_components(k) for k in item]
        nested_components = [c for c in components if self._is_known_hierarchical_column(c)]
        nested_columns = defaultdict(list)
        for comps in nested_components:
            nested_columns[comps[0]].append(".".join(comps[1:]))
        for c in nested_columns:
            result[c] = self[c].nest[nested_columns[c]]
        return result

    def __setitem__(self, key, value):
        """Custom __setitem__ for NestedFrame: auto-nest DataFrame assignment to new columns."""
        # If assigning a DataFrame to a new column, auto-nest it

        # Special handling paths for assignment of dataframes to nested columns
        if isinstance(key, str) and isinstance(value, pd.DataFrame | NestedFrame):
            # if all columns are NestedDtype, combine them into a single nested column
            if np.array([isinstance(dtype, NestedDtype) for dtype in value.dtypes]).all():
                for i, col in enumerate(value.columns):
                    if i == 0:
                        new_nested = value[col]
                    else:
                        # there must be a better way than through list fields
                        for field in value[col].nest.fields:
                            new_nested = new_nested.nest.with_list_field(
                                field, value[col].nest.get_list_series(field)
                            )
                value = new_nested
            # Assign a DataFrame as a new column, auto-nesting it
            elif key not in self.columns:
                # Note this uses the default approach for add_nested, which is a left join on index
                new_df = self.add_nested(value, name=key)
                self._update_inplace(new_df)
                return

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

        super().__setitem__(key, value)
        self._cast_cols_to_nested(struct_list=False)

    def __delitem__(self, key):
        """Delete a column or a nested field using dot notation (e.g., del nf['nested.x'])"""
        self.drop([key], axis=1, inplace=True)

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

        >>> import nested_pandas as npd
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

    def nest_lists(self, columns: list[str], name: str) -> NestedFrame:
        """Creates a new NestedFrame where the specified list-value columns are packed into a
        nested column.

        Parameters
        ----------
        columns : list[str]
            The list-value columns that should be packed into a nested column.
            All columns in the list will attempt to be packed into a single
            nested column with the name provided in `nested_name`.
        name : str
            The column name of the new nested column which we will pack the list-value
            columns into. This column will be added to the NestedFrame.

        Returns
        -------
        NestedFrame
            A new NestedFrame with the added nested columns

        Examples
        --------

        >>> import nested_pandas as npd
        >>> nf = npd.NestedFrame({"c":[1,2,3], "d":[2,4,6],
        ...                   "e":[[1,2,3], [4,5,6], [7,8,9]]},
        ...                   index=[0,1,2])

        >>> nf.nest_lists(columns=["e"], name="nested")
           c  d                nested
        0  1  2  [{e: 1}; …] (3 rows)
        1  2  4  [{e: 4}; …] (3 rows)
        2  3  6  [{e: 7}; …] (3 rows)
        """

        # Check if `name` is actually a list and `columns` is a string
        if isinstance(name, Sequence) and not isinstance(name, str) and isinstance(columns, str):
            warnings.warn(
                "DeprecationWarning: The argument order for `nest_lists` has changed: "
                "`nest_lists(name, columns)` is now `nest_lists(columns, name)`. "
                "Please update your code.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Swap the arguments
            name, columns = columns, name

        return NestedFrame.from_lists(self.copy(), list_columns=columns, name=name)

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

        >>> import nested_pandas as npd
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

        >>> import nested_pandas as npd
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
        if len(df) == 0:
            # if the dataframe is empty, just return an empty nested column
            # since there are no iterable values to pack
            packed_df = NestedFrame().add_nested(df[list_columns], name=name)
        else:
            # Check that each column has iterable elements
            for col in list_columns:
                # Check if the column is iterable based on its first value.
                # This is a simple heuristic but infers more than its dtype
                # which will probably be an object.
                sample_val = df[col].iloc[0]
                if not hasattr(sample_val, "__iter__") and not isinstance(sample_val, str | bytes):
                    raise ValueError(
                        f"Cannot pack column {col} which does not contain an iterable list based "
                        "on its first value, {sample_val}."
                    )
            packed_df = pack_lists(df[list_columns])
            packed_df.name = name

        # join the nested column to the base_column df
        if base_columns is not None:
            return df[base_columns].join(packed_df)
        # or just return the packed_df as a nestedframe if no base cols
        else:
            return NestedFrame(packed_df.to_frame())

    def drop(
        self,
        labels=None,
        *,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ):
        """Drop specified labels from rows or columns.

        Remove rows or columns by specifying label names and corresponding
        axis, or by directly specifying index or column names. When using a
        multi-index, labels on different levels can be removed by
        specifying the level. See the user guide for more information about
        the now unused levels.

        Parameters
        ----------
        labels: single label or list-like
            Index or column labels to drop. A tuple will be used as a single
            label and not treated as a list-like. Nested sub-columns are
            accessed using dot notation (e.g. "nested.col1").
        axis: {0 or ‘index’, 1 or ‘columns’}, default 0
            Whether to drop labels from the index (0 or ‘index’) or
            columns (1 or ‘columns’).
        index: single label or list-like
            Alternative to specifying axis (labels, axis=0 is equivalent to
            index=labels).
        columns: single label or list-like
            Alternative to specifying axis (labels, axis=1 is equivalent to
            columns=labels).
        level: int or level name, optional
            For MultiIndex, level from which the labels will be removed.
        inplace: bool, default False
            If False, return a copy. Otherwise, do operation in place and
            return None.
        errors: {‘ignore’, ‘raise’}, default ‘raise’
            If ‘ignore’, suppress error and only existing labels are dropped.

        Returns
        -------
        DataFrame or None
            Returns DataFrame or None DataFrame with the specified index or
            column labels removed or None if inplace=True.

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5,5, seed=1)

        >>> # drop the "t" column from "nested"
        >>> nf = nf.drop(["nested.t"], axis=1)
        >>> nf
                  a         b                                      nested
        0  0.417022  0.184677  [{flux: 31.551563, band: 'r'}; …] (5 rows)
        1  0.720324  0.372520  [{flux: 68.650093, band: 'g'}; …] (5 rows)
        2  0.000114  0.691121  [{flux: 83.462567, band: 'g'}; …] (5 rows)
        3  0.302333  0.793535   [{flux: 1.828828, band: 'g'}; …] (5 rows)
        4  0.146756  1.077633  [{flux: 75.014431, band: 'g'}; …] (5 rows)
        """

        # axis 1 requires special handling for nested columns
        if axis == 1:
            # label convergence
            if isinstance(labels, str):
                labels = [labels]
            nested_labels = [label for label in labels if self._is_known_hierarchical_column(label)]
            base_labels = [label for label in labels if not self._is_known_hierarchical_column(label)]

            # split nested_labels by nested column
            if len(nested_labels) > 0:
                nested_cols = set([label.split(".")[0] for label in nested_labels])

                # drop targeted sub-columns for each nested column
                for col in nested_cols:
                    sub_cols = [label.split(".")[1] for label in nested_labels if label.split(".")[0] == col]
                    if inplace:
                        self[col] = self[col].nest.without_field(sub_cols)
                    else:
                        self = self.assign(**{f"{col}": self[col].nest.without_field(sub_cols)})

            # drop remaining base columns
            if len(base_labels) > 0:
                return super().drop(
                    labels=base_labels,
                    axis=axis,
                    index=index,
                    columns=columns,
                    level=level,
                    inplace=inplace,
                    errors=errors,
                )
            else:
                return self if not inplace else None
        # Otherwise just drop like pandas
        return super().drop(
            labels=labels,
            axis=axis,
            index=index,
            columns=columns,
            level=level,
            inplace=inplace,
            errors=errors,
        )

    def min(self, exclude_nest: bool = False, numeric_only: bool = False, **kwargs):
        """

        Return the minimum value of each column as a series, including nested columns
        with prefix to indicate the source column.

        This computes the column-wise minimum (axis=0) across base and nested columns.
        Row-wise minimum (axis=1) are not supported, as reductions along columns
        are the primary intended behavior for NestedFrame.

        By default, missing values (NaNs) will be skipped in the computation.

        For non-numeric columns (e.g., strings), the method returns the
        lexicographically smallest value when `numeric_only=False` (default).

        Parameters
        ----------
        exclude_nest : bool, default False
            If set to True, will exclude the nested structure and
            only computes the minimum over the base columns
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs
            See the documentation for :meth:`pandas.DataFrame.min`
            for complete details on the keyword arguments accepted by
            :meth:`min`.

        Returns
        -------
        pandas.Series

        Examples
        --------
        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5,5, seed=1)

        >>> nf_min = nf.min()
        >>> nf_min
        a              0.000114
        b              0.184677
        nested.t       0.547752
        nested.flux    1.828828
        nested.band           g
        dtype: object

        See Also
        --------
        :meth:`pandas.DataFrame.min`

        """

        if not self.nested_columns:
            return super().min(numeric_only=numeric_only, **kwargs)

        # handle base columns
        base_col = [col for col in self.columns if col not in self.nested_columns]
        base_min = super().__getitem__(base_col).min(numeric_only=numeric_only, **kwargs)

        if exclude_nest:
            return base_min

        # handle nested columns
        nested_mins = []
        for nest_col in self.nested_columns:
            nested_df = self[nest_col].nest.to_flat()
            nested_df.columns = [f"{nest_col}.{col}" for col in nested_df.columns]
            nested_mins.append(nested_df.min(numeric_only=numeric_only, **kwargs))

        # Combine base and nested min values into a single Series if applicable and return
        if base_min.empty:
            return pd.concat(nested_mins)
        else:
            return pd.concat([base_min] + nested_mins)

    def max(self, exclude_nest: bool = False, numeric_only: bool = False, **kwargs):
        """

        Return the maximum value of each column as a series, including nested columns
        with prefix to indicate the source column.

        This computes the column-wise maximum (axis=0) across base and nested columns.
        Row-wise maximum (axis=1) are not supported, as reductions along columns
        are the primary intended behavior for NestedFrame.

        By default, missing values (NaNs) will be skipped in the computation.

        For non-numeric columns (e.g., strings), the method returns the
        lexicographically largest value when `numeric_only=False` (default).

        Parameters
        ----------
        exclude_nest : bool, default False
            If set to True, will exclude the nested structure and
            only computes the maximum over the base columns
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs
            See the documentation for :meth:`pandas.DataFrame.max`
            for complete details on the keyword arguments accepted by
            :meth:`max`.

        Returns
        -------
        pandas.Series

        Examples
        --------
        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5,5, seed=1)

        >>> nf_max = nf.max()
        >>> nf_max
        a               0.720324
        b               1.077633
        nested.t       19.365232
        nested.flux    98.886109
        nested.band            r
        dtype: object

        See Also
        --------
        :meth:`pandas.DataFrame.max`

        """

        if not self.nested_columns:
            return super().max(numeric_only=numeric_only, **kwargs)

        # handle base columns
        base_col = [col for col in self.columns if col not in self.nested_columns]
        base_max = super().__getitem__(base_col).max(numeric_only=numeric_only, **kwargs)

        if exclude_nest:
            return base_max

        # handle nested columns
        nested_maxs = []
        for nest_col in self.nested_columns:
            nested_df = self[nest_col].nest.to_flat()
            nested_df.columns = [f"{nest_col}.{col}" for col in nested_df.columns]
            nested_maxs.append(nested_df.max(numeric_only=numeric_only, **kwargs))

        # Combine base and nested max values into a single Series if applicable and return
        if base_max.empty:
            return pd.concat(nested_maxs)
        else:
            return pd.concat([base_max] + nested_maxs)

    def eval(self, expr: str, *, inplace: bool = False, **kwargs) -> Any | None:
        """Evaluate a string describing operations on NestedFrame columns.

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
            See the documentation for :meth:`pandas.DataFrame.eval` for
            complete details on the keyword arguments accepted by :meth:`eval`.

        Returns
        -------
        ndarray, scalar, pandas object, nested-pandas object, or None
            The result of the evaluation or None if ``inplace=True``.

        See Also
        --------
        :meth:`pandas.DataFrame.eval`

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
        """Query the columns of a NestedFrame with a boolean expression. Specified
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
                    See the documentation for :meth:`pandas.DataFrame.query`
            for complete details on the keyword arguments accepted by
            :meth:`query`.

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


        See Also
        --------
        :meth:`pandas.DataFrame.query`

        Notes
        -----
        Queries that target a particular nested structure return a dataframe
        with rows of that particular nested structure filtered. For example,
        querying the NestedFrame "df" with nested structure "my_nested" as
        below will return all rows of df, but with mynested filtered by the
        condition: `nf.query("mynested.a > 2")`

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
                  a         b                                             nested
        0  0.417022  0.184677  [{t: 8.38389, flux: 31.551563, band: 'r'}; …] ...
        1  0.720324  0.372520  [{t: 13.70439, flux: 68.650093, band: 'g'}; …]...
        2  0.000114  0.691121  [{t: 4.089045, flux: 83.462567, band: 'g'}; …]...
        3  0.302333  0.793535  [{t: 17.562349, flux: 1.828828, band: 'g'}; …]...
        4  0.146756  1.077633  [{t: 0.547752, flux: 75.014431, band: 'g'}; …]...
        >>> # or on a specific nested column
        >>> nf.dropna(subset="nested.t")
                  a         b                                             nested
        0  0.417022  0.184677  [{t: 8.38389, flux: 31.551563, band: 'r'}; …] ...
        1  0.720324  0.372520  [{t: 13.70439, flux: 68.650093, band: 'g'}; …]...
        2  0.000114  0.691121  [{t: 4.089045, flux: 83.462567, band: 'g'}; …]...
        3  0.302333  0.793535  [{t: 17.562349, flux: 1.828828, band: 'g'}; …]...
        4  0.146756  1.077633  [{t: 0.547752, flux: 75.014431, band: 'g'}; …]...

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
                axis=axis,
                how=how,
                thresh=thresh,
                subset=subset,
                inplace=inplace,
                ignore_index=ignore_index,
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

    def reduce(self, func, *args, infer_nesting=True, append_columns=False, **kwargs) -> NestedFrame:  # type: ignore[override]
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
            A list of string column names to pull from the NestedFrame to pass along
            to the function. If the function has additional arguments, pass them as
            keyword arguments (e.g. `arg_name=value`).
        infer_nesting : bool, default True
            If True, the function will pack output columns into nested
            structures based on column names adhering to a nested naming
            scheme. E.g. "nested.b" and "nested.c" will be packed into a column
            called "nested" with columns "b" and "c". If False, all outputs
            will be returned as base columns.
        append_columns : bool, default False
            if True, the output columns should be appended to those in the original NestedFrame.
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
        >>>
        >>> # define a custom user function
        >>> # reduce will return a NestedFrame with two columns
        >>> def example_func(base_col, nested_col):
        ...     return {
        ...         "mean": np.mean(nested_col),
        ...         "mean_minus_base": np.mean(nested_col) - base_col,
        ...     }
        >>>
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
        ...    '''reduce will return a NestedFrame with nested structure'''
        ...    return {"offsets.t_a": nested_col - base_col1,
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
                raise TypeError(
                    f"Received an argument '{arg}' that is not a string. "
                    "All arguments to `reduce` must be strings corresponding to"
                    " column names to pass along to the function. If your function"
                    " has additional arguments, pass them as kwargs (arg_name=value)."
                )
            components = self._parse_hierarchical_components(arg)
            if not self._is_known_column(components):
                raise ValueError(
                    f"Received a string argument '{arg}' that was not found in the columns list. "
                    "All arguments to `reduce` must be strings corresponding to"
                    " column names to pass along to the function. If your function"
                    " has additional arguments, pass them as kwargs (arg_name=value)."
                )
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

        results = [func(*cols, *extra_args, **kwargs) for cols in zip(*iterators, strict=True)]
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

        if append_columns:
            # Append the results to the original NestedFrame
            return pd.concat([self, results_nf], axis=1)

        # Otherwise, return the results as a new NestedFrame
        return results_nf

    def to_pandas(self, list_struct=False) -> pd.DataFrame:
        """Convert to an ordinal pandas DataFrame, with no NestedDtype series.

        NestedDtype is cast to pd.ArrowDtype

        Parameters
        ----------
        list_struct: bool
            If True, cast nested columns to pandas struct-list arrow extension
            array columns. If False (default), cast nested columns to
            list-struct array columns.

        Returns
        -------
        pd.DataFrame
            Ordinal pandas DataFrame.
        """
        df = pd.DataFrame(self)
        for col in self.nested_columns:
            df[col] = df[col].array.to_arrow_ext_array(list_struct=list_struct)
        return df

    def to_parquet(self, path, **kwargs) -> None:
        """Creates parquet file(s) with the data of a NestedFrame, either
        as a single parquet file where each nested dataset is packed into its
        own column or as an individual parquet file for each layer.

        Note that here we always opt to use the pyarrow engine for writing
        parquet files.

        Parameters
        ----------
        path : str
            The path to the parquet file
        kwargs : keyword arguments, optional
            Keyword arguments to pass to
            `pyarrow.parquet.write_table
            <https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html>`_

        Returns
        -------
        None

        Examples
        --------
        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5,5, seed=1)
        >>> nf.to_parquet("nestedframe.parquet")
        """
        df = self.to_pandas(list_struct=False)

        # Write through pyarrow
        # This is potentially not zero-copy
        # Note: Without pandas metadata, index writing is not as robust set
        # preserve_index=None for best behavior but index will generally
        # need to be set manually on load
        table = pa.Table.from_pandas(df, preserve_index=None)

        # Drop pandas metadata to make sure nesteddtypes are not preserved
        # Do this by rebuilding the schema
        table = table.cast(pa.schema([field for field in table.schema]))

        return pq.write_table(table, path, **kwargs)
