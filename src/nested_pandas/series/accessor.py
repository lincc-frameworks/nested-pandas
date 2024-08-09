# Python 3.9 doesn't support "|" for types
from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator, Mapping
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import ArrayLike
from pandas.api.extensions import register_series_accessor

from nested_pandas.series.dtype import NestedDtype
from nested_pandas.series.packer import pack_sorted_df_into_struct

__all__ = ["NestSeriesAccessor"]


@register_series_accessor("nest")
class NestSeriesAccessor(Mapping):
    """Accessor for operations on Series of NestedDtype

    This accessor implements `MutableMapping` interface over the fields of the
    struct, so you can access, change and delete the fields as if it was a
    dictionary, with `[]`, `[] =` and `del` operators.
    """

    def __init__(self, series):
        self._check_series(series)

        self._series = series

    @staticmethod
    def _check_series(series):
        dtype = series.dtype
        if not isinstance(dtype, NestedDtype):
            raise AttributeError(f"Can only use .nest accessor with a Series of NestedDtype, got {dtype}")

    def to_lists(self, fields: list[str] | None = None) -> pd.DataFrame:
        """Convert nested series into dataframe of list-array columns

        Parameters
        ----------
        fields : list[str] or None, optional
            Names of the fields to include. Default is None, which means all fields.

        Returns
        -------
        pd.DataFrame
            Dataframe of list-arrays.
        """
        fields = fields if fields is not None else list(self._series.array.field_names)
        if len(fields) == 0:
            raise ValueError("Cannot convert a struct with no fields to lists")

        list_chunks = defaultdict(list)
        for chunk in self._series.array._chunked_array.iterchunks():
            struct_array = cast(pa.StructArray, chunk)
            for field in fields:
                list_array = cast(pa.ListArray, struct_array.field(field))
                list_chunks[field].append(list_array)

        list_series = {}
        for field, chunks in list_chunks.items():
            chunked_array = pa.chunked_array(chunks)
            list_series[field] = pd.Series(
                chunked_array,
                dtype=pd.ArrowDtype(chunked_array.type),
                index=self._series.index,
                name=field,
                copy=False,
            )

        return pd.DataFrame(list_series)

    def to_flat(self, fields: list[str] | None = None) -> pd.DataFrame:
        """Convert nested series into dataframe of flat arrays

        Parameters
        ----------
        fields : list[str] or None, optional
            Names of the fields to include. Default is None, which means all fields.

        Returns
        -------
        pd.DataFrame
            Dataframe of flat arrays.
        """
        fields = fields if fields is not None else list(self._series.array.field_names)
        if len(fields) == 0:
            raise ValueError("Cannot flatten a struct with no fields")

        index = pd.Series(self.get_flat_index(), name=self._series.index.name)

        flat_chunks = defaultdict(list)
        for chunk in self._series.array._chunked_array.iterchunks():
            struct_array = cast(pa.StructArray, chunk)
            for field in fields:
                list_array = cast(pa.ListArray, struct_array.field(field))
                flat_array = list_array.flatten()
                flat_chunks[field].append(flat_array)

        flat_series = {}
        for field, chunks in flat_chunks.items():
            chunked_array = pa.chunked_array(chunks)
            flat_series[field] = pd.Series(
                chunked_array,
                index=pd.Series(index, name=self._series.index.name),
                name=field,
                copy=False,
                dtype=pd.ArrowDtype(chunked_array.type),
            )

        return pd.DataFrame(flat_series)

    @property
    def flat_length(self) -> int:
        """Length of the flat arrays"""
        return self._series.array.flat_length

    @property
    def fields(self) -> list[str]:
        """Names of the nested columns"""
        return self._series.array.field_names

    def with_field(self, field: str, value: ArrayLike) -> pd.Series:
        """Set the field from flat-array of values and return a new series

        It is an alias for `.nest.with_flat_field`.

        Parameters
        ----------
        field : str
            Name of the field to set. If not present, it will be added.
        value : ArrayLike
            Array of values to set. It must be a scalar or have the same length
             as the flat arrays, e.g. `self.flat_length`.

        Returns
        -------
        pd.Series
            The new series with the field set.
        """
        return self.with_flat_field(field, value)

    def with_flat_field(self, field: str, value: ArrayLike) -> pd.Series:
        """Set the field from flat-array of values and return a new series

        Parameters
        ----------
        field : str
            Name of the field to set. If not present, it will be added.
        value : ArrayLike
            Array of values to set. It must be a scalar or have the same length
             as the flat arrays, e.g. `self.flat_length`.

        Returns
        -------
        pd.Series
            The new series with the field set.
        """
        new_array = self._series.array.copy()
        new_array.set_flat_field(field, value)
        return pd.Series(new_array, copy=False, index=self._series.index, name=self._series.name)

    def with_list_field(self, field: str, value: ArrayLike) -> pd.Series:
        """Set the field from list-array of values and return a new series

        Parameters
        ----------
        field : str
            Name of the field to set. If not present, it will be added.
        value : ArrayLike
            Array of values to set. It must be a list-array of the same length
             as the series, e.g. length of the series.

        Returns
        -------
        pd.Series
            The new series with the field set.
        """
        new_array = self._series.array.copy()
        new_array.set_list_field(field, value)
        return pd.Series(new_array, copy=False, index=self._series.index, name=self._series.name)

    def without_field(self, field: str | list[str]) -> pd.Series:
        """Remove the field(s) from the series and return a new series

        Note, that at least one field must be left in the series.

        Parameters
        ----------
        field : str or list[str]
            Name of the field(s) to remove.

        Returns
        -------
        pd.Series
            The new series without the field(s).
        """
        if isinstance(field, str):
            field = [field]

        new_array = self._series.array.copy()
        new_array.pop_fields(field)
        return pd.Series(new_array, copy=False, index=self._series.index, name=self._series.name)

    def query_flat(self, query: str) -> pd.Series:
        """Query the flat arrays with a boolean expression

        Currently, it will remove empty rows from the output series.
        # TODO: preserve the index keeping empty rows

        Parameters
        ----------
        query : str
            Boolean expression to filter the rows.

        Returns
        -------
        pd.Series
            The filtered series.
        """
        flat = self.to_flat().query(query)

        if len(flat) == 0:
            return pd.Series(
                [], dtype=self._series.dtype, index=pd.Index([], dtype=flat.index.dtype, name=flat.index.name)
            )
        return pack_sorted_df_into_struct(flat)

    def get_flat_index(self) -> pd.Index:
        """Index of the flat arrays"""
        flat_index = np.repeat(self._series.index, np.diff(self._series.array.list_offsets))
        # pd.Index supports np.repeat, so flat_index is the same type as self._series.index
        flat_index = cast(pd.Index, flat_index)
        return flat_index

    def get_flat_series(self, field: str) -> pd.Series:
        """Get the flat-array field as a Series

        Parameters
        ----------
        field : str
            Name of the field to get.

        Returns
        -------
        pd.Series
            The flat-array field.
        """

        # TODO: we should make proper missed values handling here

        struct_array = cast(pa.StructArray, pa.array(self._series))
        list_array = cast(pa.ListArray, struct_array.field(field))
        flat_array = list_array.flatten()
        return pd.Series(
            flat_array,
            dtype=pd.ArrowDtype(flat_array.type),
            index=self.get_flat_index(),
            name=field,
            copy=False,
        )

    def get_list_series(self, field: str) -> pd.Series:
        """Get the list-array field as a Series

        Parameters
        ----------
        field : str
            Name of the field to get.

        Returns
        -------
        pd.Series
            The list-array field.
        """
        struct_array = cast(pa.StructArray, pa.array(self._series))
        list_array = struct_array.field(field)
        return pd.Series(
            list_array,
            dtype=pd.ArrowDtype(list_array.type),
            index=self._series.index,
            name=field,
            copy=False,
        )

    def __getitem__(self, key: str | list[str]) -> pd.Series:
        if isinstance(key, list):
            new_array = self._series.array.view_fields(key)
            return pd.Series(new_array, index=self._series.index, name=self._series.name)

        return self.get_flat_series(key)

    def __setitem__(self, key: str, value: ArrayLike) -> None:
        """Replace the field values from flat-array of values

        Currently, only replacement of the whole field is supported, the length
        and dtype of the input value must match the field.
        https://github.com/lincc-frameworks/nested-pandas/issues/87
        """
        # TODO: we can be much-much smarter about the performance here
        # TODO: think better about underlying pa.ChunkArray in both self._series.array and value

        ndim = np.ndim(value)

        # Everything is empty, do nothing
        if len(self._series) == 0 and ndim != 0:
            array = pa.array(value, from_pandas=True)
            if len(array) == 0:
                return

        # Set single value for all rows
        if ndim == 0:
            self._series.array.set_flat_field(key, value, keep_dtype=True)
            return

        if isinstance(value, pd.Series) and not self.get_flat_index().equals(value.index):
            raise ValueError("Cannot set field with a Series of different index")

        pa_array = pa.array(value, from_pandas=True)

        # Input is a flat array of values
        if len(pa_array) != self.flat_length:
            ValueError(
                f"Cannot set field {key} with value of length {len(pa_array)}, the value is expected to be "
                f"either a scalar, a 'flat' array of length {self.flat_length}, or a 'list' array of length "
                f"{len(self._series)}."
            )

        self._series.array.set_flat_field(key, pa_array, keep_dtype=True)

    def __iter__(self) -> Generator[str, None, None]:
        return iter(self._series.array.field_names)

    def __len__(self) -> int:
        return len(self._series.array.field_names)

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self._series.equals(other._series)

    def clear(self) -> None:
        """Mandatory MutableMapping method, always fails with NotImplementedError

        The reason is that we cannot delete all nested fields from the nested series.
        """
        raise NotImplementedError("Cannot delete fields from nested series")
