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

    Available as ".nest" property of a Series with NestedDtype.

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

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5, 2, seed=1)

        >>> nf["nested"].nest.to_lists()
                                   t                       flux       band
        0  [ 8.38389029 13.4093502 ]  [80.07445687 89.46066635]  ['r' 'g']
        1  [13.70439001  8.34609605]  [96.82615757  8.50442114]  ['g' 'g']
        2  [ 4.08904499 11.17379657]  [31.34241782  3.90547832]  ['g' 'g']
        3  [17.56234873  2.80773877]  [69.23226157 16.98304196]  ['r' 'r']
        4    [0.54775186 3.96202978]  [87.63891523 87.81425034]  ['g' 'r']
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

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5, 2, seed=1)

        >>> nf["nested"].nest.to_flat()
                   t       flux band
        0    8.38389  80.074457    r
        0   13.40935  89.460666    g
        1   13.70439  96.826158    g
        1   8.346096   8.504421    g
        2   4.089045  31.342418    g
        2  11.173797   3.905478    g
        3  17.562349  69.232262    r
        3   2.807739  16.983042    r
        4   0.547752  87.638915    g
        4    3.96203   87.81425    r

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
    def list_lengths(self) -> list[int]:
        """Lengths of the list arrays"""
        return self._series.array.list_lengths

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

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5, 2, seed=1)

        >>> nested_with_avg = nf["nested"].nest.with_field("avg_flux", 50.0)
        >>> # Look at one row of the series
        >>> nested_with_avg[0]
                  t       flux band  avg_flux
        0   8.38389  80.074457    r      50.0
        1  13.40935  89.460666    g      50.0
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

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5, 2, seed=1)

        >>> nested_with_avg = nf["nested"].nest.with_flat_field("avg_flux",
        ...                                                     50.0)
        >>> # Look at one row of the series
        >>> nested_with_avg[0]
                  t       flux band  avg_flux
        0   8.38389  80.074457    r      50.0
        1  13.40935  89.460666    g      50.0
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
             as the series.

        Returns
        -------
        pd.Series
            The new series with the field set.

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(2, 2, seed=1)

        >>> nf_new_band = nf["nested"].nest.with_list_field("new_band",
        ...                                                 [["g","g"],
        ...                                                  ["r","r"]])
        >>> # Look at one row of the series
        >>> nf_new_band[0]
                  t       flux band new_band
        0  2.935118  39.676747    g        g
        1  3.725204  41.919451    r        g

        """
        new_array = self._series.array.copy()
        new_array.set_list_field(field, value)
        return pd.Series(new_array, copy=False, index=self._series.index, name=self._series.name)

    def with_filled_field(self, field: str, value: ArrayLike) -> pd.Series:
        """Set the field by repeating values and return a new series

        The input value array must have as many elements as the Series,
        each of them will be repeated in the corresponding list.

        .nest.with_repeated_field("a", [1, 2, 3]) will create a nested field
        "a" with values [[1, 1, ...], [2, 2, ...], [3, 3, ...]].

        Parameters
        ----------
        field : str
            Name of the field to set. If not present, it will be added.
        value : ArrayLike
            Array of values to set. It must have the same length as the series.

        Returns
        -------
        pd.Series
            The new series with the field set.

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(3, 2, seed=1)

        >>> nf_filled = nf["nested"].nest.with_filled_field("a", [1,2,3])

        >>> # Look at one row of the series
        >>> nf_filled[0]
                   t       flux band  a
        0   3.725204  20.445225    g  1
        1  10.776335  67.046751    r  1
        """
        new_array = self._series.array.copy()
        new_array.fill_field_lists(field, value)
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

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5, 2, seed=1)

        >>> nf["nested"].nest.without_field("flux")
        0      [{t: 8.38389, band: 'r'}; …] (2 rows)
        1     [{t: 13.70439, band: 'g'}; …] (2 rows)
        2     [{t: 4.089045, band: 'g'}; …] (2 rows)
        3    [{t: 17.562349, band: 'r'}; …] (2 rows)
        4     [{t: 0.547752, band: 'g'}; …] (2 rows)
        Name: nested, dtype: nested<t: [double], band: [string]>
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

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5, 5, seed=1)

        >>> nf["nested"].nest.query_flat("flux > 50")
        0          [{t: 13.40935, flux: 98.886109, band: 'g'}]
        1    [{t: 13.70439, flux: 68.650093, band: 'g'}; …]...
        2          [{t: 4.089045, flux: 83.462567, band: 'g'}]
        3    [{t: 2.807739, flux: 78.927933, band: 'r'}; …]...
        4    [{t: 0.547752, flux: 75.014431, band: 'g'}; …]...
        dtype: nested<t: [double], flux: [double], band: [string]>
        """
        flat = self.to_flat().query(query)

        if len(flat) == 0:
            return pd.Series(
                [], dtype=self._series.dtype, index=pd.Index([], dtype=flat.index.dtype, name=flat.index.name)
            )
        return pack_sorted_df_into_struct(flat)

    def get_flat_index(self) -> pd.Index:
        """Index of the flat arrays

        Returns
        -------
        pd.Index
            The index of the flat arrays. It is a repeated index of the
            original index, with the number of repetitions equal to the
            number of elements in the list-array field.

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5, 2, seed=1)

        >>> nf["nested"].nest.get_flat_index()
        Index([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype='int64')
        """
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

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5, 2, seed=1)

        >>> nf["nested"].nest.get_flat_series("flux")
        0    80.074457
        0    89.460666
        1    96.826158
        1     8.504421
        2    31.342418
        2     3.905478
        3    69.232262
        3    16.983042
        4    87.638915
        4     87.81425
        Name: flux, dtype: double[pyarrow]
        """

        flat_chunks = []
        for nested_chunk in self._series.array._chunked_array.iterchunks():
            struct_array = cast(pa.StructArray, nested_chunk)
            list_array = cast(pa.ListArray, struct_array.field(field))
            flat_array = list_array.flatten()
            flat_chunks.append(flat_array)

        flat_chunked_array = pa.chunked_array(flat_chunks)

        return pd.Series(
            flat_chunked_array,
            dtype=pd.ArrowDtype(flat_chunked_array.type),
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

        Examples
        --------

        >>> from nested_pandas.datasets.generation import generate_data
        >>> nf = generate_data(5, 2, seed=1)

        >>> nf["nested"].nest.get_list_series("flux")
        0    [80.07445687 89.46066635]
        1    [96.82615757  8.50442114]
        2    [31.34241782  3.90547832]
        3    [69.23226157 16.98304196]
        4    [87.63891523 87.81425034]
        Name: flux, dtype: list<item: double>[pyarrow]
        """

        list_chunks = []
        for nested_chunk in self._series.array._chunked_array.iterchunks():
            struct_array = cast(pa.StructArray, nested_chunk)
            list_array = struct_array.field(field)
            list_chunks.append(list_array)
        list_chunked_array = pa.chunked_array(list_chunks)
        return pd.Series(
            list_chunked_array,
            dtype=pd.ArrowDtype(list_chunked_array.type),
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
