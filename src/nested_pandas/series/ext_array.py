# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import ArrayLike

# Needed by ArrowExtensionArray.to_numpy(na_value=no_default)
from pandas._libs.lib import no_default

# It is considered to be an experimental, so we need to be careful with it.
from pandas.core.arrays import ArrowExtensionArray

from nested_pandas.series.dtype import NestedDtype
from nested_pandas.series.utils import enumerate_chunks, is_pa_type_a_list

__all__ = ["NestedExtensionArray"]


class NestedExtensionArray(ArrowExtensionArray):
    """Pandas extension array for nested dataframes

    Parameters
    ----------
    values : pyarrow.Array or pyarrow.ChunkedArray
        The array to be wrapped, must be a struct array with all fields being list
        arrays of the same lengths.

    validate : bool, default True
        Whether to validate the input array.

    Raises
    ------
    ValueError
        If the input array is not a struct array or if any of the fields is not
        a list array or if the list arrays have different lengths.
    """

    _dtype: NestedDtype

    def __init__(self, values: pa.Array | pa.ChunkedArray, *, validate: bool = True) -> None:
        super().__init__(values=values)

        # Fix the dtype to be NestedDtype
        self._dtype = NestedDtype.from_pandas_arrow_dtype(self._dtype)

        if validate:
            self._validate(self._pa_array)

    @staticmethod
    def _convert_df_to_pa_scalar(df: pd.DataFrame, *, type: pa.DataType | None) -> pa.Scalar:
        d = {column: series.values for column, series in df.to_dict("series").items()}
        return pa.scalar(d, type=type)

    @staticmethod
    def _convert_df_value_to_pa(value: object, *, type: pa.DataType | None) -> object:
        # Convert "scalar" pd.DataFrame to a dict
        if isinstance(value, pd.DataFrame):
            return NestedExtensionArray._convert_df_to_pa_scalar(value, type=type)
        # Convert pd.DataFrame collection to a list of dicts
        if hasattr(value, "__getitem__") and isinstance(value, Iterable):
            if hasattr(value, "iloc"):
                first = value.iloc[0]
            else:
                try:
                    first = value[0]  # type: ignore[index]
                except IndexError:
                    return value
            if isinstance(first, pd.DataFrame):
                return [NestedExtensionArray._convert_df_to_pa_scalar(v, type=type) for v in value]
        return value

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy: bool = False) -> Self:  # type: ignore[name-defined] # noqa: F821
        scalars = cls._convert_df_value_to_pa(scalars, type=None)
        # The previous line may return an iterator, but parent's _from_sequence needs Sequence
        if not isinstance(scalars, Sequence) and isinstance(scalars, Collection):
            scalars = list(scalars)
        if isinstance(dtype, NestedDtype):
            dtype = dtype.to_pandas_arrow_dtype()
        return super()._from_sequence(scalars, dtype=dtype, copy=copy)

    @staticmethod
    def _validate(array: pa.ChunkedArray) -> None:
        for chunk in array.iterchunks():
            if not pa.types.is_struct(chunk.type):
                raise ValueError(f"Expected a StructArray, got {chunk.type}")
            struct_array = cast(pa.StructArray, chunk)

            first_list_array: pa.ListArray | None = None
            for field in struct_array.type:
                inner_array = struct_array.field(field.name)
                if not is_pa_type_a_list(inner_array.type):
                    raise ValueError(f"Expected a ListArray, got {inner_array.type}")
                list_array = cast(pa.ListArray, inner_array)

                if first_list_array is None:
                    first_list_array = list_array
                    continue
                # compare offsets from the first list array with the current one
                if not first_list_array.offsets.equals(list_array.offsets):
                    raise ValueError("Offsets of all ListArrays must be the same")

    @classmethod
    def from_arrow_ext_array(cls, array: ArrowExtensionArray) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Create a NestedExtensionArray from pandas' ArrowExtensionArray"""
        return cls(array._pa_array)

    def to_arrow_ext_array(self) -> ArrowExtensionArray:
        """Convert the extension array to pandas' ArrowExtensionArray"""
        return ArrowExtensionArray(self._pa_array)

    def _replace_pa_array(self, pa_array: pa.ChunkedArray, *, validate: bool) -> None:
        if validate:
            self._validate(pa_array)
        self._pa_array = pa_array
        self._dtype = NestedDtype(pa_array.chunk(0).type)

    def __getitem__(self, item):
        value = super().__getitem__(item)
        # Convert "scalar" value to pd.DataFrame
        if not isinstance(value, dict):
            return value
        return pd.DataFrame(value, copy=True)

    def __iter__(self) -> Iterator[Any]:
        for value in super().__iter__():
            # Convert "scalar" value to pd.DataFrame
            if not isinstance(value, dict):
                yield value
            else:
                yield pd.DataFrame(value, copy=True)

    def to_numpy(self, dtype: None = None, copy: bool = False, na_value: Any = no_default) -> np.ndarray:
        """Convert the extension array to a numpy array.

        Parameters
        ----------
        dtype : None
            This parameter is left for compatibility with the base class
            method, but it is not used. dtype of the returned array is
            always object.
        copy : bool, default False
            Whether to copy the data. It is not garanteed that the data
            will not be copied if copy is False.
        na_value : Any, default no_default
            TODO: support NA values

        Returns
        -------
        np.ndarray
            The numpy array of pd.DataFrame objects. Each element is a single
            time-series.
        """
        array = super().to_numpy(dtype=dtype, copy=copy, na_value=na_value)

        # Hack with np.empty is the only way to force numpy to create 1-d array of objects
        result = np.empty(shape=array.shape, dtype=object)

        # We do copy=False here because user's 'copy' is already handled by ArrowExtensionArray.to_numpy
        result[:] = [pd.DataFrame(value, copy=False) if not pd.isna(value) else pd.NA for value in array]
        return result

    def __setitem__(self, key, value) -> None:
        value = self._convert_df_value_to_pa(value, type=self._dtype.pyarrow_dtype)
        super().__setitem__(key, value)

    @property
    def list_offsets(self) -> pa.ChunkedArray:
        """The list offsets of the field arrays.

        It is a chunk array of list offsets of the first field array.
        (Since all fields are validated to have the same offsets.)

        Returns
        -------
        pa.ChunkedArray
            The list offsets of the field arrays.
        """
        return pa.chunked_array([chunk.field(0).offsets for chunk in self._pa_array.iterchunks()])

    @property
    def field_names(self) -> list[str]:
        """Names of the nested columns"""
        return [field.name for field in self._pa_array.chunk(0).type]

    @property
    def flat_length(self) -> int:
        """Length of the flat arrays"""
        return sum(chunk.field(0).value_lengths().sum().as_py() for chunk in self._pa_array.iterchunks())

    def view_fields(self, fields: str | list[str]) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Get a view of the series with only the specified fields

        Parameters
        ----------
        fields : str or list of str
            The name of the field or a list of names of the fields to include.

        Returns
        -------
        NestedExtensionArray
            The view of the series with only the specified fields.
        """
        if isinstance(fields, str):
            fields = [fields]
        if len(set(fields)) != len(fields):
            raise ValueError("Duplicate field names are not allowed")
        if not set(fields).issubset(self.field_names):
            raise ValueError(f"Some fields are not found, given: {fields}, available: {self.field_names}")

        chunks = []
        for chunk in self._pa_array.iterchunks():
            chunk = cast(pa.StructArray, chunk)
            struct_dict = {}
            for field in fields:
                struct_dict[field] = chunk.field(field)
            struct_array = pa.StructArray.from_arrays(struct_dict.values(), struct_dict.keys())
            chunks.append(struct_array)
        pa_array = pa.chunked_array(chunks)

        return self.__class__(pa_array, validate=False)

    def set_flat_field(self, field: str, value: ArrayLike) -> None:
        """Set the field from flat-array of values

        Parameters
        ----------
        field : str
            The name of the field.
        value : ArrayLike
            The 'flat' array of values to be set.
        """
        # TODO: optimize for the case when the input is a pa.ChunkedArray

        if np.ndim(value) == 0:
            value = np.repeat(value, self.flat_length)

        pa_array = pa.array(value)

        if len(pa_array) != self.flat_length:
            raise ValueError("The input must be a scalar or have the same length as the flat arrays")

        offsets = self.list_offsets.combine_chunks()
        list_array = pa.ListArray.from_arrays(values=pa_array, offsets=offsets)

        return self.set_list_field(field, list_array)

    def set_list_field(self, field: str, value: ArrayLike) -> None:
        """Set the field from list-array

        Parameters
        ----------
        field : str
            The name of the field.
        value : ArrayLike
            The list-array of values to be set.
        """
        # TODO: optimize for the case when the input is a pa.ChunkedArray

        pa_array = pa.array(value)

        if not is_pa_type_a_list(pa_array.type):
            raise ValueError(f"Expected a list array, got {pa_array.type}")

        if len(pa_array) != len(self):
            raise ValueError("The length of the list-array must be equal to the length of the series")

        chunks = []
        for sl, chunk in enumerate_chunks(self._pa_array):
            chunk = cast(pa.StructArray, chunk)

            # Build a new struct array. We collect all existing fields and add the new one.
            struct_dict = {}
            for pa_field in chunk.type:
                struct_dict[pa_field.name] = chunk.field(pa_field.name)
            struct_dict[field] = pa.array(pa_array[sl])

            struct_array = pa.StructArray.from_arrays(struct_dict.values(), struct_dict.keys())
            chunks.append(struct_array)
        pa_array = pa.chunked_array(chunks)

        self._replace_pa_array(pa_array, validate=True)

    def pop_field(self, field: str):
        """Delete a field from the struct array

        Parameters
        ----------
        field : str
            The name of the field to be deleted.
        """
        if field not in self.field_names:
            raise ValueError(f"Field '{field}' not found")

        if len(self.field_names) == 1:
            raise ValueError("Cannot delete the last field")

        chunks = []
        for chunk in self._pa_array.iterchunks():
            chunk = cast(pa.StructArray, chunk)
            struct_dict = {}
            for pa_field in chunk.type:
                if pa_field.name != field:
                    struct_dict[pa_field.name] = chunk.field(pa_field.name)
            struct_array = pa.StructArray.from_arrays(struct_dict.values(), struct_dict.keys())
            chunks.append(struct_array)
        pa_array = pa.chunked_array(chunks)

        self._replace_pa_array(pa_array, validate=False)
