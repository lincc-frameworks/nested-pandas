# This code is massively adopted from pandas' ArrowExtensionArray. Pandas license is required for this code:
#
# BSD 3-Clause License
#
# Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team
# All rights reserved.
#
# Copyright (c) 2011-2024, Open source contributors.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator, Sequence
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import ArrayLike
from pandas import Index
from pandas._typing import InterpolateOptions, Self
from pandas.api.extensions import no_default
from pandas.core.arrays import ArrowExtensionArray, ExtensionArray
from pandas.core.indexers import check_array_indexer, unpack_tuple_and_ellipses, validate_indices

from nested_pandas.series.dtype import NestedDtype
from nested_pandas.series.utils import enumerate_chunks, is_pa_type_a_list

__all__ = ["NestedExtensionArray"]


def to_pyarrow_dtype(dtype: NestedDtype | pd.ArrowDtype | pa.DataType | None) -> pa.DataType | None:
    """Convert the dtype to pyarrow.DataType"""
    if isinstance(dtype, NestedDtype):
        return dtype.pyarrow_dtype
    if isinstance(dtype, pd.ArrowDtype):
        return dtype.pyarrow_dtype
    if isinstance(dtype, pa.DataType):
        return dtype
    return None


def replace_with_mask(array: pa.ChunkedArray, mask: pa.BooleanArray, value: pa.Array) -> pa.ChunkedArray:
    """Replace the elements of the array with the value where the mask is True"""
    # TODO: performance optimization
    # https://github.com/lincc-frameworks/nested-pandas/issues/52

    # If mask is [False, True, False, True], mask_cumsum will be [0, 1, 1, 2]
    # So we put value items to the right positions in broadcast_value, while duplicate some other items for
    # the positions where mask is False.
    mask_cumsum = pa.compute.cumulative_sum(mask.cast(pa.int32()))
    value_index = pa.compute.subtract(mask_cumsum, 1)
    value_index = pa.compute.if_else(pa.compute.less(value_index, 0), 0, value_index)

    broadcast_value = value.take(value_index)
    return pa.compute.if_else(mask, broadcast_value, array)


def convert_df_to_pa_scalar(df: pd.DataFrame, *, pa_type: pa.DataType | None) -> pa.Scalar:
    d = {column: series.values for column, series in df.to_dict("series").items()}
    return pa.scalar(d, type=pa_type, from_pandas=True)


class NestedExtensionArray(ExtensionArray):
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

    # ExtensionArray overrides #

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy: bool = False) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Construct a NestedExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            The sequence of scalars: disctionaries, DataFrames, None, pd.NA, pa.Array or anything convertible
            to PyArrow scalars.
        dtype : dtype or None
            dtype of the resulting array
        copy : bool
            Ignored, because PyArrow arrays are immutable.
        """
        del copy

        pa_type = to_pyarrow_dtype(dtype)
        pa_array = cls._box_pa_array(scalars, pa_type=pa_type)
        return cls(pa_array)

    # Tricky to implement, but required by things like pd.read_csv
    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy: bool = False) -> Self:  # type: ignore[name-defined] # noqa: F821
        return super()._from_sequence_of_strings(strings, dtype=dtype, copy=copy)

    # We do not implement it. ArrowExtensionArray does not implement it for struct arrays
    @classmethod
    def _from_factorized(cls, values, original):
        return super()._from_factorized(values, original)

    def __getitem__(self, item):
        item = check_array_indexer(self, item)

        if isinstance(item, np.ndarray):
            if len(item) == 0:
                return type(self)(pa.chunked_array([], type=self._chunked_array.type), validate=False)
            pa_item = pa.array(item)
            if item.dtype.kind in "iu":
                return type(self)(self._chunked_array.take(pa_item), validate=False)
            if item.dtype.kind == "b":
                return type(self)(self._chunked_array.filter(pa_item), validate=False)
            # It should be covered by check_array_indexer above
            raise IndexError(
                "Only integers, slices and integer or " "boolean arrays are valid indices."
            )  # pragma: no cover

        if isinstance(item, tuple):
            item = unpack_tuple_and_ellipses(item)

        if item is Ellipsis:
            item = slice(None)

        scalar_or_array = self._chunked_array[item]
        if isinstance(scalar_or_array, pa.StructScalar):
            return self._convert_struct_scalar_to_df(scalar_or_array, copy=False)
        # Logically, it must be a pa.ChunkedArray if it is not a scalar
        pa_array = cast(pa.ChunkedArray, scalar_or_array)
        return type(self)(pa_array, validate=False)

    def __setitem__(self, key, value) -> None:
        # TODO: optimize for many chunks
        # https://github.com/lincc-frameworks/nested-pandas/issues/53

        key = check_array_indexer(self, key)

        if isinstance(key, tuple):
            key = unpack_tuple_and_ellipses(key)

        if not isinstance(key, np.ndarray):
            np_mask = np.zeros(len(self), dtype=np.bool_)
            np_mask[key] = True
            key = np_mask

        if len(key) == 0:
            return

        argsort: np.ndarray | None = None
        if key.dtype.kind in "iu":
            _, argsort = np.unique(key, return_index=True)
            np_mask = np.zeros(len(self), dtype=np.bool_)
            np_mask[key] = True
            pa_mask = pa.array(np_mask)
        elif key.dtype.kind == "b":
            pa_mask = pa.array(key)
        # Should be covered by check_array_indexer
        else:  # pragma: no cover
            raise IndexError(
                "Only integers, slices and integer or boolean arrays are valid indices."
            )  # pragma: no cover

        # Try to convert to struct_scalar first, if it fails, convert to array
        try:
            scalar = self._box_pa_scalar(value, pa_type=self._pyarrow_dtype)
        except (ValueError, TypeError):
            # Copy will happen later in replace_with_mask() anyway
            value = self._box_pa_array(value, pa_type=self._pyarrow_dtype)
        else:
            # Our replace_with_mask implementation doesn't work with scalars
            value = pa.array([scalar] * pa.compute.sum(pa_mask).as_py())

        if argsort is not None:
            value = value.take(argsort)

        # We cannot use pa.compute.replace_with_mask(), it is not implemented for struct arrays:
        # https://github.com/apache/arrow/issues/29558
        # self._chunked_array = pa.compute.replace_with_mask(self._chunked_array, pa_mask, value)
        self._chunked_array = replace_with_mask(self._chunked_array, pa_mask, value)

    def __len__(self) -> int:
        return len(self._chunked_array)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        for value in self._chunked_array:
            yield self._convert_struct_scalar_to_df(value, copy=False)

    # We do not implement it yet, because pa.compute.equal does not work for struct arrays
    # ArrowExtensionArray does not implement it for struct arrays neither
    def __eq__(self, other):
        return super().__eq__(other)

    def to_numpy(self, dtype: None = None, copy: bool = False, na_value: Any = no_default) -> np.ndarray:
        """Convert the extension array to a numpy array.

        Parameters
        ----------
        dtype : None
            This parameter is left for compatibility with the base class
            method, but it is not used. dtype of the returned array is
            always object.
        copy : bool, default False
            Whether to copy the data. It is not guaranteed that the data
            will not be copied if copy is False.
        na_value : Any, optional
            The value to use for missing values. If not provided, None
            will be used.

        Returns
        -------
        np.ndarray
            The numpy array of pd.DataFrame objects. Each element is a single
            time-series.
        """
        if na_value is no_default:
            na_value = None

        # Hack with np.empty is the only way to force numpy to create 1-d array of objects
        result = np.empty(shape=len(self), dtype=object)

        for i, value in enumerate(self._chunked_array):
            result[i] = self._convert_struct_scalar_to_df(value, copy=copy, na_value=na_value)

        return result

    @property
    def dtype(self) -> NestedDtype:
        """ExtensionArray dtype"""
        return self._dtype

    @property
    def nbytes(self) -> int:
        """Number of bytes consumed by the data in memory."""
        return self._chunked_array.nbytes

    def isna(self) -> np.ndarray:
        """Boolean NumPy array indicating if each value is missing."""
        # Fast paths adopted from ArrowExtensionArray
        null_count = self._chunked_array.null_count
        if null_count == 0:
            return np.zeros(len(self), dtype=bool)
        if null_count == len(self):
            return np.ones(len(self), dtype=bool)

        return self._chunked_array.is_null().to_numpy()

    @property
    def _hasna(self) -> bool:
        return self._chunked_array.null_count > 0

    # We do not implement it yet, neither ArrowExtensionArray does for struct arrays
    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: int,
        index: Index,
        limit,
        limit_direction,
        limit_area,
        copy: bool,
        **kwargs,
    ) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Interpolate missing values, not implemented yet."""
        super().interpolate(
            method=method,
            axis=axis,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            copy=copy,
            **kwargs,
        )

    def take(
        self,
        indices,
        *,
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> Self:  # type: ignore[name-defined] # noqa: F821
        """
        Take elements from an array.

        Parameters
        ----------
        indices : sequence of int or one-dimensional np.ndarray of int
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              negative values raise a ``ValueError``.

        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case pd.NA is used.

        Returns
        -------
        NestedExtensionArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.
        """
        # Massively adopted from ArrowExtensionArray

        indices_array = np.asanyarray(indices)

        if len(self._chunked_array) == 0 and (indices_array >= 0).any():
            raise IndexError("cannot do a non-empty take from empty array")
        if indices_array.size > 0 and indices_array.max() >= len(self._chunked_array):
            raise IndexError("out of bounds value in 'indices'.")

        if allow_fill:
            fill_value = self._box_pa_scalar(fill_value, pa_type=self._pyarrow_dtype)

            fill_mask = indices_array < 0
            if not fill_mask.any():
                # Nothing to fill
                return type(self)(self._chunked_array.take(indices))
            validate_indices(indices_array, len(self._chunked_array))
            indices_array = pa.array(indices_array, mask=fill_mask)

            result = self._chunked_array.take(indices_array)
            if not pa.compute.is_null(fill_value).as_py():
                result = pa.compute.if_else(fill_mask, fill_value, result)
            return type(self)(result)

        if (indices_array < 0).any():
            # Don't modify in-place
            indices_array = np.copy(indices_array)
            indices_array[indices_array < 0] += len(self._chunked_array)
        return type(self)(self._chunked_array.take(indices_array))

    def copy(self) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Return a copy of the extension array.

        This implementation returns a shallow copy of the extension array,
        because the underlying PyArrow array is immutable.
        """
        return type(self)(self._chunked_array, validate=False)

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        # TODO: make formatted strings more pretty
        # https://github.com/lincc-frameworks/nested-pandas/issues/50
        if boxed:

            def box_formatter(value):
                if value is pd.NA:
                    return str(pd.NA)
                scalar = convert_df_to_pa_scalar(value, pa_type=self._pyarrow_dtype)
                return str(scalar.as_py())

            return box_formatter
        return repr

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:  # type: ignore[name-defined] # noqa: F821
        chunks = [chunk for array in to_concat for chunk in array._chunked_array.iterchunks()]
        pa_array = pa.chunked_array(chunks)
        return cls(pa_array)

    def equals(self, other) -> bool:
        """
        Check equality with another NestedExtensionArray.

        Parameters
        ----------
        other : NestedExtensionArray
            The other NestedExtensionArray to compare with.

        Returns
        -------
        bool
            Whether the two NestedExtensionArrays are equal.
        """
        if not isinstance(other, type(self)):
            return False
        return self._chunked_array == other._chunked_array

    def dropna(self) -> Self:
        """Return a new ExtensionArray with missing values removed.

        Note that this applies to the top-level struct array, not to the list arrays.
        """
        return type(self)(pa.compute.drop_null(self._chunked_array))

    # End of ExtensionArray overrides #

    # Additional magic methods #

    def __arrow_array__(self, type=None):
        """Convert the extension array to a PyArrow array."""
        if type is None:
            return self._chunked_array
        return self._chunked_array.cast(type)

    def __array__(self, dtype=None):
        """Convert the extension array to a numpy array."""
        return self.to_numpy(dtype=dtype, copy=False)

    # Adopted from ArrowExtensionArray
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_chunked_array"] = self._chunked_array.combine_chunks()
        return state

    # Adopted from ArrowExtensionArray
    def __setstate__(self, state):
        state["_chunked_array"] = pa.chunked_array(state["_chunked_array"])
        self.__dict__.update(state)

    # End of Additional magic methods #

    @classmethod
    def _box_pa_scalar(cls, value, *, pa_type: pa.DataType | None) -> pa.Scalar:
        """Convert a value to a PyArrow scalar with the specified type."""
        if isinstance(value, pa.Scalar):
            if pa_type is None:
                return value
            return value.cast(pa_type)
        if value is pd.NA or value is None:
            return pa.scalar(None, type=pa_type, from_pandas=True)
        if isinstance(value, pd.DataFrame):
            return convert_df_to_pa_scalar(value, pa_type=pa_type)
        return pa.scalar(value, type=pa_type, from_pandas=True)

    @classmethod
    def _box_pa_array(cls, value, *, pa_type: pa.DataType | None) -> pa.Array | pa.ChunkedArray:
        """Convert a value to a PyArrow array with the specified type."""
        if isinstance(value, cls):
            pa_array = value._chunked_array
        elif isinstance(value, (pa.Array, pa.ChunkedArray)):
            pa_array = value
        else:
            try:
                pa_array = pa.array(value, type=pa_type)
            except (ValueError, TypeError, KeyError):
                scalars: list[pa.Scalar] = []
                for v in value:
                    # If pa_type is not specified, then cast to the first non-null type
                    if pa_type is None and len(scalars) > 0 and not isinstance(scalars[-1], pa.NullScalar):
                        pa_type = scalars[-1].type
                    scalars.append(cls._box_pa_scalar(v, pa_type=pa_type))
                # We recast the scalars to the specified type.
                # Logically, we should 1) have `pa_type is not None` here,
                # 2) only "head" null-scalars to be not cast to the specified type.
                # However, we just cast everything to the specified type here.
                if pa_type is None:
                    pa_type = scalars[-1].type
                scalars = [s.cast(pa_type) for s in scalars]
                pa_array = pa.array(scalars)
                # We already copied the data into scalars

        # We always cast - even if the type is the same, it does not hurt
        # If the type is different the result may still be a view, so we do not set copy=False
        if pa_type is not None:
            pa_array = pa_array.cast(pa_type)

        return pa_array

    @classmethod
    def _convert_struct_scalar_to_df(cls, value: pa.StructScalar, *, copy: bool, na_value: Any = None) -> Any:
        """Converts a struct scalar of equal-length list scalars to a pd.DataFrame

        No validation is done, so the input must be a struct scalar with all fields being list scalars
        of the same lengths.
        """
        if pa.compute.is_null(value).as_py():
            return na_value
        d = {name: pd.Series(list_scalar.values, copy=copy) for name, list_scalar in value.items()}
        return pd.DataFrame(d, copy=False)

    _chunked_array: pa.ChunkedArray
    _dtype: NestedDtype

    def __init__(self, values: pa.Array | pa.ChunkedArray, *, validate: bool = True) -> None:
        if isinstance(values, pa.Array):
            values = pa.chunked_array([values])

        if validate:
            self._validate(values)

        self._chunked_array = values
        self._dtype = NestedDtype(values.type)

    @classmethod
    def from_sequence(cls, scalars, *, dtype: NestedDtype | pd.ArrowDtype | pa.DataType = None) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Construct a NestedExtensionArray from a sequence of items

        Parameters
        ----------
        scalars : Sequence
            The sequence of items: dictionaries (key is column name, value is array-like of nested elements),
            DataFrames, None, pd.NA, pa.Array or anything convertible to PyArrow scalars of struct type with
            list fields of the same lengths.
        dtype : dtype or None
            NestedDtype of the resulting array, or a type to infer from: pd.ArrowDtype or pa.DataType.

        Returns
        -------
        NestedExtensionArray
            The constructed extension array.
        """
        return cls._from_sequence(scalars, dtype=dtype)

    @property
    def _pyarrow_dtype(self) -> pa.DataType:
        """PyArrow data type of the extension array"""
        return self._dtype.pyarrow_dtype

    @property
    def chunked_array(self) -> pa.ChunkedArray:
        """The underlying PyArrow ChunkedArray"""
        return self._chunked_array

    @staticmethod
    def _validate(array: pa.ChunkedArray) -> None:
        """Raises ValueError if the input array is not a struct array with all fields being
        list arrays of the same lengths.
        """
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
        return ArrowExtensionArray(self._chunked_array)

    def _replace_chunked_array(self, pa_array: pa.ChunkedArray, *, validate: bool) -> None:
        if validate:
            self._validate(pa_array)
        self._chunked_array = pa_array
        self._dtype = NestedDtype(pa_array.chunk(0).type)

    @property
    def list_offsets(self) -> pa.Array:
        """The list offsets of the field arrays.

        It is a chunk array of list offsets of the first field array.
        (Since all fields are validated to have the same offsets.)

        Returns
        -------
        pa.ChunkedArray
            The list offsets of the field arrays.
        """
        # Quick and cheap path for a single chunk
        if self._chunked_array.num_chunks == 1:
            struct_array = cast(pa.StructArray, self._chunked_array.chunk(0))
            return cast(pa.ListArray, struct_array.field(0)).offsets

        chunks = []
        # The offset of the current chunk in the flat array.
        # Offset arrays use int32 type, so we cast to it
        chunk_offset = pa.scalar(0, type=pa.int32())
        for chunk in self._chunked_array.iterchunks():
            list_array = cast(pa.ListArray, chunk.field(0))
            if chunk_offset.equals(pa.scalar(0, type=pa.int32())):
                offsets = list_array.offsets
            else:
                offsets = pa.compute.add(list_array.offsets[1:], chunk_offset)
            chunks.append(offsets)
            chunk_offset = offsets[-1]
        return pa.concat_arrays(chunks)

    @property
    def field_names(self) -> list[str]:
        """Names of the nested columns"""
        return [field.name for field in self._chunked_array.chunk(0).type]

    @property
    def flat_length(self) -> int:
        """Length of the flat arrays"""
        return sum(chunk.field(0).value_lengths().sum().as_py() for chunk in self._chunked_array.iterchunks())

    @property
    def num_chunks(self) -> int:
        """Number of chunks in underlying pyarrow.ChunkedArray"""
        return self._chunked_array.num_chunks

    def get_list_index(self) -> np.ndarray:
        """Keys mapping values to lists"""
        if len(self) == 0:
            # Since we have no list offsets, return an empty array
            return np.array([], dtype=int)
        list_index = np.arange(len(self))
        return np.repeat(list_index, np.diff(self.list_offsets))

    def iter_field_lists(self, field: str) -> Generator[np.ndarray, None, None]:
        """Iterate over single field nested lists, as numpy arrays

        Parameters
        ----------
        field : str
            The name of the field to iterate over.

        Yields
        ------
        np.ndarray
            The numpy array view over a list scalar.
        """
        for chunk in self._chunked_array.iterchunks():
            struct_array: pa.StructArray = cast(pa.StructArray, chunk)
            list_array: pa.ListArray = cast(pa.ListArray, struct_array.field(field))
            for list_scalar in list_array:
                yield np.asarray(list_scalar.values)

    def view_fields(self, fields: str | list[str]) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Get a view of the extension array with only the specified fields

        Parameters
        ----------
        fields : str or list of str
            The name of the field or a list of names of the fields to include.

        Returns
        -------
        NestedExtensionArray
            The view of the array with only the specified fields.
        """
        if isinstance(fields, str):
            fields = [fields]
        if len(set(fields)) != len(fields):
            raise ValueError("Duplicate field names are not allowed")
        if not set(fields).issubset(self.field_names):
            raise ValueError(f"Some fields are not found, given: {fields}, available: {self.field_names}")

        chunks = []
        for chunk in self._chunked_array.iterchunks():
            chunk = cast(pa.StructArray, chunk)
            struct_dict = {}
            for field in fields:
                struct_dict[field] = chunk.field(field)
            struct_array = pa.StructArray.from_arrays(struct_dict.values(), struct_dict.keys())
            chunks.append(struct_array)
        pa_array = pa.chunked_array(chunks)

        return self.__class__(pa_array, validate=False)

    def set_flat_field(self, field: str, value: ArrayLike, *, keep_dtype: bool = False) -> None:
        """Set the field from flat-array of values

        Note that if this updates the dtype, it would not affect the dtype of
        the pd.Series back-ended by this extension array.

        Parameters
        ----------
        field : str
            The name of the field.
        value : ArrayLike
            The 'flat' array of values to be set.
        keep_dtype : bool, default False
            Whether to keep the original dtype of the field. If True,
            now new field will be created, and the dtype of the existing
            field will be kept. If False, the dtype of the field will be
            inferred from the input value.
        """
        # TODO: optimize for the case when the input is a pa.ChunkedArray

        if keep_dtype:
            if field not in self.field_names:
                raise ValueError(
                    "If keep_dtype is True, the field must exist in the series. "
                    f"Got: {field}, available: {self.field_names}"
                )
            # Get the current element type of list-array
            pa_type = self._chunked_array.chunk(0).field(field).type.value_type
        else:
            pa_type = None

        if np.ndim(value) == 0:
            value = np.repeat(value, self.flat_length)

        try:
            pa_array = pa.array(value, from_pandas=True, type=pa_type)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"New values must be convertible to the existing element pyarrow type, {pa_type}. "
                "If you want to replace field with values of a new type, use series.nest.with_flat_field() "
                "or NestedExtensionArray.set_flat_field(..., keep_dtype=False) instead."
            ) from e

        if len(pa_array) != self.flat_length:
            raise ValueError("The input must be a struct_scalar or have the same length as the flat arrays")

        list_array = pa.ListArray.from_arrays(values=pa_array, offsets=self.list_offsets)

        return self.set_list_field(field, list_array, keep_dtype=keep_dtype)

    def set_list_field(self, field: str, value: ArrayLike, *, keep_dtype: bool = False) -> None:
        """Set the field from list-array

        Note that if this updates the dtype, it would not affect the dtype of
        the pd.Series back-ended by this extension array.

        Parameters
        ----------
        field : str
            The name of the field.
        value : ArrayLike
            The list-array of values to be set.
        keep_dtype : bool, default False
            Whether to keep the original dtype of the field. If True,
            now new field will be created, and the dtype of the existing
            field will be kept. If False, the dtype of the field will be
            inferred from the input value.
        """
        # TODO: optimize for the case when the input is a pa.ChunkedArray

        if keep_dtype:
            if field not in self.field_names:
                raise ValueError(
                    "If keep_dtype is True, the field must exist in the series. "
                    f"Got: {field}, available: {self.field_names}"
                )
            pa_type = self._chunked_array.chunk(0).field(field).type
        else:
            pa_type = None

        try:
            pa_array = pa.array(value, from_pandas=True, type=pa_type)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"New values must be convertible to the existing list pyarrow type, {pa_type}. "
                "If you want to replace field with values of a new type, use series.nest.with_list_field() "
                "or NestedExtensionArray.set_list_field(..., keep_dtype=False) instead."
            ) from e

        if not is_pa_type_a_list(pa_array.type):
            raise ValueError(f"Expected a list array, got {pa_array.type}")

        if len(pa_array) != len(self):
            raise ValueError("The length of the list-array must be equal to the length of the series")

        chunks = []
        for sl, chunk in enumerate_chunks(self._chunked_array):
            chunk = cast(pa.StructArray, chunk)

            # Build a new struct array. We collect all existing fields and add/replace the new one.
            struct_dict = {}
            for pa_field in chunk.type:
                struct_dict[pa_field.name] = chunk.field(pa_field.name)
            struct_dict[field] = pa_array[sl]

            struct_array = pa.StructArray.from_arrays(struct_dict.values(), struct_dict.keys())
            chunks.append(struct_array)
        chunked_array = pa.chunked_array(chunks)

        self._replace_chunked_array(chunked_array, validate=True)

    def pop_fields(self, fields: Iterable[str]):
        """Delete fields from the struct array

        Note that at least one field must be left in the struct array.

        Parameters
        ----------
        fields : iterable of str
            The names of the fields to delete.
        """
        fields = frozenset(fields)

        if not fields.issubset(self.field_names):
            raise ValueError(f"Some fields are not found, given: {fields}, available: {self.field_names}")

        if len(self.field_names) - len(fields) == 0:
            raise ValueError("Cannot delete all fields")

        chunks = []
        for chunk in self._chunked_array.iterchunks():
            chunk = cast(pa.StructArray, chunk)
            struct_dict = {}
            for pa_field in chunk.type:
                if pa_field.name not in fields:
                    struct_dict[pa_field.name] = chunk.field(pa_field.name)
            struct_array = pa.StructArray.from_arrays(struct_dict.values(), struct_dict.keys())
            chunks.append(struct_array)
        pa_array = pa.chunked_array(chunks)

        self._replace_chunked_array(pa_array, validate=False)
