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

"""Pandas extension array for fixed-shape tensor columns."""

from __future__ import annotations  # Self in Python 3.10

from collections.abc import Callable, Iterator, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import DTypeLike
from pandas.api.extensions import no_default
from pandas.api.types import pandas_dtype
from pandas.core.arrays import ArrowExtensionArray, ExtensionArray  # type: ignore[attr-defined]
from pandas.core.indexers import (  # type: ignore[attr-defined]
    check_array_indexer,
    unpack_tuple_and_ellipses,
    validate_indices,
)

from nested_pandas.series.ext_array import replace_with_mask
from nested_pandas.tensors.dtype import TensorDtype

__all__ = ["TensorExtensionArray"]


TENSOR_FORMATTING_MAX_ELEMENTS = 12
"""Tensors with more elements than this show a shape/dtype descriptor instead of their values in reprs"""


def _tensor_descriptor(value: np.ndarray) -> str:
    """Short description of a tensor for reprs, e.g. '[64×64] float32'."""
    return f"[{'×'.join(str(size) for size in value.shape)}] {value.dtype}"


def _format_tensor(value: Any) -> str:
    """Format a single tensor (or missing value) for Series and DataFrame text reprs.

    Small tensors show all their values, larger ones only a shape and dtype
    descriptor. Values are formatted via ``tolist()`` rather than
    ``np.array2string`` so that every row of a column uses the same style,
    with no alignment padding or per-tensor switching to scientific notation.
    """
    if _is_na(value):
        return str(pd.NA)
    if value.size > TENSOR_FORMATTING_MAX_ELEMENTS:
        return _tensor_descriptor(value)
    return str(value.tolist())


def _is_na(value: Any) -> bool:
    """Whether a scalar value represents a missing tensor."""
    return value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value))


def _numpy_value_dtype(dtype: TensorDtype) -> np.dtype:
    """Numpy dtype of the tensor elements."""
    return np.dtype(dtype.value_type.to_pandas_dtype())


def _normalize_dtype(dtype: Any) -> TensorDtype | None:
    """Convert the supported dtype spellings to TensorDtype, passing None through."""
    if dtype is None or isinstance(dtype, TensorDtype):
        return dtype
    if isinstance(dtype, pa.FixedShapeTensorType | pd.ArrowDtype):
        return TensorDtype(dtype)
    if isinstance(dtype, str):
        pd_dtype = pandas_dtype(dtype)
        if isinstance(pd_dtype, TensorDtype):
            return pd_dtype
    raise TypeError(
        "Expected TensorDtype, pa.FixedShapeTensorType, pd.ArrowDtype of a tensor type or a tensor dtype "
        f"string, got {dtype!r}"
    )


def _storage_of(array: pa.ChunkedArray) -> pa.ChunkedArray:
    """Fixed-size-list storage of a tensor-typed chunked array."""
    pa_type = cast(pa.FixedShapeTensorType, array.type)
    return pa.chunked_array([chunk.storage for chunk in array.iterchunks()], type=pa_type.storage_type)


def _tensor_to_flat(value: Any, dtype: TensorDtype) -> np.ndarray:
    """Validate a single tensor against the dtype and return its values flattened in C order."""
    if isinstance(value, pa.FixedShapeTensorScalar):
        array = value.to_numpy()
    elif isinstance(value, pa.Scalar):
        array = np.asarray(value.as_py())
        # A fixed_size_list scalar comes back flat
        if array.shape != dtype.shape and array.size == dtype.size:
            array = array.reshape(dtype.shape)
    else:
        array = np.asarray(value)
    if array.shape != dtype.shape:
        raise ValueError(f"Expected a tensor of shape {dtype.shape}, got shape {array.shape}")
    return np.ascontiguousarray(array).reshape(-1)


class TensorExtensionArray(ExtensionArray):
    """Pandas extension array for fixed-shape tensors

    Every element is a numpy array of the same shape and element type, stored
    as a pyarrow ``fixed_shape_tensor`` extension array. Scalar access and
    :meth:`to_stack` return read-only numpy views over the arrow buffers
    whenever possible, so no data is copied.

    Parameters
    ----------
    values : pyarrow.Array or pyarrow.ChunkedArray
        The array to be wrapped. Either a ``fixed_shape_tensor`` extension
        array, or its ``fixed_size_list`` storage array, in which case
        ``dtype`` is required.
    dtype : TensorDtype, optional
        The dtype of the array. Required when ``values`` is a storage array,
        otherwise inferred from ``values.type``. If given and different from
        the type of ``values``, the storage is cast to the dtype.

    Raises
    ------
    TypeError
        If ``values`` is not a pyarrow array.
    ValueError
        If ``values`` is neither a tensor nor a fixed_size_list array, or if it
        is a fixed_size_list array and ``dtype`` is not given.
    """

    # Constructor and initialized attributes #

    _pa_array: pa.ChunkedArray
    _dtype: TensorDtype

    def __init__(self, values: pa.Array | pa.ChunkedArray, *, dtype: TensorDtype | None = None) -> None:
        if isinstance(values, pa.Array):
            values = pa.chunked_array([values])
        if not isinstance(values, pa.ChunkedArray):
            raise TypeError(f"values must be a pyarrow Array or ChunkedArray, got {type(values)}")
        dtype = _normalize_dtype(dtype)

        if isinstance(values.type, pa.FixedShapeTensorType):
            if dtype is None:
                dtype = TensorDtype(values.type)
            elif dtype.pyarrow_dtype != values.type:
                values = self._wrap_storage(_storage_of(values), dtype)
        elif pa.types.is_fixed_size_list(values.type):
            if dtype is None:
                raise ValueError("dtype is required when constructing from a fixed_size_list storage array")
            values = self._wrap_storage(values, dtype)
        else:
            raise ValueError(
                f"values must be a fixed_shape_tensor or fixed_size_list array, got type {values.type}"
            )

        self._pa_array = values
        self._dtype = dtype

    @staticmethod
    def _wrap_storage(storage: pa.ChunkedArray, dtype: TensorDtype) -> pa.ChunkedArray:
        """Cast fixed_size_list storage to the dtype's storage type and wrap it as the tensor type."""
        if storage.type != dtype.storage_type:
            if storage.type.list_size != dtype.size:
                raise ValueError(
                    f"Cannot convert tensors of {storage.type.list_size} elements to {dtype}, "
                    f"which has {dtype.size} elements"
                )
            storage = storage.cast(dtype.storage_type)
        pa_type = dtype.pyarrow_dtype
        chunks = [pa.ExtensionArray.from_storage(pa_type, chunk) for chunk in storage.iterchunks()]
        return pa.chunked_array(chunks, type=pa_type)

    # End of Constructor and initialized attributes #

    # ExtensionArray overrides #

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy: bool = False) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Construct a TensorExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            The sequence of scalars: numpy arrays (or anything convertible to
            them) of the tensor shape, None, pd.NA, or pyarrow scalars. A
            numpy array of shape ``(n, *shape)`` is accepted as a stack.
        dtype : TensorDtype, pa.FixedShapeTensorType, pd.ArrowDtype or str, optional
            dtype of the resulting array. Inferred from the first non-missing
            element if not given.
        copy : bool
            Ignored, because PyArrow arrays are immutable.
        """
        del copy
        dtype = _normalize_dtype(dtype)

        if isinstance(scalars, cls):
            if dtype is None or dtype == scalars.dtype:
                return scalars
            return scalars.astype(dtype)
        if isinstance(scalars, pa.Array | pa.ChunkedArray):
            return cls(scalars, dtype=dtype)
        if isinstance(scalars, np.ndarray) and scalars.dtype != np.object_ and scalars.ndim >= 2:
            return cls.from_stack(scalars, dtype=dtype)

        scalars = list(scalars)
        mask = np.array([_is_na(value) for value in scalars], dtype=bool)

        if dtype is None:
            first = next((value for value, na in zip(scalars, mask, strict=True) if not na), None)
            if first is None:
                raise ValueError("Cannot infer TensorDtype from a sequence without non-missing values")
            if isinstance(first, pa.FixedShapeTensorScalar):
                first = first.to_numpy()
            first = np.asarray(first)
            dtype = TensorDtype(pa.fixed_shape_tensor(pa.from_numpy_dtype(first.dtype), list(first.shape)))

        np_value_dtype = _numpy_value_dtype(dtype)
        flats = [
            np.zeros(dtype.size, dtype=np_value_dtype) if na else _tensor_to_flat(value, dtype)
            for value, na in zip(scalars, mask, strict=True)
        ]
        values = np.concatenate(flats) if flats else np.empty(0, dtype=np_value_dtype)
        storage = pa.FixedSizeListArray.from_arrays(
            pa.array(values, type=dtype.value_type),
            dtype.size,
            mask=pa.array(mask) if mask.any() else None,
        )
        return cls(storage, dtype=dtype)

    def __getitem__(self, item: ScalarIndexer) -> Self | np.ndarray:  # type: ignore[name-defined, override] # noqa: F821
        item = check_array_indexer(self, item)

        if isinstance(item, np.ndarray):
            if len(item) == 0:
                return type(self)(pa.chunked_array([], type=self.dtype.pyarrow_dtype), dtype=self.dtype)
            pa_item = pa.array(item)
            if item.dtype.kind in "iu":
                return type(self)(self._pa_array.take(pa_item), dtype=self.dtype)
            if item.dtype.kind == "b":
                return type(self)(self._pa_array.filter(pa_item), dtype=self.dtype)
            # It should be covered by check_array_indexer above
            raise IndexError(
                "Only integers, slices and integer or boolean arrays are valid indices."
            )  # pragma: no cover

        if isinstance(item, tuple):
            item = unpack_tuple_and_ellipses(item)

        if item is Ellipsis:
            item = slice(None)

        scalar_or_array = self._pa_array[item]
        if isinstance(scalar_or_array, pa.Scalar):
            return self._scalar_to_numpy(scalar_or_array)
        # Logically, it must be a pa.ChunkedArray if it is not a scalar
        return type(self)(cast(pa.ChunkedArray, scalar_or_array), dtype=self.dtype)

    def __setitem__(self, key, value) -> None:
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

        n_set = pc.sum(pa_mask).as_py() or 0

        if self._is_scalar_value(value):
            # Our replace_with_mask implementation doesn't work with scalars, so broadcast
            scalar = self._scalar_storage(value)
            value_storage: pa.Array | pa.ChunkedArray = pa.repeat(scalar, n_set)
        else:
            value_storage = type(self)._from_sequence(value, dtype=self.dtype).storage
            if len(value_storage) != n_set:
                raise ValueError(
                    f"Cannot set {n_set} elements from a sequence of length {len(value_storage)}"
                )
            if argsort is not None:
                value_storage = value_storage.take(argsort)

        # pa.compute.replace_with_mask() and if_else() have no kernels for the extension type,
        # so we work on the fixed_size_list storage and wrap it back.
        storage = replace_with_mask(self.storage, pa_mask, value_storage)
        self._pa_array = self._wrap_storage(storage, self.dtype)

    def __len__(self) -> int:
        return len(self._pa_array)

    def __iter__(self) -> Iterator[np.ndarray | Any]:
        for scalar in self._pa_array:
            yield self._scalar_to_numpy(scalar)

    def to_numpy(
        self,
        dtype: DTypeLike | None = None,
        copy: bool = False,
        na_value: Any = no_default,
    ) -> np.ndarray:
        """Convert the extension array to a numpy object array of tensors.

        Use :meth:`to_stack` to get a single ``(n, *shape)`` numpy array instead.

        Parameters
        ----------
        dtype : None
            This parameter is left for compatibility with the base class
            method, but it is not used. dtype of the returned array is
            always an object.
        copy : bool, default False
            Whether to copy the tensors. If False, the tensors are read-only
            views over the arrow buffers.
        na_value : Any, optional
            The value to use for missing values. If not provided, pd.NA
            will be used.

        Returns
        -------
        np.ndarray
            The numpy array of np.ndarray objects.
        """
        del dtype
        if na_value is no_default:
            na_value = pd.NA

        # Hack with np.empty is the only way to force numpy to create a 1-d array of objects
        result = np.empty(shape=len(self), dtype=object)

        for i, scalar in enumerate(self._pa_array):
            value = self._scalar_to_numpy(scalar, na_value=na_value)
            if copy and isinstance(value, np.ndarray):
                value = value.copy()
            result[i] = value

        return result

    @property
    def dtype(self) -> TensorDtype:
        """ExtensionArray dtype"""
        return self._dtype

    @property
    def nbytes(self) -> int:
        """Number of bytes consumed by the data in memory."""
        return self._pa_array.nbytes

    def isna(self) -> np.ndarray:
        """Boolean NumPy array indicating if each value is missing."""
        # Fast paths adopted from ArrowExtensionArray
        null_count = self._pa_array.null_count
        if null_count == 0:
            return np.zeros(len(self), dtype=bool)
        if null_count == len(self):
            return np.ones(len(self), dtype=bool)

        return self._pa_array.is_null().to_numpy()

    @property
    def _hasna(self) -> bool:
        return self._pa_array.null_count > 0

    def astype(self, dtype, copy: bool = True):
        """Cast to a NumPy array or ExtensionArray with 'dtype'.

        Casting to another TensorDtype casts the storage, so the element type
        may change, and the shape may change as long as the number of
        elements is the same. Casting to pd.ArrowDtype gives an
        ArrowExtensionArray of the tensor type or of any type the storage can
        be cast to.

        Parameters
        ----------
        dtype : dtype or str
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary.

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        dtype = pandas_dtype(dtype)

        if dtype == self.dtype:
            return self.copy() if copy else self

        if isinstance(dtype, TensorDtype):
            return type(self)(self.storage, dtype=dtype)

        if isinstance(dtype, pd.ArrowDtype):
            pa_type = dtype.pyarrow_dtype
            if pa_type == self.dtype.pyarrow_dtype:
                return ArrowExtensionArray(self._pa_array)
            return ArrowExtensionArray(self.storage.cast(pa_type))

        return super().astype(dtype, copy=copy)

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
        TensorExtensionArray

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
        if indices_array.dtype.kind not in "iu":
            # E.g. an empty list gives a float array
            indices_array = indices_array.astype(np.int64)

        if len(self) == 0 and (indices_array >= 0).any():
            raise IndexError("cannot do a non-empty take from the empty array")
        if indices_array.size > 0 and indices_array.max() >= len(self):
            raise IndexError("out of bounds value in 'indices'.")

        if allow_fill:
            fill_mask = indices_array < 0
            if not fill_mask.any():
                # Nothing to fill
                return type(self)(self._pa_array.take(pa.array(indices_array)), dtype=self.dtype)
            validate_indices(indices_array, len(self))
            pa_indices = pa.array(indices_array, mask=fill_mask)

            # Null indices give null elements
            storage = self.storage.take(pa_indices)
            if not _is_na(fill_value):
                fill_storage = pa.repeat(self._scalar_storage(fill_value), int(fill_mask.sum()))
                storage = replace_with_mask(storage, pa.array(fill_mask), fill_storage)
            return type(self)(storage, dtype=self.dtype)

        if (indices_array < 0).any():
            # Don't modify in-place
            indices_array = np.copy(indices_array)
            indices_array[indices_array < 0] += len(self)
        return type(self)(self._pa_array.take(pa.array(indices_array)), dtype=self.dtype)

    def copy(self) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Return a copy of the extension array.

        This implementation returns a shallow copy of the extension array,
        because the underlying PyArrow array is immutable.
        """
        return type(self)(self._pa_array, dtype=self.dtype)

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        if boxed:
            return _format_tensor
        return repr

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:  # type: ignore[name-defined] # noqa: F821
        to_concat = list(to_concat)
        dtype = to_concat[0].dtype
        chunks = [chunk for ext_array in to_concat for chunk in ext_array._pa_array.iterchunks()]
        return cls(pa.chunked_array(chunks, type=dtype.pyarrow_dtype), dtype=dtype)

    def equals(self, other) -> bool:
        """
        Check equality with another TensorExtensionArray.

        Parameters
        ----------
        other : TensorExtensionArray
            The other TensorExtensionArray to compare with.

        Returns
        -------
        bool
            Whether the two arrays have the same dtype and the same values.
        """
        if not isinstance(other, type(self)):
            return False
        return self.dtype == other.dtype and self.storage.equals(other.storage)

    def dropna(self) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Return a new ExtensionArray with missing tensors removed."""
        return type(self)(pc.drop_null(self._pa_array), dtype=self.dtype)

    # End of ExtensionArray overrides #

    # Additional magic methods #

    def __arrow_array__(self, type=None):  # noqa: A002
        """Convert the extension array to a PyArrow array.

        With no ``type`` this is the ``fixed_shape_tensor`` extension array,
        so ``pa.Table.from_pandas`` and Parquet preserve the tensor type.
        """
        if type is None or type == self.dtype.pyarrow_dtype:
            return self._pa_array
        if isinstance(type, pa.FixedShapeTensorType):
            return self._wrap_storage(self.storage, TensorDtype(type))
        return self.storage.cast(type)

    def __array__(self, dtype=None, copy=None):
        """Convert the extension array to a numpy object array of tensors."""
        return self.to_numpy(dtype=dtype, copy=bool(copy))

    # End of Additional magic methods #

    @staticmethod
    def _scalar_to_numpy(scalar: pa.Scalar, na_value: Any = pd.NA) -> np.ndarray | Any:
        """Convert a tensor scalar to a read-only numpy view, or to na_value if it is null."""
        if not scalar.is_valid:
            return na_value
        return cast(pa.FixedShapeTensorScalar, scalar).to_numpy()

    def _is_scalar_value(self, value: Any) -> bool:
        """Whether a value passed to __setitem__ is a single tensor rather than a sequence of them."""
        if _is_na(value) or isinstance(value, pa.Scalar):
            return True
        return isinstance(value, np.ndarray) and value.shape == self.dtype.shape

    def _scalar_storage(self, value: Any) -> pa.Scalar:
        """Convert a single tensor (or missing value) to a fixed_size_list storage scalar."""
        return type(self)._from_sequence([value], dtype=self.dtype).storage[0]

    @property
    def pa_array(self) -> pa.ChunkedArray:
        """Pyarrow chunked array of the ``fixed_shape_tensor`` extension type."""
        return self._pa_array

    @property
    def storage(self) -> pa.ChunkedArray:
        """Pyarrow chunked array of the ``fixed_size_list`` storage type."""
        return _storage_of(self._pa_array)

    @property
    def num_chunks(self) -> int:
        """Number of chunks in the underlying pyarrow.ChunkedArray"""
        return self._pa_array.num_chunks

    @classmethod
    def from_sequence(
        cls, scalars, *, dtype: TensorDtype | pa.FixedShapeTensorType | pd.ArrowDtype | str | None = None
    ) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Construct a TensorExtensionArray from a sequence of tensors

        Parameters
        ----------
        scalars : Sequence
            The sequence of items: numpy arrays (or anything convertible to
            them) of the tensor shape, None, pd.NA, or pyarrow scalars. A
            numpy array of shape ``(n, *shape)`` is accepted as a stack, see
            :meth:`from_stack`.
        dtype : TensorDtype, pa.FixedShapeTensorType, pd.ArrowDtype or str, optional
            dtype of the resulting array. Inferred from the first non-missing
            element if not given.

        Returns
        -------
        TensorExtensionArray
            The constructed extension array.
        """
        return cls._from_sequence(scalars, dtype=dtype)

    @classmethod
    def from_stack(
        cls,
        stack: np.ndarray,
        *,
        dtype: TensorDtype | pa.FixedShapeTensorType | pd.ArrowDtype | str | None = None,
    ) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Construct a TensorExtensionArray from a numpy array of shape ``(n, *shape)``

        The construction is zero-copy for C-contiguous input; other layouts
        are copied into C order first.

        Parameters
        ----------
        stack : np.ndarray
            Array whose first axis enumerates the tensors.
        dtype : TensorDtype, pa.FixedShapeTensorType, pd.ArrowDtype or str, optional
            dtype of the resulting array. Inferred from the numpy dtype and
            ``stack.shape[1:]`` if not given.

        Returns
        -------
        TensorExtensionArray
            The constructed extension array.
        """
        stack = np.asarray(stack)
        if stack.ndim < 2:
            raise ValueError(f"stack must have shape (n, *shape), at least two dimensions, got {stack.shape}")

        dtype = _normalize_dtype(dtype)
        if dtype is None:
            value_type = pa.from_numpy_dtype(stack.dtype)
            dtype = TensorDtype(pa.fixed_shape_tensor(value_type, list(stack.shape[1:])))
        elif stack.shape[1:] != dtype.shape:
            raise ValueError(f"Expected a stack of tensors of shape {dtype.shape}, got {stack.shape[1:]}")

        flat = np.ascontiguousarray(stack).reshape(-1)
        storage = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=dtype.value_type), dtype.size)
        return cls(storage, dtype=dtype)

    def to_stack(self, na_value: Any = np.nan) -> np.ndarray:
        """Convert the extension array to a single numpy array of shape ``(n, *shape)``

        Without missing values the result is a read-only zero-copy view over
        the arrow buffers. With missing values the data is copied, the result
        dtype is widened as needed to hold ``na_value``, and the missing
        tensors are filled with it.

        Parameters
        ----------
        na_value : Any, default np.nan
            Value to fill missing tensors with.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(self), *self.dtype.shape)``.
        """
        if len(self) == 0:
            return np.empty((0, *self.dtype.shape), dtype=_numpy_value_dtype(self.dtype))

        # combine_chunks() copies even for a single chunk, so avoid it when we can
        if self._pa_array.num_chunks == 1:
            combined = cast(pa.FixedShapeTensorArray, self._pa_array.chunk(0))
        else:
            combined = cast(pa.FixedShapeTensorArray, self._pa_array.combine_chunks())
        stack = combined.to_numpy_ndarray()
        if combined.null_count == 0:
            return stack

        # to_numpy_ndarray() silently returns garbage for null tensors, so fill them here
        result_dtype = np.result_type(stack.dtype, np.min_scalar_type(na_value))
        result = np.array(stack, dtype=result_dtype)
        result[self.isna()] = na_value
        return result

    @classmethod
    def from_arrow_ext_array(cls, array: ArrowExtensionArray, *, dtype: TensorDtype | None = None) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Create a TensorExtensionArray from pandas' ArrowExtensionArray

        Parameters
        ----------
        array : ArrowExtensionArray
            Array of the tensor type or of its fixed_size_list storage type.
        dtype : TensorDtype, optional
            Required when ``array`` is of the storage type.
        """
        return cls(array._pa_array, dtype=dtype)

    def to_arrow_ext_array(self, storage: bool = False) -> ArrowExtensionArray:
        """Convert the extension array to pandas' ArrowExtensionArray

        Parameters
        ----------
        storage : bool, default False
            If False (default), wrap the ``fixed_shape_tensor`` extension array,
            otherwise wrap its ``fixed_size_list`` storage array.
        """
        return ArrowExtensionArray(self.storage if storage else self._pa_array)
