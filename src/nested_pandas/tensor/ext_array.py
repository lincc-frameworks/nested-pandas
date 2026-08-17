"""Pandas extension array for tensor columns."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import DTypeLike
from pandas._libs.lib import no_default
from pandas.api.extensions import ExtensionArray
from pandas.api.indexers import check_array_indexer

from nested_pandas.tensor.arrow_ext import (
    TensorType,
    build_fixed_shape_tensor_array,
    build_tensor_struct_array,
    fsl_storage_to_struct_storage,
    is_tensor_pyarrow_type,
    struct_storage_to_fsl_storage,
)
from nested_pandas.tensor.dtype import TensorDtype

__all__ = ["TensorArray"]


def _is_na_scalar(value) -> bool:
    """Whether a python-level scalar counts as a missing tensor value."""
    return value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value))


def _np_view(array: pa.Array) -> np.ndarray:
    """Convert a primitive arrow array to numpy, zero-copy when possible."""
    try:
        return array.to_numpy(zero_copy_only=True)
    except pa.ArrowInvalid:
        return array.to_numpy(zero_copy_only=False)


class TensorArray(ExtensionArray):
    """Pandas extension array of tensor (numpy ndarray) values.

    Each element is an ``np.ndarray`` (or ``pd.NA``). The data is stored in
    arrow layout: fixed-shape columns as the canonical
    ``arrow.fixed_shape_tensor`` storage (a flat fixed-size list per row),
    variable-shape columns as ``struct<data: list<T>, shape:
    fixed_size_list<int32>[ndim]>``. Scalar access returns numpy views over
    the arrow buffers whenever possible, so no pixel data is copied.

    Parameters
    ----------
    values : pa.Array or pa.ChunkedArray
        Either a tensor extension array (fixed or variable shape) or its
        storage-typed equivalent. Use :meth:`from_stack` /
        :meth:`_from_sequence` to construct from numpy data.
    dtype : TensorDtype or None
        Required when ``values`` is storage-typed (the shape cannot be
        recovered from a flat fixed-size list); otherwise inferred and
        validated against the extension type.
    """

    _storage: pa.ChunkedArray
    _dtype: TensorDtype

    def __init__(self, values: pa.Array | pa.ChunkedArray, *, dtype: TensorDtype | None = None) -> None:
        if isinstance(values, pa.Array):
            values = pa.chunked_array([values])
        if not isinstance(values, pa.ChunkedArray):
            raise TypeError(
                f"values must be a pyarrow Array or ChunkedArray, got {type(values)}. "
                "Use TensorArray.from_stack or TensorArray._from_sequence for numpy input."
            )

        if is_tensor_pyarrow_type(values.type):
            inferred = TensorDtype.from_pyarrow(values.type)
            if dtype is None:
                dtype = inferred
            # Keep the provided dtype (it may be a TensorDtype subclass, e.g.
            # an image dtype), but its tensor parameters must match the data.
            elif (dtype.value_type, dtype.shape, dtype.ndim) != (
                inferred.value_type,
                inferred.shape,
                inferred.ndim,
            ):
                raise ValueError(f"dtype {dtype} does not match array type {inferred}")
            chunks = [chunk.storage for chunk in values.chunks]
        else:
            if dtype is None:
                raise ValueError(
                    "dtype is required to construct a TensorArray from a storage-typed arrow array"
                )
            chunks = list(values.chunks)

        # In memory, fixed-shape columns always use fixed-size-list storage;
        # convert struct-typed chunks (the nullable-serialization layout) back.
        storage_type = dtype.pyarrow_dtype.storage_type
        struct_type = TensorType(dtype.value_type, dtype.ndim).storage_type
        normalized = []
        for chunk in chunks:
            if dtype.is_fixed_shape and pa.types.is_struct(chunk.type):
                chunk = struct_storage_to_fsl_storage(
                    chunk.cast(struct_type), dtype.shape, dtype.value_type
                )
            if not chunk.type.equals(storage_type):
                chunk = chunk.cast(storage_type)
            normalized.append(chunk)

        self._storage = pa.chunked_array(normalized, type=storage_type)
        self._dtype = dtype

    @classmethod
    def _from_storage(cls, storage: pa.Array | pa.ChunkedArray, dtype: TensorDtype) -> TensorArray:
        """Wrap a storage-typed arrow array without validation-by-cast."""
        return cls(storage, dtype=dtype)

    # ExtensionArray overrides #

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy: bool = False) -> TensorArray:
        """Construct a TensorArray from a sequence of tensors.

        Parameters
        ----------
        scalars : Sequence
            Array-likes (anything ``np.asarray`` accepts) and missing values
            (``None``, ``pd.NA``, ``np.nan``).
        dtype : TensorDtype or None
            The target dtype. When None, it is inferred: a common shape gives
            a fixed-shape dtype, a common ndim a variable-shape dtype.
        copy : bool
            Ignored; the data is always copied into arrow buffers.
        """
        del copy
        if isinstance(scalars, pa.Array | pa.ChunkedArray):
            return cls(scalars, dtype=dtype if isinstance(dtype, TensorDtype) else None)

        tensors = [scalar if _is_na_scalar(scalar) else np.asarray(scalar) for scalar in scalars]

        if dtype is None:
            dtype = cls._infer_dtype(tensors)
        elif not isinstance(dtype, TensorDtype):
            raise TypeError(f"dtype must be a TensorDtype, got {dtype}")

        if dtype.is_fixed_shape:
            pa_array = build_fixed_shape_tensor_array(tensors, dtype.value_type, dtype.shape)
        else:
            pa_array = build_tensor_struct_array(tensors, dtype.value_type, dtype.ndim)
        return cls(pa_array, dtype=dtype)

    @classmethod
    def _infer_dtype(cls, tensors: list) -> TensorDtype:
        """Infer a TensorDtype from a list of numpy tensors and NA markers."""
        valid = [tensor for tensor in tensors if not _is_na_scalar(tensor)]
        if not valid:
            raise ValueError(
                "Cannot infer a TensorDtype from all-missing or empty data; pass dtype explicitly"
            )
        value_type = pa.from_numpy_dtype(np.result_type(*(tensor.dtype for tensor in valid)))
        shapes = {tensor.shape for tensor in valid}
        if len(shapes) == 1:
            return TensorDtype(value_type, shape=shapes.pop())
        ndims = {tensor.ndim for tensor in valid}
        if len(ndims) == 1:
            return TensorDtype(value_type, ndim=ndims.pop())
        raise ValueError(f"Cannot infer a TensorDtype from tensors with mixed ndim: {sorted(ndims)}")

    @classmethod
    def _from_factorized(cls, values, original):
        raise NotImplementedError("TensorArray does not support factorization")

    def __getitem__(self, item):
        if isinstance(item, int | np.integer):
            index = int(item)
            if index < 0:
                index += len(self)
            if not 0 <= index < len(self):
                raise IndexError(f"index {item} is out of bounds for array of length {len(self)}")
            return self._scalar_at(index)

        item = check_array_indexer(self, item)
        if isinstance(item, slice):
            if item.step is None or item.step == 1:
                start, stop, _ = item.indices(len(self))
                return self._from_storage(
                    self._storage.slice(start, max(stop - start, 0)), self._dtype
                )
            item = np.arange(len(self))[item]

        item = np.asarray(item)
        if item.dtype == bool:
            return self._from_storage(self._storage.filter(pa.array(item)), self._dtype)
        return self.take(item)

    def __setitem__(self, key, value) -> None:
        key = check_array_indexer(self, key)
        indices = np.arange(len(self))[key]
        scalars = list(self.to_numpy(na_value=None))

        if _is_na_scalar(value) or isinstance(value, np.ndarray) and value.ndim == self._dtype.ndim:
            for index in np.atleast_1d(indices):
                scalars[index] = value
        else:
            values = list(value)
            indices = np.atleast_1d(indices)
            if len(values) != len(indices):
                raise ValueError(f"cannot set {len(indices)} values with {len(values)} items")
            for index, item in zip(indices, values, strict=True):
                scalars[index] = item

        new_array = self._from_sequence(scalars, dtype=self._dtype)
        self._storage = new_array._storage

    def __len__(self) -> int:
        return len(self._storage)

    def __iter__(self) -> Iterator:
        for index in range(len(self)):
            yield self._scalar_at(index)

    def __eq__(self, other):
        # Elementwise equality is ambiguous for array-valued cells; pandas
        # relies on identity-style behavior here, like for struct arrays.
        return super().__eq__(other)

    @property
    def dtype(self) -> TensorDtype:
        """ExtensionArray dtype"""
        return self._dtype

    @property
    def nbytes(self) -> int:
        """Number of bytes consumed by the data in memory."""
        return self._storage.nbytes

    def isna(self) -> np.ndarray:
        """Boolean numpy array indicating missing values."""
        null_count = self._storage.null_count
        if null_count == 0:
            return np.zeros(len(self), dtype=bool)
        if null_count == len(self):
            return np.ones(len(self), dtype=bool)
        return self._storage.is_null().to_numpy(zero_copy_only=False)

    @property
    def _hasna(self) -> bool:
        return self._storage.null_count > 0

    def take(self, indices, *, allow_fill: bool = False, fill_value: Any = None) -> TensorArray:
        """Take elements from the array; see ExtensionArray.take."""
        indices_array = np.asanyarray(indices)

        if len(self) == 0 and (indices_array >= 0).any():
            raise IndexError("cannot do a non-empty take from the empty array")
        if indices_array.size > 0 and indices_array.max() >= len(self):
            raise IndexError("out of bounds value in 'indices'.")

        if allow_fill:
            fill_mask = indices_array < 0
            if (indices_array < -1).any():
                raise ValueError("Invalid value in 'indices'. Must be all >= -1 for 'allow_fill'")
            if not fill_mask.any():
                return self._from_storage(self._storage.take(indices_array), self._dtype)
            pa_indices = pa.array(indices_array, mask=fill_mask)
            result = self._from_storage(self._storage.take(pa_indices), self._dtype)
            if not _is_na_scalar(fill_value):
                result[fill_mask] = fill_value
            return result

        if (indices_array < 0).any():
            indices_array = np.copy(indices_array)
            indices_array[indices_array < 0] += len(self)
        return self._from_storage(self._storage.take(indices_array), self._dtype)

    def copy(self) -> TensorArray:
        """Return a shallow copy; the underlying arrow data is immutable."""
        return self._from_storage(self._storage, self._dtype)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[TensorArray]) -> TensorArray:
        dtypes = {array.dtype for array in to_concat}
        if len(dtypes) > 1:
            raise TypeError(f"Cannot concatenate TensorArrays with different dtypes: {dtypes}")
        chunks = [chunk for array in to_concat for chunk in array._storage.chunks]
        dtype = to_concat[0].dtype
        storage = pa.chunked_array(chunks, type=dtype.pyarrow_dtype.storage_type)
        return cls._from_storage(storage, dtype)

    def dropna(self) -> TensorArray:
        """Return a new TensorArray with missing values removed."""
        return self._from_storage(self._storage.drop_null(), self._dtype)

    def to_numpy(
        self, dtype: DTypeLike | None = None, copy: bool = False, na_value: Any = no_default
    ) -> np.ndarray:
        """Convert to a 1-d object numpy array of per-row ndarrays.

        Use :meth:`to_stack` for a single (n, ...) numpy block instead.
        """
        del dtype, copy
        if na_value is no_default:
            na_value = None
        result = np.empty(len(self), dtype=object)
        for index in range(len(self)):
            value = self._scalar_at(index)
            result[index] = na_value if value is pd.NA else value
        return result

    def __array__(self, dtype=None, copy=None):
        del copy
        return self.to_numpy(dtype=dtype)

    def astype(self, dtype, copy: bool = True):
        if isinstance(dtype, TensorDtype):
            if dtype == self._dtype:
                return self.copy() if copy else self
            return self._from_sequence(self.to_numpy(na_value=pd.NA), dtype=dtype)
        return super().astype(dtype, copy=copy)

    def equals(self, other) -> bool:
        """Whether this array equals another, including dtype and missing values."""
        if not isinstance(other, type(self)):
            return False
        return self._dtype == other._dtype and self._storage == other._storage

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        del boxed

        def formatter(value):
            if value is pd.NA or value is None:
                return str(pd.NA)
            shape = "×".join(str(size) for size in value.shape)
            return f"[{shape}] {value.dtype}"

        return formatter

    # End of ExtensionArray overrides #

    def __arrow_array__(self, type=None):  # noqa: A002
        """Convert to a pyarrow extension array (used e.g. by parquet writing).

        Fixed-shape columns without missing values become the canonical
        ``arrow.fixed_shape_tensor``. Fixed-shape columns *with* missing
        values become :class:`TensorType` with a declared shape, because the
        pyarrow parquet reader cannot reconstruct fixed-size lists with null
        slots; the declared shape brings the column back as the same
        fixed-shape dtype on read.
        """
        storage = self._combined_storage()
        if self._dtype.is_fixed_shape and storage.null_count > 0:
            extension_type = TensorType(self._dtype.value_type, self._dtype.ndim, shape=self._dtype.shape)
            storage = fsl_storage_to_struct_storage(storage, self._dtype.shape)
        else:
            extension_type = self._dtype.pyarrow_dtype
        extension = pa.ExtensionArray.from_storage(extension_type, storage)
        if type is not None and not extension.type.equals(type):
            if extension.type.storage_type.equals(type):
                return storage
            raise ValueError(f"Cannot convert TensorArray of type {extension.type} to {type}")
        return extension

    # Tensor-specific API #

    @classmethod
    def from_stack(cls, stack: np.ndarray, value_type: pa.DataType | None = None) -> TensorArray:
        """Construct a fixed-shape TensorArray from an (n, ...) numpy block.

        Parameters
        ----------
        stack : np.ndarray
            Array of at least two dimensions; the first indexes rows.
        value_type : pa.DataType or None
            The tensor element type; defaults to the numpy dtype of ``stack``.
        """
        stack = np.asarray(stack)
        if stack.ndim < 2:
            raise ValueError(f"stack must have at least 2 dimensions, got {stack.ndim}")
        if value_type is None:
            value_type = pa.from_numpy_dtype(stack.dtype)
        dtype = TensorDtype(value_type, shape=stack.shape[1:])
        stack = np.ascontiguousarray(stack, dtype=dtype.np_value_dtype)
        return cls(pa.FixedShapeTensorArray.from_numpy_ndarray(stack), dtype=dtype)

    def to_stack(self, na_value: Any = np.nan) -> np.ndarray:
        """Convert to a single (n, ...) numpy block.

        For fixed-shape columns without missing values this is zero-copy.
        Missing rows are filled with ``na_value`` (which forces a float
        result dtype for integer tensors when NaN). Variable-shape columns
        are only stackable when every row happens to share one shape.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(self), *tensor_shape)``.
        """
        shape = self._dtype.shape
        if shape is None:
            shapes = {tuple(row_shape) for row_shape in self.shapes[~self.isna()]}
            if len(shapes) > 1:
                raise ValueError(f"Cannot stack variable-shape tensors with shapes {sorted(shapes)}")
            if not shapes:
                raise ValueError("Cannot stack an all-missing variable-shape tensor array")
            shape = shapes.pop()

        storage = self._combined_storage()
        flat_valid = self._flat_valid_values(storage)

        if storage.null_count == 0:
            return flat_valid.reshape(len(self), *shape)

        result_dtype = np.result_type(self._dtype.np_value_dtype, np.asarray(na_value).dtype)
        result = np.full((len(self), *shape), na_value, dtype=result_dtype)
        result[~self.isna()] = flat_valid.reshape(-1, *shape)
        return result

    def _combined_storage(self) -> pa.Array:
        """The storage as a single arrow array, avoiding a copy when single-chunked."""
        if self._storage.num_chunks == 1:
            return self._storage.chunk(0)
        return self._storage.combine_chunks()

    def _flat_valid_values(self, storage: pa.Array) -> np.ndarray:
        """Row-major flattened values of all non-null rows."""
        valid = storage.drop_null() if storage.null_count else storage
        if self._dtype.is_fixed_shape:
            return _np_view(valid.flatten())
        return _np_view(valid.field("data").flatten())

    @property
    def shapes(self) -> np.ndarray:
        """Per-row tensor shapes as an (n, ndim) integer array (zeros for missing rows)."""
        if self._dtype.is_fixed_shape:
            return np.broadcast_to(self._dtype.shape, (len(self), self._dtype.ndim)).copy()
        storage = self._combined_storage()
        if storage.null_count == 0:
            flat = storage.field("shape").flatten()
            return flat.to_numpy(zero_copy_only=False).reshape(len(self), self._dtype.ndim)
        result = np.zeros((len(self), self._dtype.ndim), dtype=np.int32)
        valid_flat = storage.drop_null().field("shape").flatten()
        result[~self.isna()] = valid_flat.to_numpy(zero_copy_only=False).reshape(-1, self._dtype.ndim)
        return result

    def _scalar_at(self, index: int) -> np.ndarray | Any:
        """The tensor at a flat position, as a numpy view when possible."""
        scalar = self._storage[index]
        if not scalar.is_valid:
            return self._dtype.na_value
        if self._dtype.is_fixed_shape:
            return _np_view(scalar.values).reshape(self._dtype.shape)
        shape = tuple(scalar["shape"].as_py())
        return _np_view(scalar["data"].values).reshape(shape)
