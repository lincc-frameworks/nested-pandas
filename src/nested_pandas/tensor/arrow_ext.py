"""Arrow-level types and builders backing tensor columns.

Fixed-shape tensor columns without missing values are serialized with
pyarrow's built-in canonical ``arrow.fixed_shape_tensor`` extension type, so
files written by nested-pandas are readable by any Arrow implementation that
knows the canonical type.

Everything else — variable-shape (ragged) tensors, and fixed-shape tensors
with missing rows — uses :class:`TensorType` below, with storage
``struct<data: list<value_type>, shape: list<int32>>``. This deviates from the
Arrow canonical ``arrow.variable_shape_tensor`` storage (which keeps ``shape``
as ``fixed_size_list<int32>[ndim]``) because the pyarrow parquet reader cannot
reconstruct fixed-size lists with null slots ("Expected all lists to be of
size=N", apache/arrow#34510 family); a plain list has no such problem. The
same reader limitation is why nullable fixed-shape columns cannot be
serialized with the canonical type either.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa

__all__ = [
    "TensorType",
    "tensor_type",
    "is_tensor_pyarrow_type",
    "build_fixed_shape_tensor_array",
    "build_tensor_struct_array",
    "fsl_storage_to_struct_storage",
    "struct_storage_to_fsl_storage",
]


class TensorType(pa.ExtensionType):
    """Arrow extension type for tensor values with per-row shapes.

    Storage is ``struct<data: list<value_type>, shape: list<int32>>``, where
    ``data`` holds the row-major flattened tensor values and ``shape`` the
    per-row dimensions.

    Parameters
    ----------
    value_type : pa.DataType
        The type of the tensor elements, e.g. ``pa.float32()``.
    ndim : int
        The number of tensor dimensions, fixed for the whole column.
    shape : tuple of int or None, default None
        When given, every row is declared to have this shape: the column
        round-trips to a fixed-shape ``TensorDtype``. This is how nullable
        fixed-shape columns are serialized, since the canonical
        ``arrow.fixed_shape_tensor`` cannot hold nulls in parquet.
    kind : str or None, default None
        Semantic tag carried in the type metadata, so ``TensorDtype``
        subclasses (e.g. image dtypes, ``kind="image"``) keep their identity
        through serialization. None means a plain tensor column.
    """

    def __init__(
        self,
        value_type: pa.DataType,
        ndim: int,
        shape: tuple[int, ...] | None = None,
        kind: str | None = None,
    ):
        if ndim < 1:
            raise ValueError(f"ndim must be a positive integer, got {ndim}")
        if shape is not None:
            shape = tuple(int(size) for size in shape)
            if len(shape) != ndim:
                raise ValueError(f"shape {shape} does not match ndim {ndim}")
        self._value_type = value_type
        self._ndim = int(ndim)
        self._shape = shape
        self._kind = kind
        storage_type = pa.struct(
            [
                pa.field("data", pa.list_(value_type)),
                pa.field("shape", pa.list_(pa.int32())),
            ]
        )
        super().__init__(storage_type, "nested_pandas.tensor")

    @property
    def value_type(self) -> pa.DataType:
        """The type of the tensor elements."""
        return self._value_type

    @property
    def ndim(self) -> int:
        """The number of tensor dimensions."""
        return self._ndim

    @property
    def shape(self) -> tuple[int, ...] | None:
        """The declared fixed shape, or None for variable-shape columns."""
        return self._shape

    @property
    def kind(self) -> str | None:
        """The semantic tag of the column, or None for a plain tensor."""
        return self._kind

    def __arrow_ext_serialize__(self) -> bytes:
        metadata: dict = {
            "ndim": self._ndim,
            "shape": list(self._shape) if self._shape is not None else None,
        }
        if self._kind is not None:
            metadata["kind"] = self._kind
        return json.dumps(metadata).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        metadata = json.loads(serialized.decode())
        value_type = storage_type.field("data").type.value_type
        shape = metadata.get("shape")
        return cls(
            value_type,
            metadata["ndim"],
            shape=tuple(shape) if shape is not None else None,
            kind=metadata.get("kind"),
        )

    def to_pandas_dtype(self):
        from nested_pandas.tensor.dtype import TensorDtype

        return TensorDtype.from_pyarrow(self)


def tensor_type(
    value_type: pa.DataType, ndim: int, shape: tuple[int, ...] | None = None, kind: str | None = None
) -> TensorType:
    """Create a :class:`TensorType` instance."""
    return TensorType(value_type, ndim, shape=shape, kind=kind)


def is_tensor_pyarrow_type(pa_type: pa.DataType) -> bool:
    """Whether the pyarrow type is a tensor extension type (fixed or variable shape)."""
    return isinstance(pa_type, pa.FixedShapeTensorType | TensorType)


def _is_na(value) -> bool:
    """Whether a scalar counts as a missing tensor value."""
    import pandas as pd

    return value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value))


def build_fixed_shape_tensor_array(tensors, value_type: pa.DataType, shape: tuple[int, ...]) -> pa.Array:
    """Build an ``arrow.fixed_shape_tensor`` extension array from numpy tensors.

    Parameters
    ----------
    tensors : sequence of np.ndarray or None
        The tensors, all of the given shape; ``None``/``pd.NA``/``nan`` mark
        missing values.
    value_type : pa.DataType
        The tensor element type; values are cast to it.
    shape : tuple of int
        The common tensor shape.

    Returns
    -------
    pa.Array
        An extension array of type ``pa.fixed_shape_tensor(value_type, shape)``.
    """
    tensor_type_ = pa.fixed_shape_tensor(value_type, shape)
    np_value = np.dtype(value_type.to_pandas_dtype())

    arrays: list[np.ndarray | None] = []
    has_na = False
    for tensor in tensors:
        if _is_na(tensor):
            arrays.append(None)
            has_na = True
            continue
        array = np.asarray(tensor, dtype=np_value)
        if array.shape != tuple(shape):
            raise ValueError(f"tensor of shape {array.shape} does not match dtype shape {tuple(shape)}")
        arrays.append(array)

    if not has_na and arrays:
        stack = np.ascontiguousarray(np.stack(arrays))
        return pa.FixedShapeTensorArray.from_numpy_ndarray(stack)

    # Slow path for missing values (or an empty input): build the flat
    # fixed-size-list storage from python lists, then attach the tensor type.
    cells = [None if array is None else array.ravel().tolist() for array in arrays]
    storage = pa.array(cells, type=tensor_type_.storage_type)
    return pa.ExtensionArray.from_storage(tensor_type_, storage)


def build_tensor_struct_array(
    tensors, value_type: pa.DataType, ndim: int, shape: tuple[int, ...] | None = None
) -> pa.Array:
    """Build a :class:`TensorType` extension array from numpy tensors.

    Parameters
    ----------
    tensors : sequence of np.ndarray or None
        The tensors, all with the given number of dimensions;
        ``None``/``pd.NA``/``nan`` mark missing values.
    value_type : pa.DataType
        The tensor element type; values are cast to it.
    ndim : int
        The common number of tensor dimensions.
    shape : tuple of int or None, default None
        When given, validate every tensor against it and declare the type as
        fixed shape.

    Returns
    -------
    pa.Array
        An extension array of type ``TensorType(value_type, ndim, shape)``.
    """
    tensor_type_ = TensorType(value_type, ndim, shape=shape)
    np_value = np.dtype(value_type.to_pandas_dtype())

    flats: list[np.ndarray] = []
    data_offsets = [0]
    shape_values: list[tuple[int, ...]] = []
    shape_offsets = [0]
    mask = np.zeros(len(tensors), dtype=bool)
    for i, tensor in enumerate(tensors):
        if _is_na(tensor):
            mask[i] = True
            data_offsets.append(data_offsets[-1])
            shape_offsets.append(shape_offsets[-1])
            continue
        array = np.ascontiguousarray(tensor, dtype=np_value)
        if array.ndim != ndim:
            raise ValueError(f"tensor with {array.ndim} dimensions does not match dtype ndim {ndim}")
        if shape is not None and array.shape != shape:
            raise ValueError(f"tensor of shape {array.shape} does not match dtype shape {shape}")
        flats.append(array.ravel())
        data_offsets.append(data_offsets[-1] + array.size)
        shape_values.append(array.shape)
        shape_offsets.append(shape_offsets[-1] + ndim)

    flat_values = np.concatenate(flats) if flats else np.empty(0, dtype=np_value)
    flat_shapes = np.asarray(shape_values, dtype=np.int32).ravel()
    pa_mask = pa.array(mask) if mask.any() else None
    data = pa.ListArray.from_arrays(
        pa.array(data_offsets, type=pa.int32()), pa.array(flat_values, type=value_type), mask=pa_mask
    )
    shape_arr = pa.ListArray.from_arrays(
        pa.array(shape_offsets, type=pa.int32()), pa.array(flat_shapes, type=pa.int32()), mask=pa_mask
    )
    storage = pa.StructArray.from_arrays(
        [data, shape_arr], names=["data", "shape"], mask=pa.array(mask)
    )
    return pa.ExtensionArray.from_storage(tensor_type_, storage)


def fsl_storage_to_struct_storage(storage: pa.Array, shape: tuple[int, ...]) -> pa.StructArray:
    """Convert fixed-size-list tensor storage to the struct storage of :class:`TensorType`.

    Used to serialize nullable fixed-shape columns, which the canonical
    ``arrow.fixed_shape_tensor`` type cannot represent in parquet.
    """
    size = int(np.prod(shape))
    ndim = len(shape)
    is_null = storage.is_null().to_numpy(zero_copy_only=False)
    n_valid = len(storage) - int(is_null.sum())

    data_offsets = np.zeros(len(storage) + 1, dtype=np.int32)
    np.cumsum(np.where(is_null, 0, size), out=data_offsets[1:])
    shape_offsets = np.zeros(len(storage) + 1, dtype=np.int32)
    np.cumsum(np.where(is_null, 0, ndim), out=shape_offsets[1:])

    valid = storage.drop_null() if is_null.any() else storage
    flat_values = valid.flatten()
    flat_shapes = np.tile(np.asarray(shape, dtype=np.int32), n_valid)

    pa_mask = pa.array(is_null) if is_null.any() else None
    data = pa.ListArray.from_arrays(pa.array(data_offsets), flat_values, mask=pa_mask)
    shape_arr = pa.ListArray.from_arrays(
        pa.array(shape_offsets), pa.array(flat_shapes, type=pa.int32()), mask=pa_mask
    )
    return pa.StructArray.from_arrays(
        [data, shape_arr], names=["data", "shape"], mask=pa.array(is_null) if is_null.any() else None
    )


def struct_storage_to_fsl_storage(
    storage: pa.Array, shape: tuple[int, ...], value_type: pa.DataType
) -> pa.Array:
    """Convert :class:`TensorType` struct storage to fixed-size-list tensor storage.

    Zero-copy for columns without missing values; missing rows require
    allocating the full flat block to keep fixed-size-list slot alignment.
    """
    size = int(np.prod(shape))
    if storage.null_count == 0:
        flat_values = storage.field("data").flatten()
        return pa.FixedSizeListArray.from_arrays(flat_values, size)

    is_null = storage.is_null().to_numpy(zero_copy_only=False)
    np_value = np.dtype(value_type.to_pandas_dtype())
    flat = np.zeros(len(storage) * size, dtype=np_value)
    valid_flat = storage.drop_null().field("data").flatten().to_numpy(zero_copy_only=False)
    flat[np.repeat(~is_null, size)] = valid_flat
    return pa.FixedSizeListArray.from_arrays(
        pa.array(flat, type=value_type), size, mask=pa.array(is_null)
    )


def _register_extension_types():
    """Register the tensor extension type with pyarrow (idempotent)."""
    try:
        pa.register_extension_type(TensorType(pa.float32(), 1))
    except pa.ArrowKeyError:
        # Already registered, e.g. by a reloaded module.
        pass


_register_extension_types()
