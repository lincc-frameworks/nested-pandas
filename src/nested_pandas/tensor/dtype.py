"""Pandas extension dtype for tensor columns."""

from __future__ import annotations

import re

# We use Type, because we must use "type" as an attribute name
from typing import Type  # noqa: UP035

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.extensions import register_extension_dtype
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype

from nested_pandas.tensor.arrow_ext import TensorType, tensor_type

__all__ = ["TensorDtype"]


def _dtype_string_pattern(prefix: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(prefix)}\[(?P<value>[^,\]]+), (?:\((?P<shape>[\d, ]*)\)|ndim=(?P<ndim>\d+))\]$")


_ARROW_TYPE_TO_DTYPE: dict[type[TensorType], type] = {}
"""Maps arrow tensor extension type classes to their TensorDtype (sub)classes."""


@register_extension_dtype
class TensorDtype(ExtensionDtype):
    """Data type for columns of n-dimensional array values.

    A tensor column stores one numpy array per row. All rows share the element
    type and the number of dimensions; the per-row shape is either fixed for
    the whole column (``shape`` provided) or may vary row to row (``ndim``
    provided). Fixed-shape columns are backed by pyarrow's canonical
    ``arrow.fixed_shape_tensor`` extension type; variable-shape columns by
    :class:`nested_pandas.tensor.arrow_ext.TensorType`.

    Parameters
    ----------
    value_type : pa.DataType, np.dtype, str, or None, default None
        The tensor element type, e.g. ``pa.float32()`` or ``"float32"``.
        Defaults to ``pa.float32()``.
    shape : tuple of int or None, default None
        The common shape of every tensor in the column. Provide for
        fixed-shape columns; mutually exclusive with ``ndim``.
    ndim : int or None, default None
        The common number of dimensions for variable-shape columns.
        Ignored when ``shape`` is provided.

    Examples
    --------
    >>> from nested_pandas import TensorDtype
    >>> TensorDtype("float32", shape=(25, 25))
    tensor[float, (25, 25)]
    >>> TensorDtype("float64", ndim=2)
    tensor[double, ndim=2]
    """

    # ExtensionDtype overrides #

    _metadata = ("value_type", "shape", "ndim")
    """Attributes to use as metadata for __eq__ and __hash__"""

    @property
    def na_value(self):
        """The missing value for this dtype"""
        return pd.NA  # type: ignore[return-value]

    type = np.ndarray
    """The type of the array's elements, always np.ndarray"""

    _name_prefix = "tensor"
    """Prefix used in the dtype string; subclasses (e.g. image dtypes) override it."""

    _arrow_type_class: type[TensorType] = TensorType
    """The arrow extension type class this dtype serializes as. Subclasses
    override it with their own :class:`TensorType` subclass (e.g. an image
    type) so the extension *name* carries their identity; overriding it
    auto-registers the pair for round-trip dispatch."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "_arrow_type_class" in cls.__dict__:
            _ARROW_TYPE_TO_DTYPE[cls._arrow_type_class] = cls

    def __init__(self, value_type=None, shape=None, ndim=None):
        if value_type is None:
            value_type = pa.float32()
        elif not isinstance(value_type, pa.DataType):
            value_type = pa.from_numpy_dtype(np.dtype(value_type))
        self._value_type = value_type

        if shape is not None:
            shape = tuple(int(size) for size in shape)
            if len(shape) == 0:
                raise ValueError("shape must have at least one dimension")
            if any(size < 0 for size in shape):
                raise ValueError(f"shape must be non-negative, got {shape}")
            self._shape: tuple[int, ...] | None = shape
            self._ndim = len(shape)
        elif ndim is not None:
            if int(ndim) < 1:
                raise ValueError(f"ndim must be a positive integer, got {ndim}")
            self._shape = None
            self._ndim = int(ndim)
        else:
            raise ValueError("either shape (fixed-shape tensors) or ndim (variable-shape) is required")

    @property
    def name(self) -> str:
        """The string representation of the tensor type"""
        if self._shape is not None:
            shape_str = ", ".join(str(size) for size in self._shape)
            return f"{self._name_prefix}[{self._value_type}, ({shape_str})]"
        return f"{self._name_prefix}[{self._value_type}, ndim={self._ndim}]"

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def construct_array_type(cls) -> Type[ExtensionArray]:  # noqa: UP006
        """Corresponding extension array type"""
        from nested_pandas.tensor.ext_array import TensorArray

        return TensorArray

    @classmethod
    def construct_from_string(cls, string: str) -> TensorDtype:
        """Construct TensorDtype from a string representation.

        Accepts the formats produced by :attr:`name`, e.g.
        ``"tensor[float, (25, 25)]"`` and ``"tensor[double, ndim=2]"``.
        """
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        match = _dtype_string_pattern(cls._name_prefix).match(string)
        if match is None:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")
        try:
            value_type = pa.type_for_alias(match["value"].strip())
        except ValueError as err:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}': {err}") from err
        if match["shape"] is not None:
            shape = tuple(int(size) for size in match["shape"].split(",") if size.strip())
            if len(shape) == 0:
                raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}': empty shape")
            return cls(value_type, shape=shape)
        return cls(value_type, ndim=int(match["ndim"]))

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray):
        """Construct a TensorArray from an arrow array (extension or storage typed)."""
        from nested_pandas.tensor.ext_array import TensorArray

        return TensorArray(array, dtype=self)

    # End of ExtensionDtype overrides #

    @property
    def value_type(self) -> pa.DataType:
        """The tensor element type."""
        return self._value_type

    @property
    def shape(self) -> tuple[int, ...] | None:
        """The common tensor shape, or None for variable-shape columns."""
        return self._shape

    @property
    def ndim(self) -> int:
        """The common number of tensor dimensions."""
        return self._ndim

    @property
    def is_fixed_shape(self) -> bool:
        """Whether every tensor in the column has the same shape."""
        return self._shape is not None

    @property
    def np_value_dtype(self) -> np.dtype:
        """The tensor element type as a numpy dtype."""
        return np.dtype(self._value_type.to_pandas_dtype())

    @property
    def pyarrow_dtype(self) -> pa.DataType:
        """The primary pyarrow extension type backing this dtype.

        Plain fixed-shape tensor dtypes map to the canonical
        ``arrow.fixed_shape_tensor``; everything else (variable shape, or a
        dtype subclass with its own arrow type) to that arrow type class.
        Note that nullable fixed-shape columns are *serialized* as the struct
        layout with a declared shape, see ``TensorArray.__arrow_array__``.
        """
        if self._shape is not None and self._arrow_type_class is TensorType and type(self) is TensorDtype:
            return pa.fixed_shape_tensor(self._value_type, self._shape)
        return self._arrow_type_class(self._value_type, self._ndim, shape=self._shape)

    @property
    def pyarrow_storage_type(self) -> pa.DataType:
        """The arrow storage layout used in memory.

        Fixed-shape columns use flat fixed-size lists regardless of kind;
        variable-shape columns the ``struct<data, shape>`` layout.
        """
        if self._shape is not None:
            return pa.fixed_shape_tensor(self._value_type, self._shape).storage_type
        return tensor_type(self._value_type, self._ndim).storage_type

    @classmethod
    def from_pyarrow(cls, pa_type: pa.DataType) -> TensorDtype:
        """Construct from a pyarrow tensor extension type.

        Parameters
        ----------
        pa_type : pa.DataType
            Either a ``pa.FixedShapeTensorType`` or a :class:`TensorType`
            (with or without a declared fixed shape). :class:`TensorType`
            subclasses resolve to their registered ``TensorDtype`` subclass,
            e.g. ``nested_pandas.image`` to the image dtype.
        """
        if isinstance(pa_type, pa.FixedShapeTensorType):
            return cls(pa_type.value_type, shape=tuple(pa_type.shape))
        if isinstance(pa_type, TensorType):
            klass = _ARROW_TYPE_TO_DTYPE.get(type(pa_type), cls)
            if pa_type.shape is not None:
                return klass(pa_type.value_type, shape=pa_type.shape)
            return klass(pa_type.value_type, ndim=pa_type.ndim)
        raise TypeError(f"Cannot construct a 'TensorDtype' from pyarrow type '{pa_type}'")


_ARROW_TYPE_TO_DTYPE[TensorType] = TensorDtype
