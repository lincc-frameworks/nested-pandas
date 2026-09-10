"""Pandas extension dtype for fixed-shape tensor columns."""

from __future__ import annotations  # Self is not available in python 3.10

import re

# We use Type, because we must use "type" as an attribute name
from typing import Type  # noqa: UP035

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.extensions import register_extension_dtype
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype

__all__ = ["TensorDtype"]


_NAME_PATTERN = re.compile(
    r"^tensor\[(?P<value_type>.+?), \((?P<shape>[\d, ]*)\)(?:, dim_names=\[(?P<dim_names>[^\]]*)\])?\]$"
)
"""Regular expression matching the strings produced by ``TensorDtype.name``"""


@register_extension_dtype
class TensorDtype(ExtensionDtype):
    """Data type for columns of fixed-shape n-dimensional arrays

    Every element of a tensor column is a numpy array with the same shape and
    element type. The data is stored with pyarrow's canonical
    ``arrow.fixed_shape_tensor`` extension type, so it round-trips through
    Arrow and Parquet with its shape preserved.

    Parameters
    ----------
    pyarrow_dtype : pyarrow.FixedShapeTensorType or pd.ArrowDtype
        The pyarrow tensor type, or a ``pd.ArrowDtype`` wrapping one.
        An identity ``permutation`` is accepted and dropped; any other
        ``permutation`` is not supported.

    Examples
    --------
    >>> import pyarrow as pa
    >>> from nested_pandas import TensorDtype

    From pa.FixedShapeTensorType:

    >>> dtype = TensorDtype(pa.fixed_shape_tensor(pa.float64(), [2, 3]))
    >>> dtype
    tensor[double, (2, 3)]

    From pd.ArrowDtype:

    >>> import pandas as pd
    >>> dtype = TensorDtype(pd.ArrowDtype(pa.fixed_shape_tensor(pa.int32(), [4], dim_names=["x"])))
    >>> dtype
    tensor[int32, (4,), dim_names=[x]]
    """

    # ExtensionDtype overrides #

    _metadata = ("pyarrow_dtype",)
    """Attributes to use as metadata for __eq__ and __hash__"""

    @property
    def na_value(self):
        """The missing value for this dtype"""
        return pd.NA  # type: ignore[return-value]

    type = np.ndarray
    """The type of the array's elements, always np.ndarray"""

    @property
    def name(self) -> str:
        """The string representation of the tensor type, e.g. 'tensor[double, (2, 3)]'"""
        parts = [str(self.pyarrow_dtype.value_type), str(tuple(self.pyarrow_dtype.shape))]
        if self.pyarrow_dtype.dim_names is not None:
            parts.append(f"dim_names=[{', '.join(self.pyarrow_dtype.dim_names)}]")
        return f"tensor[{', '.join(parts)}]"

    @name.setter
    def name(self, value: str):
        raise TypeError("name cannot be changed")

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def construct_array_type(cls) -> Type[ExtensionArray]:
        """Corresponding array type, always TensorExtensionArray"""
        # TODO: enable once TensorExtensionArray is merged
        # from nested_pandas.tensors.ext_array import TensorExtensionArray
        #
        # return TensorExtensionArray
        raise NotImplementedError("TensorExtensionArray is not implemented yet")

    @classmethod
    def construct_from_string(cls, string: str) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Construct TensorDtype from a string representation.

        The element type is parsed the same way as ``pd.ArrowDtype`` parses
        its strings, so pyarrow type aliases and common temporal types are
        supported, but other parametric types are not.

        Parameters
        ----------
        string : str
            The string representation of the tensor type. For example,
            'tensor[double, (2, 3)]' or 'tensor[int32, (4,), dim_names=[x]]'.
            It must be consistent with the string representation of the dtype
            given by the `name` attribute.

        Returns
        -------
        TensorDtype
            The constructed TensorDtype.

        Raises
        ------
        TypeError
            If the string is not a valid tensor type string, if the element
            type cannot be parsed, or if the shape and dim_names are
            inconsistent.
        """
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        match = _NAME_PATTERN.match(string)
        if match is None:
            raise TypeError(
                f"Cannot construct a '{cls.__name__}' from '{string}'. Expected a string like "
                "'tensor[<element type>, (<shape>)]', optionally followed by ', dim_names=[<names>]', "
                "for example 'tensor[double, (2, 3)]' or 'tensor[int32, (4, 4), dim_names=[y, x]]'"
            )

        # Try pyarrow type aliases first, then reuse pandas' parsing of pyarrow type strings for
        # common parametric types like timestamp[ns, tz=UTC]. Not the other way round, because
        # pandas reserves "string[pyarrow]" for its StringDtype and refuses to parse it.
        value_type_str = match["value_type"]
        try:
            value_type = pa.type_for_alias(value_type_str)
        except ValueError:
            try:
                value_type = pd.ArrowDtype.construct_from_string(f"{value_type_str}[pyarrow]").pyarrow_dtype
            except (TypeError, NotImplementedError) as e:
                raise TypeError(
                    f"Cannot parse tensor element type '{value_type_str}'. "
                    "Please use TensorDtype(pa.fixed_shape_tensor(...)) instead."
                ) from e

        shape = [int(s) for s in match["shape"].split(",") if s.strip()]
        dim_names = None if match["dim_names"] is None else [s.strip() for s in match["dim_names"].split(",")]
        try:
            return cls(pa.fixed_shape_tensor(value_type, shape, dim_names=dim_names))
        except (ValueError, pa.ArrowInvalid) as e:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}': {e}") from e

    # Optional methods #

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> ExtensionArray:
        """Construct a TensorExtensionArray from a pyarrow array.

        Parameters
        ----------
        array : pa.Array | pa.ChunkedArray
            The input pyarrow array, either of this tensor type or of its
            fixed_size_list storage type.

        Returns
        -------
        TensorExtensionArray
            The constructed TensorExtensionArray.
        """
        # TODO: enable once TensorExtensionArray is merged
        # from nested_pandas.tensors.ext_array import TensorExtensionArray
        #
        # return TensorExtensionArray(array, dtype=self)
        raise NotImplementedError("TensorExtensionArray is not implemented yet")

    # Additional methods and attributes #

    pyarrow_dtype: pa.FixedShapeTensorType

    def __init__(self, pyarrow_dtype: pa.FixedShapeTensorType | pd.ArrowDtype) -> None:
        # Allow pd.ArrowDtypes on init
        if isinstance(pyarrow_dtype, pd.ArrowDtype):
            pyarrow_dtype = pyarrow_dtype.pyarrow_dtype
        if not isinstance(pyarrow_dtype, pa.FixedShapeTensorType):
            raise TypeError(
                f"TensorDtype can only be constructed with pa.FixedShapeTensorType, got {pyarrow_dtype!r}"
            )
        permutation = pyarrow_dtype.permutation
        if permutation is not None:
            if list(permutation) == list(range(len(pyarrow_dtype.shape))):
                # An identity permutation is the same layout as no permutation, and pyarrow's own
                # FixedShapeTensorArray.from_numpy_ndarray() always sets one for C-ordered input.
                # pyarrow compares the two types equal but hashes them differently, so normalize
                # to the permutation-free type to keep TensorDtype equality and hashing consistent.
                pyarrow_dtype = pa.fixed_shape_tensor(
                    pyarrow_dtype.value_type, pyarrow_dtype.shape, dim_names=pyarrow_dtype.dim_names
                )
            else:
                # A permutation reorders the inner dimensions of each tensor in storage. Supporting it
                # would require a permutation branch in every array method, and pyarrow would return
                # non-contiguous transposed views, breaking the "every element has dtype.shape"
                # invariant. Columns with a permutation can be normalized to C order (one copy) and
                # rebuilt without it.
                raise NotImplementedError(
                    "TensorDtype does not support fixed_shape_tensor types with a non-trivial permutation "
                    f"yet, got {pyarrow_dtype}. Convert the tensors to C order with numpy and rebuild the "
                    "column from a fixed_shape_tensor type without a permutation."
                )
        self.pyarrow_dtype = pyarrow_dtype

    @property
    def value_type(self) -> pa.DataType:
        """Pyarrow type of the tensor elements."""
        return self.pyarrow_dtype.value_type

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of every tensor in the column."""
        return tuple(self.pyarrow_dtype.shape)

    @property
    def ndim(self) -> int:
        """Number of tensor dimensions."""
        return len(self.pyarrow_dtype.shape)

    @property
    def size(self) -> int:
        """Number of elements in every tensor, i.e. the fixed_size_list length."""
        return self.storage_type.list_size

    @property
    def dim_names(self) -> tuple[str, ...] | None:
        """Names of the tensor dimensions, or None if not set."""
        dim_names = self.pyarrow_dtype.dim_names
        return None if dim_names is None else tuple(dim_names)

    @property
    def storage_type(self) -> pa.FixedSizeListType:
        """Pyarrow fixed_size_list type the tensors are stored as."""
        return self.pyarrow_dtype.storage_type

    def to_pandas_arrow_dtype(self, storage: bool = False) -> pd.ArrowDtype:
        """Convert TensorDtype to a pandas.ArrowDtype.

        Parameters
        ----------
        storage : bool, default False
            If False (default) wrap the tensor extension type,
            otherwise wrap its fixed_size_list storage type.

        Returns
        -------
        pd.ArrowDtype
            The corresponding pandas.ArrowDtype.
        """
        if storage:
            return pd.ArrowDtype(self.storage_type)
        return pd.ArrowDtype(self.pyarrow_dtype)
