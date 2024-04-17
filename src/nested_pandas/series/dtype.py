# Use Self, which is not available until Python 3.11
from __future__ import annotations

from collections.abc import Mapping

# We use Type, because we must use "type" as an attribute name
from typing import Type, cast  # noqa: UP035

import pandas as pd
import pyarrow as pa
from pandas import ArrowDtype
from pandas.api.extensions import register_extension_dtype
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype

from nested_pandas.series.utils import is_pa_type_a_list

__all__ = ["NestedDtype"]


@register_extension_dtype
class NestedDtype(ExtensionDtype):
    """Data type to handle packed time series data"""

    # ExtensionDtype overrides #

    _metadata = ("pyarrow_dtype",)
    """Attributes to use as metadata for __eq__ and __hash__"""

    @property
    def na_value(self) -> Type[pd.NA]:
        """The missing value for this dtype"""
        return pd.NA

    type = pd.DataFrame
    """The type of the array's elements, always pd.DataFrame"""

    @property
    def name(self) -> str:
        """The string representation of the nested type"""
        fields = ", ".join([f"{field.name}: [{field.type.value_type!s}]" for field in self.pyarrow_dtype])
        return f"nested<{fields}>"

    @classmethod
    def construct_array_type(cls) -> Type[ExtensionArray]:
        """Corresponded array type, always NestedExtensionArray"""
        from nested_pandas.series.ext_array import NestedExtensionArray

        return NestedExtensionArray

    @classmethod
    def construct_from_string(cls, string: str) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Construct NestedDtype from a string representation.

        This works only for simple types, i.e. non-parametric pyarrow types.

        Parameters
        ----------
        string : str
            The string representation of the nested type. For example,
            'nested<x: [int64], y: [float64]'. It must be consistent with
            the string representation of the dtype given by the `name`
            attribute.

        Returns
        -------
        NestedDtype
            The constructed NestedDtype.

        Raises
        ------
        TypeError
            If the string is not a valid nested type string or if the element types
            are parametric pyarrow types.
        """
        if not string.startswith("nested<") or not string.endswith(">"):
            raise TypeError("Not a valid nested type string, expected 'nested<...>'")
        fields_str = string.removeprefix("nested<").removesuffix(">")

        field_strings = fields_str.split(", ")

        fields = {}
        for field_string in field_strings:
            try:
                field_name, field_type = field_string.split(": ", maxsplit=1)
            except ValueError as e:
                raise TypeError(
                    "Not a valid nested type string, expected 'nested<x: [type], ...>', got invalid field "
                    f"string '{field_string}'"
                ) from e
            if not field_type.startswith("[") or not field_type.endswith("]"):
                raise TypeError(
                    "Not a valid nested type string, expected 'nested<x: [type], ...>', got invalid field "
                    f"type string '{field_type}'"
                )

            value_type = field_type.removeprefix("[").removesuffix("]")
            # We follow ArrowDtype implementation heere and do not try to parse complex types
            try:
                pa_value_type = pa.type_for_alias(value_type)
            except ValueError as e:
                raise TypeError(
                    f"Parsing pyarrow specific parameters in the string is not supported yet: {value_type}. "
                    "Please use NestedDtype() or NestedDtype.from_fields() instead."
                ) from e

            fields[field_name] = pa_value_type

        return cls.from_fields(fields)

    # ArrowDtype would return None so we do
    def _get_common_dtype(self, dtypes: list) -> None:
        return None

    # Optional methods #

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> ExtensionArray:
        """Construct a NestedExtensionArray from a pyarrow array.

        Parameters
        ----------
        array : pa.Array | pa.ChunkedArray
            The input pyarrow array.

        Returns
        -------
        NestedExtensionArray
            The constructed NestedExtensionArray.
        """
        from nested_pandas.series.ext_array import NestedExtensionArray

        return NestedExtensionArray(array)

    # Additional methods and attributes #

    pyarrow_dtype: pa.StructType

    def __init__(self, pyarrow_dtype: pa.DataType) -> None:
        self.pyarrow_dtype = self._validate_dtype(pyarrow_dtype)

    @classmethod
    def from_fields(cls, fields: Mapping[str, pa.DataType]) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Make NestedDtype from a mapping of field names and list item types.

        Parameters
        ----------
        fields : Mapping[str, pa.DataType]
            A mapping of field names and their item types. Since all fields are lists, the item types are
            inner types of the lists, not the list types themselves.

        Returns
        -------
        NestedDtype
            The constructed NestedDtype.

        Examples
        --------
        >>> dtype = NestedDtype.from_fields({"a": pa.float64(), "b": pa.int64()})
        >>> dtype
        nested<a: [double], b: [int64]>
        >>> assert (
        ...     dtype.pyarrow_dtype
        ...     == pa.struct({"a": pa.list_(pa.float64()), "b": pa.list_(pa.int64())})
        ... )
        """
        pyarrow_dtype = pa.struct({field: pa.list_(pa_type) for field, pa_type in fields.items()})
        pyarrow_dtype = cast(pa.StructType, pyarrow_dtype)
        return cls(pyarrow_dtype=pyarrow_dtype)

    @staticmethod
    def _validate_dtype(pyarrow_dtype: pa.DataType) -> pa.StructType:
        if not isinstance(pyarrow_dtype, pa.DataType):
            raise TypeError(f"Expected a 'pyarrow.DataType' object, got {type(pyarrow_dtype)}")
        if not pa.types.is_struct(pyarrow_dtype):
            raise ValueError("NestedDtype can only be constructed with pyarrow struct type.")
        pyarrow_dtype = cast(pa.StructType, pyarrow_dtype)

        for field in pyarrow_dtype:
            if not is_pa_type_a_list(field.type):
                raise ValueError(
                    "NestedDtype can only be constructed with pyarrow struct type, all fields must be list "
                    f"type. Given struct has unsupported field {field}"
                )
        return pyarrow_dtype

    @property
    def fields(self) -> dict[str, pa.DataType]:
        """The mapping of field names and their item types."""
        return {field.name: field.type.value_type for field in self.pyarrow_dtype}

    @property
    def field_names(self) -> list[str]:
        """The list of field names of the nested type"""
        return [field.name for field in self.pyarrow_dtype]

    @classmethod
    def from_pandas_arrow_dtype(cls, pandas_arrow_dtype: ArrowDtype):
        """Construct NestedDtype from a pandas.ArrowDtype.

        Parameters
        ----------
        pandas_arrow_dtype : ArrowDtype
            The pandas.ArrowDtype to construct NestedDtype from.

        Returns
        -------
        NestedDtype
            The constructed NestedDtype.

        Raises
        ------
        ValueError
            If the given dtype is not a valid nested type.
        """
        return cls(pyarrow_dtype=pandas_arrow_dtype.pyarrow_dtype)

    def to_pandas_arrow_dtype(self) -> ArrowDtype:
        """Convert NestedDtype to a pandas.ArrowDtype.

        Returns
        -------
        ArrowDtype
            The corresponding pandas.ArrowDtype.
        """
        return ArrowDtype(self.pyarrow_dtype)
