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
    """Data type to handle packed time series data

    Parameters
    ----------
    pyarrow_dtype : pyarrow.StructType or pd.ArrowDtype
        The pyarrow data type to use for the nested type. It must be a struct
        type where all fields are list types.
    inner_dtypes : Mapping[str, object] or None, default None
        A mapping of field names and their inner types. This will be used to:
        1. Cast to the correct types when getting flat representations
           of the nested fields.
        2. To handle information of the double-nested fields, you should use
           this NestedDtype for the inner types in this case.
        Dtypes must be pandas-recognisable types, such as Python native types,
        numpy dtypes or extension array dtypes. Please wrap pyarrow types with
        pd.ArrowDtype.
        We trust these dtypes and make no attempt to validate them when
        casting.
        If None, all inner types are assumed to be the same as the
        corresponding list element types.
    """

    # ExtensionDtype overrides #

    _metadata = (
        "pyarrow_dtype",
        "inner_dtypes",
    )
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
        # Replace pd.ArrowDtype with pa.DataType, because it has nicer __str__
        nice_dtypes = {
            field: dtype.pyarrow_dtype if isinstance(dtype, pd.ArrowDtype) else dtype
            for field, dtype in self.fields.items()
        }
        fields = ", ".join([f"{field}: [{dtype!s}]" for field, dtype in nice_dtypes.items()])
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
    inner_dtypes: dict[str, object]

    def __init__(
        self, pyarrow_dtype: pa.DataType | pd.ArrowDtype, inner_dtypes: Mapping[str, object] | None = None
    ) -> None:
        if isinstance(pyarrow_dtype, pd.ArrowDtype):
            pyarrow_dtype = pyarrow_dtype.pyarrow_dtype
        self.pyarrow_dtype = self._validate_dtype(pyarrow_dtype)
        self.inner_dtypes = self._validate_inner_dtypes(self.pyarrow_dtype, inner_dtypes)

    @classmethod
    def from_fields(cls, fields: Mapping[str, pa.DataType | pa.ArrowDtype | Self]) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Make NestedDtype from a mapping of field names and list item types.

        Parameters
        ----------
        fields : Mapping[str, pa.DataType | NestedDtype]
            A mapping of field names and their item types. Since all fields are
            lists, the item types are inner types of the lists, not the list
            types themselves.

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
        pa_fields = {}
        inner_dtypes = {}
        for field, dtype in fields.items():
            if isinstance(dtype, NestedDtype):
                inner_dtypes[field] = dtype
                dtype = dtype.pyarrow_dtype
            elif isinstance(dtype, pd.ArrowDtype):
                dtype = dtype.pyarrow_dtype
            pa_fields[field] = dtype
        pyarrow_dtype = pa.struct({field: pa.list_(pa_type) for field, pa_type in pa_fields.items()})
        return cls(pyarrow_dtype=pyarrow_dtype, inner_dtypes=inner_dtypes or None)

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

    @staticmethod
    def _validate_inner_dtypes(
        pyarrow_dtype: pa.StructType, inner_dtypes: Mapping[str, object] | None
    ) -> dict[str, object]:
        # Short circuit if there are no inner dtypes
        if inner_dtypes is None or len(inner_dtypes) == 0:
            return {}

        inner_dtypes = dict(inner_dtypes)

        for field_name, inner_dtype in inner_dtypes.items():
            if field_name not in pyarrow_dtype.names:
                raise ValueError(f"Field '{field_name}' not found in the pyarrow struct type.")
            element_type = pyarrow_dtype[field_name].type.value_type
            test_series = pd.Series([], dtype=pd.ArrowDtype(element_type))
            try:
                _ = test_series.astype(inner_dtype)
            except TypeError as e:
                raise TypeError(
                    f"Could not cast the inner dtype '{inner_dtype}' for field '{field_name}' to the"
                    f" corresponding element type '{element_type}'. {e}"
                ) from e
        return inner_dtypes

    def inner_dtype(self, field: str) -> object:
        """Get the inner dtype for a field.

        Parameters
        ----------
        field : str
            The field name.

        Returns
        -------
        object
            The inner dtype for the field.
        """
        return self.inner_dtypes.get(field, pd.ArrowDtype(self.pyarrow_dtype[field].type.value_type))

    @property
    def fields(self) -> dict[str, object]:
        """The mapping of field names and pandas dtypes of their items"""
        return {field.name: self.inner_dtype(field.name) for field in self.pyarrow_dtype}

    @property
    def field_names(self) -> list[str]:
        """The list of field names of the nested type"""
        return [field.name for field in self.pyarrow_dtype]

    def to_pandas_arrow_dtype(self) -> ArrowDtype:
        """Convert NestedDtype to a pandas.ArrowDtype.

        Returns
        -------
        ArrowDtype
            The corresponding pandas.ArrowDtype.
        """
        return ArrowDtype(self.pyarrow_dtype)
