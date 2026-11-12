from __future__ import annotations  # Self is not available in python 3.10

from collections.abc import Mapping

# We use Type, because we must use "type" as an attribute name
from typing import Type, cast  # noqa: UP035

import pandas as pd
import pyarrow as pa
from pandas import ArrowDtype
from pandas.api.extensions import register_extension_dtype
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype

from nested_pandas.series.registry import (
    arrow_type_from_string,
    pandas_dtype_from_arrow,
    register_arrow_dtype,
)
from nested_pandas.series.utils import (
    is_pa_type_a_list,
    is_pa_type_is_list_struct,
    normalize_struct_list_type,
    normalize_tensor_types,
    transpose_list_struct_type,
    transpose_struct_list_type,
)

__all__ = ["NestedDtype"]


@register_extension_dtype
class NestedDtype(ExtensionDtype):
    """Data type to handle packed time series data

    Parameters
    ----------
    pyarrow_dtype : pyarrow.StructType, pd.ArrowDtype, or Mapping[str, pa.DataType]
        The pyarrow data type to use for the nested type. It may be provided as
        a pyarrow.StructType, a pandas.ArrowDtype, or a mapping of column names to
        pyarrow data types (such as a dictionary).

    Examples
    --------
    >>> import pyarrow as pa
    >>> from nested_pandas import NestedDtype

    From pa.StructType:

    >>> dtype = NestedDtype(pa.struct([pa.field("a", pa.list_(pa.int64())),
    ...                                pa.field("b", pa.list_(pa.float64()))]))
    >>> dtype
    nested<a: [int64], b: [double]>

    From pd.ArrowDtype:

    >>> import pandas as pd
    >>> dtype = NestedDtype(pd.ArrowDtype(pa.struct([pa.field("a", pa.list_(pa.int64())),
    ...                                           pa.field("b", pa.list_(pa.float64()))])))
    >>> dtype
    nested<a: [int64], b: [double]>

    From mapping of column names to pyarrow data types:

    >>> dtype = NestedDtype({"a": pa.int64(), "b": pa.float64()})
    >>> dtype
    nested<a: [int64], b: [double]>
    """

    # ExtensionDtype overrides #

    _metadata = ("pyarrow_dtype",)
    """Attributes to use as metadata for __eq__ and __hash__"""

    @property
    def na_value(self):
        """The missing value for this dtype"""
        return pd.NA  # type: ignore[return-value]

    type = pd.DataFrame
    """The type of the array's elements, always pd.DataFrame"""

    @property
    def name(self) -> str:
        """The string representation of the nested type"""
        # Replace pd.ArrowDtype with pa.DataType, because it has nicer __str__
        field_dtypes = {field: self.column_dtype(field) for field in list(self.column_dtypes.keys())}
        nice_dtypes = {
            field: dtype.pyarrow_dtype if isinstance(dtype, pd.ArrowDtype) else dtype
            for field, dtype in field_dtypes.items()
        }
        fields = ", ".join([f"{field}: [{dtype!s}]" for field, dtype in nice_dtypes.items()])
        return f"nested<{fields}>"

    @name.setter
    def name(self, value: str):
        raise TypeError("name cannot be changed")

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def construct_array_type(cls) -> Type[ExtensionArray]:
        """Corresponded array type, always NestedExtensionArray"""
        from nested_pandas.series.ext_array import NestedExtensionArray

        return NestedExtensionArray

    @classmethod
    def construct_from_string(cls, string: str) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Construct NestedDtype from a string representation.

        Sub-column types may be names of pyarrow types, including the
        common parametric ones ``pd.ArrowDtype`` parses, such as
        ``timestamp[ns]``, and names of pandas extension dtypes, such as
        ``nested<x: [int64]>`` and ``tensor[double, (2, 2)]``.

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
            If the string is not a valid nested type string or if the type of a
            sub-column cannot be parsed, as for the less common parametric
            pyarrow types.
        """
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        if not string.startswith("nested<") or not string.endswith(">"):
            raise TypeError("Not a valid nested type string, expected 'nested<...>'")
        fields_str = string.removeprefix("nested<").removesuffix(">")

        field_strings = _split_top_level(fields_str)

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
            try:
                fields[field_name] = arrow_type_from_string(value_type)
            except TypeError as e:
                raise TypeError(
                    f"Cannot parse the type of '{field_name}' in the nested type string: {e}. "
                    "Please use NestedDtype() or NestedDtype.from_columns() instead."
                ) from e

        return cls.from_columns(fields)

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

    def __init__(self, pyarrow_dtype: pa.DataType | Mapping) -> None:
        # Allow pd.ArrowDtypes on init
        if isinstance(pyarrow_dtype, pd.ArrowDtype):
            pyarrow_dtype = pyarrow_dtype.pyarrow_dtype

        # Allow from_columns-style mapping inputs
        if isinstance(pyarrow_dtype, Mapping):
            pyarrow_dtype = pa.struct({col: pa.large_list(pa_type) for col, pa_type in pyarrow_dtype.items()})
            pyarrow_dtype = cast(pa.StructType, pyarrow_dtype)

        self.pyarrow_dtype, self.list_struct_pa_dtype = self._validate_dtype(pyarrow_dtype)

    @property
    def _struct_list_pa_dtype(self) -> pa.StructType:
        """Struct-list pyarrow type representing the nested type."""
        return self.pyarrow_dtype

    @classmethod
    def from_columns(cls, columns: Mapping[str, pa.DataType]) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Make NestedDtype from a mapping of column names and list item types.

        Parameters
        ----------
        columns : Mapping[str, pa.DataType]
            A mapping of column names and their item types. Since all fields are lists, the item types are
            inner types of the lists, not the list types themselves.

        Returns
        -------
        NestedDtype
            The constructed NestedDtype.

        Examples
        --------
        >>> dtype = NestedDtype.from_columns({"a": pa.float64(), "b": pa.int64()})
        >>> dtype
        nested<a: [double], b: [int64]>
        >>> assert (
        ...     dtype.pyarrow_dtype
        ...     == pa.struct({"a": pa.large_list(pa.float64()), "b": pa.large_list(pa.int64())})
        ... )
        """
        pyarrow_dtype = pa.struct({column: pa.large_list(pa_type) for column, pa_type in columns.items()})
        pyarrow_dtype = cast(pa.StructType, pyarrow_dtype)
        return cls(pyarrow_dtype=pyarrow_dtype)

    @staticmethod
    def _validate_dtype(pyarrow_dtype: pa.DataType) -> tuple[pa.StructType, pa.LargeListType]:
        """Check that the given pyarrow type is castable to the nested type.

        Parameters
        ----------
        pyarrow_dtype : pa.DataType
            The pyarrow type to check and cast.

        Returns
        -------
        pa.StructType
            Struct-list pyarrow type representing the nested type.
        pa.LargeListType
            List-struct pyarrow type representing the nested type.
        """
        if not isinstance(pyarrow_dtype, pa.DataType):
            raise TypeError(f"Expected a 'pyarrow.DataType' object, got {type(pyarrow_dtype)}")
        if pa.types.is_struct(pyarrow_dtype):
            struct_type = cast(pa.StructType, pyarrow_dtype)
            # Normalize list fields to large_list for backward compatibility
            # (callers may pass pa.list_ fields)
            struct_type = cast(pa.StructType, normalize_tensor_types(normalize_struct_list_type(struct_type)))
            return struct_type, transpose_struct_list_type(struct_type)
        # Support pa.large_list (and pa.list_ for backward compatibility)
        if is_pa_type_a_list(pyarrow_dtype):
            if not pa.types.is_large_list(pyarrow_dtype):
                # Normalize regular list or fixed-size list to large_list
                pyarrow_dtype = pa.large_list(pyarrow_dtype.value_type)
            list_type = cast(pa.LargeListType, normalize_tensor_types(pyarrow_dtype))
            return transpose_list_struct_type(list_type), list_type
        raise ValueError(
            "NestedDtype can only be constructed with pa.StructType, pa.LargeListType, "
            f"or pa.ListType, got {pyarrow_dtype}"
        )

    @property
    def column_dtypes(self) -> dict[str, pa.DataType]:
        """The mapping of field names and their item types."""
        return {column.name: column.type.value_type for column in self.pyarrow_dtype}

    @classmethod
    def from_pandas_arrow_dtype(cls, pandas_arrow_dtype: ArrowDtype) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Construct NestedDtype from a pandas.ArrowDtype.

        Parameters
        ----------
        pandas_arrow_dtype : ArrowDtype
            The pandas.ArrowDtype to construct NestedDtype from.
            Must be struct-list or list-struct type.

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

    def to_pandas_arrow_dtype(self, list_struct: bool = False) -> ArrowDtype:
        """Convert NestedDtype to a pandas.ArrowDtype.

        Parameters
        ----------
        list_struct : bool, default False
            If False (default) use pyarrow struct-list type,
            otherwise use pyarrow list-struct type.

        Returns
        -------
        ArrowDtype
            The corresponding pandas.ArrowDtype.
        """
        if list_struct:
            return ArrowDtype(self.list_struct_pa_dtype)
        return ArrowDtype(self.pyarrow_dtype)

    def column_dtype(self, column: str) -> ExtensionDtype:
        """Pandas dtype of a column: pd.ArrowDtype, or the extension dtype of a nested or tensor column.

        Parameters
        ----------
        column : str
            Column name

        Returns
        -------
        ExtensionDtype
            If the column is a list-struct, return NestedDtype, if it is a
            fixed-shape tensor, return TensorDtype, else wrap it as a
            pd.ArrowDtype.
        """
        list_type = self.pyarrow_dtype.field(column).type
        return pandas_dtype_from_arrow(list_type.value_type)


def _split_top_level(string: str, separator: str = ", ") -> list[str]:
    """Split a string on a separator, ignoring the separators inside any kind of brackets."""
    parts = []
    depth = 0
    start = 0
    i = 0
    while i < len(string):
        char = string[i]
        if char in "<[(":
            depth += 1
        elif char in ">])":
            depth -= 1
        elif depth == 0 and string.startswith(separator, i):
            parts.append(string[start:i])
            i += len(separator)
            start = i
            continue
        i += 1
    parts.append(string[start:])
    return parts


register_arrow_dtype(NestedDtype, is_pa_type_is_list_struct, lambda dtype: dtype.list_struct_pa_dtype)
