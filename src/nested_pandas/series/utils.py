from __future__ import annotations  # TYPE_CHECKING

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from nested_pandas.series.dtype import NestedDtype


def is_pa_type_a_list(pa_type: pa.DataType) -> bool:
    """Check if the given pyarrow type is a list type.

    I.e., one of the following types: ListArray, LargeListArray,
    FixedSizeListArray.

    Parameters
    ----------
    pa_type : pa.DataType
        The pyarrow type to check.

    Returns
    -------
    bool
        True if the given type is a list type, False otherwise.
    """
    return (
        pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type) or pa.types.is_fixed_size_list(pa_type)
    )


def is_pa_type_is_list_struct(pa_type: pa.DataType) -> bool:
    """Check if the given pyarrow type is a list-struct type.

    Parameters
    ----------
    pa_type : pa.DataType
        The pyarrow type to check.

    Returns
    -------
    bool
        True if the given type is a list-type with struct values,
        False otherwise.
    """
    return is_pa_type_a_list(pa_type) and pa.types.is_struct(pa_type.value_type)


def validate_struct_list_array_for_equal_lengths(array: pa.StructArray) -> None:
    """Check if the given struct array has lists of equal length.

    Parameters
    ----------
    array : pa.StructArray
        Input struct array.

    Raises
    ------
    ValueError
        If the struct array has lists of unequal length or type of the input
        array is not a StructArray or fields are not ListArrays.
    """
    if not pa.types.is_struct(array.type):
        raise ValueError(f"Expected a StructArray, got {array.type}")

    first_list_array: pa.ListArray | None = None
    for field in array.type:
        inner_array = array.field(field.name)
        if not is_pa_type_a_list(inner_array.type):
            raise ValueError(f"Expected a ListArray, got {inner_array.type}")
        list_array = cast(pa.ListArray, inner_array)

        if first_list_array is None:
            first_list_array = list_array
            continue
        # compare offsets from the first list array with the current one
        if not first_list_array.offsets.equals(list_array.offsets):
            raise ValueError("Offsets of all ListArrays must be the same")


def transpose_struct_list_type(t: pa.StructType) -> pa.ListType:
    """Converts a type of struct-list array into a type of list-struct array.

    Parameters
    ----------
    t : pa.DataType
        Input type of struct-list array.

    Returns
    -------
    pa.DataType
        Type of list-struct array.

    Raises
    ------
    ValueError
        If the input type is not a struct-list type.
    """
    if not pa.types.is_struct(t):
        raise ValueError(f"Expected a StructType, got {t}")

    fields = []
    for field in t:
        if not is_pa_type_a_list(field.type):
            raise ValueError(f"Expected a ListType, got {field.type}")
        list_type = cast(pa.ListType, field.type)
        fields.append(pa.field(field.name, list_type.value_type))

    list_type = cast(pa.ListType, pa.list_(pa.struct(fields)))
    return list_type


def transpose_struct_list_array(array: pa.StructArray, validate: bool = True) -> pa.ListArray:
    """Converts a struct-array of lists into a list-array of structs.

    Parameters
    ----------
    array : pa.StructArray
        Input struct array, each scalar must have lists of equal length.
    validate : bool, default True
        Whether to validate the input array for list lengths. Raises ValueError
        if something is wrong.

    Returns
    -------
    pa.ListArray
        List array of structs.
    """
    if validate:
        validate_struct_list_array_for_equal_lengths(array)

    mask = array.is_null()
    if not pa.compute.any(mask).as_py():
        mask = None

    # Since we know that all lists have the same length, we can use the first list to get offsets
    try:
        offsets = array.field(0).offsets
    except IndexError as e:
        raise ValueError("Nested arrays must have at least one field") from e
    else:
        # Shift offsets
        if offsets.offset != 0:
            offsets = pa.compute.subtract(offsets, offsets[0])

    struct_flat_array = pa.StructArray.from_arrays(
        # Select values within the offsets
        [field.values[field.offsets[0].as_py() : field.offsets[-1].as_py()] for field in array.flatten()],
        names=array.type.names,
    )
    return pa.ListArray.from_arrays(
        offsets=offsets,
        values=struct_flat_array,
        mask=mask,
    )


def transpose_struct_list_chunked(chunked_array: pa.ChunkedArray, validate: bool = True) -> pa.ChunkedArray:
    """Converts a chunked array of struct-list into a chunked array of list-struct.

    Parameters
    ----------
    chunked_array : pa.ChunkedArray
        Input chunked array of struct-list.
    validate : bool, default True
        Whether to validate the input array for list lengths. Raises ValueError
        if something is wrong.

    Returns
    -------
    pa.ChunkedArray
        Chunked array of list-struct.
    """
    if chunked_array.num_chunks == 0:
        return pa.chunked_array([], type=transpose_struct_list_type(chunked_array.type))
    return pa.chunked_array(
        [transpose_struct_list_array(array, validate) for array in chunked_array.iterchunks()]
    )


def transpose_list_struct_scalar(scalar: pa.ListScalar) -> pa.StructScalar:
    """Converts a list-scalar of structs into a struct-scalar of lists.

    Parameters
    ----------
    scalar : pa.ListScalar
        Input list-struct scalar.

    Returns
    -------
    pa.StructScalar
        Struct-list scalar.
    """
    struct_type = transpose_list_struct_type(scalar.type)
    struct_scalar = pa.scalar(
        {field: scalar.values.field(field) for field in struct_type.names},
        type=struct_type,
    )
    return cast(pa.StructScalar, struct_scalar)


def validate_list_struct_type(t: pa.ListType) -> None:
    """Raise a ValueError if not a list-struct type."""
    if not is_pa_type_a_list(t):
        raise ValueError(f"Expected a ListType, got {t}")

    if not pa.types.is_struct(t.value_type):
        raise ValueError(f"Expected a StructType as a list value type, got {t.value_type}")


def transpose_list_struct_type(t: pa.ListType) -> pa.StructType:
    """Converts a type of list-struct array into a type of struct-list array.

    Parameters
    ----------
    t : pa.DataType
        Input type of list-struct array.

    Returns
    -------
    pa.DataType
        Type of struct-list array.

    Raises
    ------
    ValueError
        If the input type is not a list-struct type.
    """
    validate_list_struct_type(t)

    struct_type = cast(pa.StructType, t.value_type)
    fields = []
    for field in struct_type:
        fields.append(pa.field(field.name, pa.list_(field.type)))

    struct_type = cast(pa.StructType, pa.struct(fields))
    return struct_type


def transpose_list_struct_array(array: pa.ListArray) -> pa.StructArray:
    """Converts a list-array of structs into a struct-array of lists.

    Parameters
    ----------
    array : pa.ListArray
        Input list array of structs.

    Returns
    -------
    pa.StructArray
        Struct array of lists.
    """
    offsets, values = array.offsets, array.values
    mask = array.is_null()
    if not pa.compute.any(mask).as_py():
        mask = None

    fields = []
    for field_values in values.flatten():
        list_array = pa.ListArray.from_arrays(offsets, field_values)
        fields.append(list_array)

    return pa.StructArray.from_arrays(
        arrays=fields,
        names=array.type.value_type.names,
        mask=mask,
    )


def transpose_list_struct_chunked(chunked_array: pa.ChunkedArray) -> pa.ChunkedArray:
    """Converts a chunked array of list-struct into a chunked array of struct-list.

    Parameters
    ----------
    chunked_array : pa.ChunkedArray
        Input chunked array of list-struct.

    Returns
    -------
    pa.ChunkedArray
        Chunked array of struct-list.
    """
    if chunked_array.num_chunks == 0:
        return pa.chunked_array([], type=transpose_list_struct_type(chunked_array.type))
    return pa.chunked_array([transpose_list_struct_array(array) for array in chunked_array.iterchunks()])


def nested_types_mapper(type: pa.DataType) -> pd.ArrowDtype | NestedDtype:
    """Type mapper for pyarrow .to_pandas(types_mapper) methods."""
    from nested_pandas.series.dtype import NestedDtype

    if pa.types.is_list(type):
        try:
            return NestedDtype(type)
        except (ValueError, TypeError):
            return pd.ArrowDtype(type)
    return pd.ArrowDtype(type)


def table_to_struct_array(table: pa.Table) -> pa.ChunkedArray:
    """pa.Table.to_struct_array

    pyarrow has a bug for empty tables:
    https://github.com/apache/arrow/issues/46355
    """
    if len(table) == 0:
        return pa.chunked_array([], type=pa.struct(table.schema))
    return table.to_struct_array()


def table_from_struct_array(array: pa.ChunkedArray | pa.array) -> pa.Table:
    """pa.Table.from_struct_array, but working with chunkless input"""
    if isinstance(array, pa.ChunkedArray) and array.num_chunks == 0:
        array = pa.array([], type=array.type)
    return pa.Table.from_struct_array(array)


def chunk_lengths(array: pa.ChunkedArray) -> list[int]:
    """Get the length of each chunk in an array."""
    return [len(chunk) for chunk in array.iterchunks()]


def rechunk(array: pa.Array | pa.ChunkedArray, chunk_lens: ArrayLike) -> pa.ChunkedArray:
    """Rechunk array to the same chunks a given chunked array.

    If no rechunk is needed the original chunked array is returned.

    Parameters
    ----------
    array : pa.Array | pa.ChunkedArray
        Input chunked or non-chunked array to rechunk.
    chunk_lens : array-like of int
        Lengths of chunks.

    Returns
    -------
    pa.ChunkedArray
        Rechunked `array`.
    """
    if len(array) != np.sum(chunk_lens):
        raise ValueError("Input array must have the same length as the total chunk lengths")
    if isinstance(array, pa.Array):
        array = pa.chunked_array([array])

    # Shortcut if no rechunk is needed:
    if chunk_lengths(array) == chunk_lens:
        return array
    chunk_indices = np.r_[0, np.cumsum(chunk_lens)]
    chunks = []
    for idx_start, idx_end in zip(chunk_indices[:-1], chunk_indices[1:], strict=True):
        chunk = array[idx_start:idx_end].combine_chunks()
        chunks.append(chunk)
    return pa.chunked_array(chunks)
