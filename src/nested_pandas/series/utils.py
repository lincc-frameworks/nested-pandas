from __future__ import annotations  # Python 3.9 requires it for X | Y type hints

from collections.abc import Generator
from typing import cast

import pyarrow as pa


def is_pa_type_a_list(pa_type: pa.DataType) -> bool:
    """Check if the given pyarrow type is a list type.

    I.e. one of the following types: ListArray, LargeListArray,
    FixedSizeListArray.

    Returns
    -------
    bool
        True if the given type is a list type, False otherwise.
    """
    return (
        pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type) or pa.types.is_fixed_size_list(pa_type)
    )


def enumerate_chunks(array: pa.ChunkedArray) -> Generator[tuple[slice, pa.Array], None, None]:
    """Iterate over pyarrow.ChunkedArray chunks with their slice indices.

    Parameters
    ----------
    array : pa.ChunkedArray
        Input chunked array.

    Yields
    ------
    slice
        `slice(index_start, index_stop)` for the current chunk.
    pa.Array
        The current chunk.
    """
    index_start = 0
    for chunk in array.iterchunks():
        index_stop = index_start + len(chunk)
        yield slice(index_start, index_stop), chunk
        index_start = index_stop


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

    # Since we know that all lists have the same length, we can use the first list to get offsets
    offsets = array.field(0).offsets
    struct_flat_array = pa.StructArray.from_arrays(
        [field.values for field in array.flatten()],
        names=array.type.names,
    )
    return pa.ListArray.from_arrays(offsets, struct_flat_array)


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
    if not is_pa_type_a_list(t):
        raise ValueError(f"Expected a ListType, got {t}")

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

    fields = []
    for field_values in values.flatten():
        list_array = pa.ListArray.from_arrays(offsets, field_values)
        fields.append(list_array)

    return pa.StructArray.from_arrays(fields, names=array.type.value_type.names)
