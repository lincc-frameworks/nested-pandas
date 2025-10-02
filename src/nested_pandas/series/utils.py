from __future__ import annotations  # TYPE_CHECKING

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from nested_pandas.series.dtype import NestedDtype


def struct_field_names(struct_type: pa.StructType) -> list[str]:
    """Return field names for a pyarrow.StructType in a pyarrow<18-compatible way.

    Note: Once we bump our pyarrow requirement to ">=18", this helper can be
    replaced with direct usage of ``struct_type.names`` throughout the codebase.
    """
    return [f.name for f in struct_type]


def struct_fields(struct_type: pa.StructType) -> list[pa.Field]:
    """Return fields of a pyarrow.StructType in a pyarrow<18-compatible way.

    Note: Once we bump our pyarrow requirement to ">=18", this helper can be
    replaced with direct usage of ``struct_type.fields`` throughout the codebase.
    """
    return [struct_type.field(i) for i in range(struct_type.num_fields)]


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


def align_struct_list_offsets(array: pa.StructArray) -> pa.StructArray:
    """Checks if all struct-list offsets are the same, and reallocates if needed

    Parameters
    ----------
    array : pa.StructArray
        Input struct array.

    Returns
    -------
    pa.StructArray
        Array with all struct-list offsets aligned. May be the input,
        if it was valid.

    Raises
    ------
    ValueError
        If the input is not a valid "nested" StructArray.
    """
    if not pa.types.is_struct(array.type):
        raise ValueError(f"Expected a StructArray, got {array.type}")

    first_offsets: pa.ListArray | None = None
    for field in array.type:
        inner_array = array.field(field.name)
        if not is_pa_type_a_list(inner_array.type):
            raise ValueError(f"Expected a ListArray, got {inner_array.type}")
        list_array = cast(pa.ListArray, inner_array)

        if first_offsets is None:
            first_offsets = list_array.offsets
            continue
        # compare offsets from the first list array with the current one
        if not first_offsets.equals(list_array.offsets):
            break
    else:
        # Return the original array if all offsets match
        return array

    new_offsets = pa.compute.subtract(first_offsets, first_offsets[0])
    value_lengths = None
    list_arrays = []
    for field in array.type:
        inner_array = array.field(field.name)
        list_array = cast(pa.ListArray, inner_array)

        if value_lengths is None:
            value_lengths = list_array.value_lengths()
        elif not value_lengths.equals(list_array.value_lengths()):
            raise ValueError(
                f"List lengths do not match for struct fields {array.type.field(0).name} and {field.name}",
            )

        list_arrays.append(
            pa.ListArray.from_arrays(
                values=list_array.values[list_array.offsets[0].as_py() : list_array.offsets[-1].as_py()],
                offsets=new_offsets,
            )
        )
    new_array = pa.StructArray.from_arrays(
        arrays=list_arrays,
        fields=struct_fields(array.type),
    )
    return new_array


def align_chunked_struct_list_offsets(array: pa.Array | pa.ChunkedArray) -> pa.ChunkedArray:
    """Checks if all struct-list offsets are the same, and reallocates if needed

    Parameters
    ----------
    array : pa.ChunkedArray or pa.Array
        Input chunked array, it must be a valid "nested" struct-list array,
        e.g. all list lengths must match. Non-chunked arrays are allowed,
        but the return array will always be chunked.

    Returns
    -------
    pa.ChunkedArray
        Chunked array with all struct-list offsets aligned.

    Raises
    ------
    ValueError
        If the input is not a valid "nested" struct-list-array.
    """
    if isinstance(array, pa.Array):
        array = pa.chunked_array([array])
    chunks = [align_struct_list_offsets(chunk) for chunk in array.iterchunks()]
    # Provide type for the case of zero-chunks array
    return pa.chunked_array(chunks, type=array.type)


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
        array = align_struct_list_offsets(array)

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
        names=struct_field_names(array.type),
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
        {field.name: scalar.values.field(field.name) for field in struct_type},
        type=struct_type,
    )
    return cast(pa.StructScalar, struct_scalar)


def validate_list_struct_type(t: pa.ListType) -> None:
    """Raise a ValueError if not a list-struct type."""
    if not is_pa_type_a_list(t):
        raise ValueError(f"Expected a ListType, got {t}")

    if not pa.types.is_struct(t.value_type):
        raise ValueError(f"Expected a StructType as a list value type, got {t.value_type}")


def validate_struct_list_type(t: pa.StructType) -> None:
    """Raise a ValueError if not a struct-list-type."""
    if not pa.types.is_struct(t):
        raise ValueError(f"Expected a StructType, got {t}")

    for field in struct_fields(t):
        if not is_pa_type_a_list(field.type):
            raise ValueError(f"Expected a ListType for field {field.name}, got {field.type}")


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
        names=struct_field_names(array.type.value_type),
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
