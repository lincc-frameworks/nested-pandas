from __future__ import annotations  # TYPE_CHECKING

from collections.abc import Sequence
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
    """Check if the given pyarrow type is any list type.

    This includes ListArray, FixedSizeListArray and LargeListArray.

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
    return (pa.types.is_large_list(pa_type) or pa.types.is_list(pa_type)) and pa.types.is_struct(
        pa_type.value_type
    )


def zero_align_large_list_offsets(array: pa.LargeListArray) -> pa.LargeListArray:
    """Realign a LargeListArray so its offsets start at zero.

    If the first offset is already zero the original array is returned unchanged.

    Parameters
    ----------
    array : pa.LargeListArray
        Input list array whose offsets may start at a non-zero value.

    Returns
    -------
    pa.LargeListArray
        List array with offsets starting at zero and values sliced accordingly.
    """
    return cast(pa.LargeListArray, _zero_align_list_offsets(array))


def _zero_align_list_offsets(array: pa.Array) -> pa.Array:
    """Realign a ``list``/``large_list`` array so its offsets start at zero.

    Nulls are preserved: rebasing rebuilds the array from ``offsets``, which
    carry no validity of their own, so the mask has to be reapplied explicitly.

    A sliced list array keeps its parent's offsets, so they start mid-buffer.
    ``from_arrays`` refuses such offsets together with a validity mask
    ("Null bitmap with offsets slice not supported"), so any reconstruction
    passing a mask must rebase first.

    If the first offset is already zero the original array is returned unchanged.
    """
    offsets = array.offsets
    first = offsets[0].as_py()
    if first == 0:
        return array
    list_cls = pa.LargeListArray if pa.types.is_large_list(array.type) else pa.ListArray
    return list_cls.from_arrays(
        values=array.values[first : offsets[-1].as_py()],
        offsets=pa.compute.subtract(offsets, offsets[0]),
        mask=array.is_null() if array.null_count else None,
    )


def zero_align_struct_list_offsets(array: pa.StructArray) -> pa.StructArray:
    """Realign all LargeList fields in a StructArray so their offsets start at zero.

    Parameters
    ----------
    array : pa.StructArray
        Input struct array whose fields are LargeListArrays.

    Returns
    -------
    pa.StructArray
        Struct array with every field's offsets starting at zero.

    Raises
    ------
    ValueError
        If list lengths do not match across struct fields.
    """
    value_lengths = None
    list_arrays = []
    for field in array.type:
        inner_array = array.field(field.name)
        list_array = cast(pa.LargeListArray, inner_array)

        if value_lengths is None:
            value_lengths = list_array.value_lengths()
        elif not value_lengths.equals(list_array.value_lengths()):
            raise ValueError(
                f"List lengths do not match for struct fields {array.type.field(0).name} and {field.name}",
            )

        list_arrays.append(zero_align_large_list_offsets(list_array))
    return pa.StructArray.from_arrays(
        arrays=list_arrays,
        fields=struct_fields(array.type),
    )


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

    first_offsets: pa.LargeListArray | None = None
    for field in array.type:
        inner_array = array.field(field.name)
        if not pa.types.is_large_list(inner_array.type):
            raise ValueError(f"Expected a LargeListArray, got {inner_array.type}")
        list_array = cast(pa.LargeListArray, inner_array)

        if first_offsets is None:
            first_offsets = list_array.offsets
            continue
        # compare offsets from the first list array with the current one
        if not first_offsets.equals(list_array.offsets):
            break
    else:
        # Return the original array if all offsets match
        return array

    return zero_align_struct_list_offsets(array)


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


def transpose_struct_list_type(t: pa.StructType) -> pa.LargeListType:
    """Converts a type of struct-list array into a type of list-struct array.

    Parameters
    ----------
    t : pa.DataType
        Input type of struct-list array.

    Returns
    -------
    pa.LargeListType
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
        if not pa.types.is_large_list(field.type):
            raise ValueError(f"Expected a LargeListType, got {field.type}")
        list_type = cast(pa.LargeListType, field.type)
        fields.append(pa.field(field.name, list_type.value_type))

    list_type = cast(pa.LargeListType, pa.large_list(pa.struct(fields)))
    return list_type


def transpose_struct_list_array(array: pa.StructArray, validate: bool = True) -> pa.LargeListArray:
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
    pa.LargeListArray
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
    return pa.LargeListArray.from_arrays(
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


def transpose_list_struct_scalar(scalar: pa.LargeListScalar | pa.ListScalar) -> pa.StructScalar:
    """Converts a list-scalar of structs into a struct-scalar of lists.

    Parameters
    ----------
    scalar : pa.LargeListScalar or pa.ListScalar
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


def validate_list_struct_type(t: pa.LargeListType) -> None:
    """Raise a ValueError if not a large-list-struct type."""
    if not pa.types.is_large_list(t):
        raise ValueError(f"Expected a LargeListType, got {t}")

    if not pa.types.is_struct(t.value_type):
        raise ValueError(f"Expected a StructType as a list value type, got {t.value_type}")


def validate_struct_list_type(t: pa.StructType) -> None:
    """Raise a ValueError if not a struct-list-type."""
    if not pa.types.is_struct(t):
        raise ValueError(f"Expected a StructType, got {t}")

    for field in struct_fields(t):
        if not pa.types.is_large_list(field.type):
            raise ValueError(f"Expected a LargeListType for field {field.name}, got {field.type}")


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
        fields.append(pa.field(field.name, pa.large_list(field.type)))

    struct_type = cast(pa.StructType, pa.struct(fields))
    return struct_type


def transpose_list_struct_array(array: pa.LargeListArray) -> pa.StructArray:
    """Converts a list-array of structs into a struct-array of lists.

    Parameters
    ----------
    array : pa.LargeListArray
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
        list_array = pa.LargeListArray.from_arrays(offsets, field_values)
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

    if pa.types.is_large_list(type):
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


def normalize_list_array(
    array: pa.ListArray | pa.FixedSizeListArray | pa.LargeListArray | pa.ChunkedArray,
) -> pa.LargeListArray | pa.ChunkedArray:
    """Convert fixed-size and regular list arrays to standard ``pa.LargeListArray``-based arrays.

    Parameters
    ----------
    array : pa.ListArray or pa.FixedSizeListArray or pa.LargeListArray or pa.ChunkedArray
        Input list-like array. May be a regular, fixed-size, or large list array,
        or a ``pa.ChunkedArray`` whose ``type`` is one of these list types.

    Returns
    -------
    pa.LargeListArray or pa.ChunkedArray
        A list array (or chunked list array) where the list type is normalized to
        ``pa.LargeListType`` while preserving the original value type.

    Raises
    ------
    ValueError
        If the input is not a list-type array (i.e. does not have a ``.type.value_type``).
    """
    # Pass large-list-array as is
    if pa.types.is_large_list(array.type):
        return array

    try:
        value_type = array.type.value_type
    except AttributeError:
        raise ValueError(
            "Input array must be a list-type array: pa.ListArray, pa.LargeListArray or pa.FixedSizeListArray"
        ) from None

    return array.cast(pa.large_list(value_type))


def normalize_struct_list_type(struct_type: pa.StructType) -> pa.StructType:
    """Convert all struct-list fields to ``pa.large_list`` types.

    Parameters
    ----------
    struct_type : pa.StructType
        Input struct type, fields are assumed to be pa.ListType,
        pa.LargeListType or pa.FixedSizeListType.

    Returns
    -------
    pa.StructType
        Output struct type with all fields as pa.LargeListType.

    Raises
    ------
    ValueError
        If the input type is not a struct-list type.
    """
    if not pa.types.is_struct(struct_type):
        raise ValueError(f"Expected a StructType, got {struct_type}")
    fields = []
    for field in struct_fields(struct_type):
        try:
            value_type = field.type.value_type
        except AttributeError:
            raise ValueError("Input struct_type must be a struct-list type") from None
        fields.append(pa.field(field.name, pa.large_list(value_type)))
    return pa.struct(fields)


def downcast_large_list_type(t: pa.LargeListType | pa.StructType) -> pa.ListType | pa.StructType:
    """Convert a ``large_list`` type or struct-of-``large_list`` type to regular ``list_`` type(s).

    Parameters
    ----------
    t : pa.LargeListType or pa.StructType
        Input type. If a ``large_list`` type, it is converted to ``list_``.
        If a struct type, all ``large_list`` fields are converted to ``list_``.

    Returns
    -------
    pa.ListType or pa.StructType
        Downcast type using int32 offsets.
    """
    if pa.types.is_large_list(t):
        return pa.list_(cast(pa.LargeListType, t).value_type)
    if pa.types.is_struct(t):
        struct_t = cast(pa.StructType, t)
        fields = []
        for field in struct_fields(struct_t):
            if pa.types.is_large_list(field.type):
                fields.append(pa.field(field.name, pa.list_(field.type.value_type)))
            else:
                fields.append(field)
        return pa.struct(fields)
    return cast(pa.ListType | pa.StructType, t)


def downcast_large_list_array(
    array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Cast ``large_list`` fields to regular ``list_`` in an array or chunked array.

    Handles both ``LargeListArray`` (direct downcast) and ``StructArray`` whose
    fields are ``large_list`` (each field is downcast to ``list_``).

    This is a compatibility helper for consumers that do not support ``large_list``
    (e.g. Parquet files written without Arrow schema metadata, older PyArrow
    consumers, etc.).

    Parameters
    ----------
    array : pa.LargeListArray, pa.StructArray, or pa.ChunkedArray thereof
        Input array with ``large_list`` type or struct-of-``large_list`` type.

    Returns
    -------
    pa.Array or pa.ChunkedArray
        The downcast array using int32 offsets.

    Raises
    ------
    ValueError
        If the array contains more than ``2**31 - 1`` total values and cannot
        be represented with int32 offsets. Pass ``large_list=True`` to the
        calling function to keep int64 offsets.
    """
    try:
        return array.cast(downcast_large_list_type(array.type))
    except pa.ArrowInvalid as e:
        raise ValueError(
            "Cannot downcast large_list to list_: the array contains more than "
            f"{2**31 - 1} values, which exceeds the int32 offset range. "
            "Pass large_list=True to keep int64 offsets."
        ) from e


def normalize_struct_list_array(array: pa.StructArray | pa.ChunkedArray) -> pa.StructArray | pa.ChunkedArray:
    """Convert all struct-list fields to "normal" list arrays.

    Parameters
    ----------
    array : pa.StructArray | pa.ChunkedArray
        Input struct array whose fields are list-like (e.g. pa.ListArray,
        pa.LargeListArray or pa.FixedSizeListArray).

    Returns
    -------
    pa.StructArray | pa.ChunkedArray
        Array with all struct-list fields converted to pa.ListArray fields.
        If no normalization is needed, the original ``array`` is returned.

    Raises
    ------
    ValueError
        If the input is not a struct array (i.e. pa.StructArray or a
        pa.ChunkedArray with struct type).
    """
    if not pa.types.is_struct(array.type):
        raise ValueError(f"Expected a StructArray, got {array.type}")
    norm_type = normalize_struct_list_type(array.type)
    if norm_type == array.type:
        return array
    return array.cast(norm_type)


def _is_varlength_list_type(pa_type: pa.DataType) -> bool:
    """``list`` or ``large_list`` -- the variable-length list types with ``.offsets``."""
    return pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type)


def _offsets_for_list_type(offsets: pa.Array, list_type: pa.DataType) -> pa.Array:
    """Return list offsets with the integer width required by ``list_type``."""
    target = pa.int64() if pa.types.is_large_list(list_type) else pa.int32()
    return offsets if offsets.type == target else offsets.cast(target)


def _contains_null_type(pa_type: pa.DataType) -> bool:
    """Whether ``pa_type`` is, or nests, pyarrow's ``null`` type anywhere.

    Used by :func:`safe_cast` to skip its manual reconstruction when no
    ``null``-typed leaf can possibly be involved, since arrow#43838 only
    corrupts a ``null``-typed child cast to its own type.

    Child types are walked generically rather than per-container, so this also
    sees a ``null`` leaf inside a container :func:`safe_cast` cannot rebuild
    itself -- ``fixed_size_list`` (no ``.offsets``) and ``map``. Those must not
    take the "no null leaf, delegate to pyarrow" fast path either; reaching the
    recursion lets the exact-type no-op return the null-bearing child untouched.
    """
    if pa.types.is_null(pa_type):
        return True
    return any(_contains_null_type(pa_type.field(i).type) for i in range(pa_type.num_fields))


def safe_cast(array: pa.Array | pa.ChunkedArray, target_type: pa.DataType) -> pa.Array | pa.ChunkedArray:
    """Cast ``array`` to ``target_type`` without triggering pyarrow's null-cast bug.

    pyarrow silently corrupts a ``null``-typed child array when it is cast to its
    own type (https://github.com/apache/arrow/issues/43838). This happens even
    inside a larger cast where only some fields change. To avoid it, we recurse
    into struct and (large_)list children and never cast a child whose type
    already matches the target: a ``null``-typed child is returned unchanged,
    which is correct since casting a value to its own type is a no-op.

    ``array.type`` is checked up front for a ``null``-typed leaf anywhere in its
    structure; when there is none, the bug cannot be triggered and the cast is
    delegated straight to pyarrow, which is both correct and faster than the
    manual reconstruction below.

    Canary: ``tests/nested_pandas/test_pyarrow_null_bugs.py::
    test_pyarrow_43838_null_list_identity_cast``. If pyarrow fixes this (the
    canary starts xpassing), calls to ``safe_cast`` can revert to plain
    ``.cast()`` and this function can likely be removed.

    Non-nested casts, and casts that genuinely change a non-null leaf type, are
    delegated to pyarrow, which handles them correctly.

    Parameters
    ----------
    array : pa.Array or pa.ChunkedArray
        Input array.
    target_type : pa.DataType
        Type to cast to.

    Returns
    -------
    pa.Array or pa.ChunkedArray
        The array cast to ``target_type``.
    """
    # Exact match at any nesting level: no-op. This both avoids the null-cast bug
    # and skips redundant work.
    if array.type == target_type:
        return array

    # No null-typed leaf anywhere in the source: the bug this function works
    # around cannot be triggered, so let pyarrow do the (faster) native cast.
    if not _contains_null_type(array.type):
        return array.cast(target_type)

    if isinstance(array, pa.ChunkedArray):
        return pa.chunked_array(
            [safe_cast(chunk, target_type) for chunk in array.iterchunks()],
            type=target_type,
        )

    if pa.types.is_struct(array.type) and pa.types.is_struct(target_type):
        source_names = set(struct_field_names(array.type))
        target_fields = struct_fields(target_type)
        # Only recurse when we can source every target field by name; otherwise
        # the structures differ in a way this helper does not model, so defer to
        # pyarrow (which does not involve a null-typed no-op cast here).
        if all(field.name in source_names for field in target_fields):
            children = [safe_cast(array.field(field.name), field.type) for field in target_fields]
            mask = array.is_null() if array.null_count else None
            return pa.StructArray.from_arrays(children, fields=target_fields, mask=mask)
    elif _is_varlength_list_type(array.type) and _is_varlength_list_type(target_type):
        # A slice keeps its parent's offsets, which from_arrays rejects together
        # with the validity mask below; rebase them to zero first.
        array = _zero_align_list_offsets(array)
        values = safe_cast(array.values, target_type.value_type)
        offsets = _offsets_for_list_type(array.offsets, target_type)
        mask = array.is_null() if array.null_count else None
        list_cls = pa.LargeListArray if pa.types.is_large_list(target_type) else pa.ListArray
        return list_cls.from_arrays(offsets, values, mask=mask)

    return array.cast(target_type)


def scalars_to_pa_array(scalars: Sequence[pa.Scalar], pa_type: pa.DataType) -> pa.Array:
    """Build an array from per-row scalars without pyarrow's null-builder gap.

    Equivalent to ``pa.array(scalars, type=pa_type)``, except it avoids
    appending struct/list scalars one at a time. pyarrow raises
    ``ArrowNotImplementedError: AppendScalar for type null`` when a struct or
    list scalar being appended has a ``null``-typed child -- its builder has no
    scalar-append support for a ``null`` child. No upstream issue is filed,
    since this is a missing feature rather than data corruption.

    Instead, struct and (large_)list containers are rebuilt directly from their
    children's already-materialized arrays (``scalar.values``, per-field
    scalars), which never goes through the unsupported append path. Leaf types
    are unaffected by the gap and are built with plain ``pa.array``.

    Canary: ``tests/nested_pandas/test_pyarrow_null_bugs.py::
    test_pyarrow_struct_array_from_null_field_scalars``. If pyarrow adds the
    missing support (the canary starts xpassing), calls to this function can
    revert to plain ``pa.array(scalars, type=pa_type)`` and it can likely be
    removed.

    Parameters
    ----------
    scalars : Sequence[pa.Scalar]
        One scalar per row, all of type ``pa_type``.
    pa_type : pa.DataType
        The common type of ``scalars``.

    Returns
    -------
    pa.Array
        An array equivalent to ``pa.array(scalars, type=pa_type)``.
    """
    if pa.types.is_struct(pa_type):
        validity = [s.is_valid for s in scalars]
        rows = list(zip(scalars, validity, strict=True))
        field_arrays = [
            scalars_to_pa_array(
                [s[field.name] if valid else pa.scalar(None, type=field.type) for s, valid in rows],
                field.type,
            )
            for field in pa_type
        ]
        mask = None if all(validity) else pa.array([not valid for valid in validity])
        return pa.StructArray.from_arrays(field_arrays, fields=list(pa_type), mask=mask)

    if _is_varlength_list_type(pa_type):
        value_type = pa_type.value_type
        offsets_dtype = pa.int64() if pa.types.is_large_list(pa_type) else pa.int32()
        offsets = [0]
        chunks = []
        validity = []
        for s in scalars:
            valid = s.is_valid
            validity.append(valid)
            offsets.append(offsets[-1] + (len(s.values) if valid else 0))
            if valid:
                # Rows may disagree on the values' concrete type (e.g. an
                # all-null row infers `null`, another infers `double`); align
                # each row to the target type before concatenating. safe_cast
                # (rather than a plain .cast()) avoids arrow#43838 here too.
                chunks.append(safe_cast(s.values, value_type))
        values = pa.concat_arrays(chunks) if chunks else pa.array([], type=value_type)
        offsets_array = pa.array(offsets, type=offsets_dtype)
        mask = None if all(validity) else pa.array([not valid for valid in validity])
        list_cls = pa.LargeListArray if pa.types.is_large_list(pa_type) else pa.ListArray
        return list_cls.from_arrays(offsets_array, values, mask=mask)

    return pa.array(scalars, type=pa_type)
