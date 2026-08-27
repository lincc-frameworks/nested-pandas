"""nested-pandas robustness against pyarrow's ``null``-typed nested cast bugs.

These assert that nested-pandas operations produce *valid* pyarrow arrays even
when a nested sub-column is ``null``-typed (all-null). The underlying pyarrow
bugs are documented in ``tests/nested_pandas/test_pyarrow_null_bugs.py``.
"""

import pyarrow as pa

from nested_pandas.series.ext_array import NestedExtensionArray


def _null_nested_array():
    """large_list<struct<n: null, v: int64>> with row lengths [1, 1, 2].

    This is the on-the-wire "list of structs" form of a nested column whose
    ``n`` sub-column is all-null (``null``-typed).
    """
    struct = pa.StructArray.from_arrays(
        [pa.nulls(4, pa.null()), pa.array([10, 20, 30, 40], pa.int64())],
        names=["n", "v"],
    )
    list_offsets = pa.array([0, 1, 2, 4], pa.int64())
    return pa.LargeListArray.from_arrays(list_offsets, struct)


def _validate(array):
    array = array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array
    array.validate(full=True)
    return array


def test_arrow_array_default_null_field():
    """``pa.array(series)`` (no explicit type) stays valid with a null field."""
    ext_array = NestedExtensionArray(_null_nested_array())
    _validate(pa.array(ext_array))


def test_arrow_array_struct_type_null_field():
    """``pa.array(series, type=struct)`` must not corrupt the null field."""
    ext_array = NestedExtensionArray(_null_nested_array())
    struct_type = ext_array.struct_array.type
    result = _validate(pa.array(ext_array, type=struct_type))
    assert result.field("n").values.null_count == 4


def test_arrow_array_list_type_null_field():
    """``pa.array(series, type=list<struct>)`` must not corrupt the null field."""
    ext_array = NestedExtensionArray(_null_nested_array())
    list_type = ext_array.list_array.type
    _validate(pa.array(ext_array, type=list_type))


def test_arrow_array_almost_same_type_null_field():
    """Casting that changes a *non-null* field while keeping the null field.

    This is the case a simple ``array.type == type`` guard misses: the overall
    type differs (the ``v`` field becomes non-nullable), but the untouched ``n``
    field is a null->null child cast that pyarrow corrupts.
    """
    ext_array = NestedExtensionArray(_null_nested_array())
    struct_type = ext_array.struct_array.type
    # Flip only the non-null field's nullability -- a pyarrow-supported difference
    # that leaves the null field 'n' byte-for-byte identical.
    almost = pa.struct(
        [f if f.name == "n" else pa.field(f.name, f.type, nullable=False) for f in struct_type]
    )
    assert struct_type != almost  # a whole-array `type ==` guard would not fire
    result = _validate(pa.array(ext_array, type=almost))
    assert result.field("n").values.null_count == 4


def _nested_array_with_inner(inner_values):
    """large_list<struct<n: <inner>, v: int64>> with row lengths [1, 1, 2].

    Like :func:`_null_nested_array`, but ``n``'s ``null``-typed leaf sits inside
    another container (a fixed-size list, a map) rather than being the element
    type of ``n``'s own list.
    """
    struct = pa.StructArray.from_arrays(
        [inner_values, pa.array([10, 20, 30, 40], pa.int64())],
        names=["n", "v"],
    )
    return pa.LargeListArray.from_arrays(pa.array([0, 1, 2, 4], pa.int64()), struct)


def _almost_same_struct_type(ext_array):
    """The array's struct type with only the non-null ``v`` field's nullability flipped.

    A pyarrow-supported difference that leaves ``n`` byte-for-byte identical, so
    a whole-array ``type ==`` guard does not fire and the null leaf is reached.
    """
    struct_type = ext_array.struct_array.type
    almost = pa.struct(
        [f if f.name == "n" else pa.field(f.name, f.type, nullable=False) for f in struct_type]
    )
    assert struct_type != almost
    return almost


def test_arrow_array_null_inside_fixed_size_list():
    """A ``null`` leaf nested in a fixed-size list must survive the cast.

    ``fixed_size_list`` has no ``.offsets``, so it cannot go through the
    offset-based reconstruction that handles ``list``/``large_list``. It still
    has to be detected as containing a ``null`` leaf, or the cast is delegated
    to pyarrow and arrow#43838 truncates the values buffer.
    """
    inner = pa.FixedSizeListArray.from_arrays(pa.nulls(8, pa.null()), 2)
    ext_array = NestedExtensionArray(_nested_array_with_inner(inner))
    assert ext_array.dtype.pyarrow_dtype.field("n").type.value_type == inner.type

    result = _validate(pa.array(ext_array, type=_almost_same_struct_type(ext_array)))
    assert result.field("n").values.values.null_count == 8


def test_arrow_array_null_inside_map():
    """A ``null``-typed map value must survive the cast, for the same reason."""
    inner = pa.MapArray.from_arrays(
        pa.array([0, 1, 2, 3, 4], pa.int32()),
        pa.array(["a", "b", "c", "d"], pa.string()),
        pa.nulls(4, pa.null()),
    )
    ext_array = NestedExtensionArray(_nested_array_with_inner(inner))
    assert ext_array.dtype.pyarrow_dtype.field("n").type.value_type == inner.type

    _validate(pa.array(ext_array, type=_almost_same_struct_type(ext_array)))
