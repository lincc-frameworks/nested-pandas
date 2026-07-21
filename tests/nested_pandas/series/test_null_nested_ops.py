"""nested-pandas operations that currently break on an all-null nested field.

A nested sub-column that is entirely null is represented with pyarrow's ``null``
data type. Reading such data works, but several write/build operations still
fail, for two unrelated reasons:

* ``_box_pa_array`` casts the stored array to the requested type via pyarrow's
  ``Array.cast``, which corrupts the null field. Same upstream cast bug as
  https://github.com/apache/arrow/issues/43838; should route through
  ``utils.safe_cast`` like ``__arrow_array__`` already does.
* ``__setitem__`` and ``_from_sequence`` build the array from per-row pyarrow
  scalars, which raises ``ArrowNotImplementedError: AppendScalar for type
  null``. This isn't corruption -- pyarrow just hasn't implemented
  scalar-append into a ``null``-typed struct field yet (canary:
  ``test_pyarrow_null_bugs.test_pyarrow_struct_array_from_null_field_scalars``).
  nested-pandas needs a workaround that avoids that code path.

These tests currently FAIL and mark the next things to fix.
"""

import pandas as pd
import pyarrow as pa

from nested_pandas.series.ext_array import NestedExtensionArray


def _null_nested_series():
    """Series of large_list<struct<n: null, v: int64>> with row lengths [1, 1, 2]."""
    struct = pa.StructArray.from_arrays(
        [pa.nulls(4, pa.null()), pa.array([10, 20, 30, 40], pa.int64())],
        names=["n", "v"],
    )
    list_offsets = pa.array([0, 1, 2, 4], pa.int64())
    ext_array = NestedExtensionArray(pa.LargeListArray.from_arrays(list_offsets, struct))
    return pd.Series(ext_array)


def _validate_struct(ext_array):
    array = ext_array.struct_array
    (array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array).validate(full=True)


def test_box_pa_array_null_field_identity_cast():
    """``_box_pa_array`` with a matching type must not corrupt the null field."""
    ext_array = _null_nested_series().array
    result = NestedExtensionArray._box_pa_array(ext_array, pa_type=ext_array.struct_array.type)
    result = result.combine_chunks() if isinstance(result, pa.ChunkedArray) else result
    result.validate(full=True)


def test_setitem_null_field():
    """Assigning an element of a nested series with a null field must work."""
    series = _null_nested_series()
    series[0] = series[1]
    _validate_struct(series.array)
    assert series.nest.to_flat()["v"].tolist() == [20, 20, 30, 40]


def test_from_sequence_null_field():
    """Rebuilding from per-row nested frames must preserve the null field."""
    series = _null_nested_series()
    rebuilt = NestedExtensionArray._from_sequence(list(series), dtype=series.dtype)
    _validate_struct(rebuilt)
    struct_array = rebuilt.struct_array
    if isinstance(struct_array, pa.ChunkedArray):
        struct_array = struct_array.combine_chunks()
    assert struct_array.field("n").values.null_count == 4
