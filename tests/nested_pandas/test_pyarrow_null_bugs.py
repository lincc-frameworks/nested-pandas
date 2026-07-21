"""Reproductions of upstream pyarrow limitations affecting ``null``-typed nested fields.

nested-pandas represents an all-null nested sub-column with pyarrow's ``null``
data type. A few pyarrow limitations affect such fields, and nested-pandas has
to work around them:

* https://github.com/apache/arrow/issues/43838 -- casting a ``null``-typed list
  to its own type silently produces an invalid array (the null values buffer is
  truncated to the row count instead of the flat element count). This is data
  corruption -- a real bug.
* https://github.com/apache/arrow/issues/44881 -- ``Table.to_pandas`` with
  ``types_mapper=pd.ArrowDtype`` corrupts a ``null``-typed nested field the same
  way. Also corruption.
* Building a struct array from per-row scalars raises
  ``ArrowNotImplementedError: AppendScalar for type null`` when one of the
  struct's fields is ``null``-typed. This is not corruption -- pyarrow simply
  hasn't implemented scalar-append for a ``null``-typed builder yet -- but
  nested-pandas still needs to avoid that code path (e.g. ``__setitem__`` and
  ``_from_sequence`` go through it today).

These tests assert the behavior nested-pandas needs, so they fail while the
upstream limitations are present. They act as canaries: once pyarrow adds the
missing support, they start passing (xpass) and the corresponding
nested-pandas workarounds can be removed.
"""

import pandas as pd
import pyarrow as pa
import pytest


def _combine(array):
    return array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array


@pytest.mark.xfail(
    strict=True,
    reason="upstream pyarrow bug: https://github.com/apache/arrow/issues/43838",
    raises=pa.ArrowInvalid,
)
def test_pyarrow_43838_null_list_identity_cast():
    """https://github.com/apache/arrow/issues/43838

    A valid ``list<null>`` (offsets ``[0, 1, 2, 4]`` over 4 nulls) must survive a
    cast to its own type. pyarrow instead truncates the null values buffer to the
    number of rows (3), leaving offsets that overrun the values.

    Also reproduces for ``pa.Scalar.cast()`` (not just ``pa.Array.cast()``),
    confirmed manually: casting a struct scalar with a ``large_list<null>``
    field to its own type truncates that field the same way. Likely the same
    root cause reached through a different API surface -- no separate upstream
    issue filed for the scalar variant.

    Workarounds, all of which avoid ever casting a null-typed child to its own
    type (an identity cast is a no-op, so the child can just be reused as-is):

    * ``utils.safe_cast`` -- the general fix; recurses into struct/list
      children and skips the cast when a child's type already matches.
    * ``NestedExtensionArray.__arrow_array__`` -- calls ``safe_cast`` instead of
      ``.cast()``.
    * ``NestedExtensionArray._box_pa_array`` -- calls ``safe_cast`` for its final
      cast, and (via ``scalars_to_pa_array``) for reconciling per-row list
      values before concatenating them.
    * ``convert_df_to_pa_scalar`` -- calls ``safe_cast`` on a column's array
      before wrapping it in a scalar, instead of casting the assembled struct
      scalar afterwards (which hit the ``Scalar.cast()`` variant above).

    If this test starts passing (xpass), these call sites' ``safe_cast`` calls
    can revert to plain ``.cast()``, and ``safe_cast`` itself can likely be
    removed.
    """
    values = pa.nulls(4, pa.null())
    array = pa.ListArray.from_arrays(pa.array([0, 1, 2, 4], pa.int32()), values)
    array.validate(full=True)  # input is valid

    result = array.cast(array.type)  # identity cast -- should be a no-op

    result.validate(full=True)
    assert result.to_pylist() == [[None], [None], [None, None]]


@pytest.mark.xfail(
    strict=True,
    reason="upstream pyarrow bug: https://github.com/apache/arrow/issues/44881",
    raises=pa.ArrowInvalid,
)
def test_pyarrow_44881_to_pandas_arrowdtype_null_field():
    """https://github.com/apache/arrow/issues/44881

    A struct-of-lists with a ``null``-typed field must survive the
    ``Table.to_pandas(types_mapper=pd.ArrowDtype)`` round-trip that nested-pandas
    relies on. pyarrow truncates the null field's values buffer during the
    conversion.

    Workaround: ``nestedframe/io.py::_cast_struct_cols_to_nested`` (branch
    ``fix-507-null-nested-parquet``, PR #507) builds nested columns straight
    from the source pyarrow ``Table`` instead of from the corrupted
    ``pd.ArrowDtype`` column that ``Table.to_pandas`` produced. If this test
    starts passing (xpass), that workaround can be simplified back to using
    the ``to_pandas`` output directly.
    """
    offsets = pa.array([0, 1, 2, 4], pa.int32())
    struct = pa.StructArray.from_arrays(
        [
            pa.ListArray.from_arrays(offsets, pa.nulls(4, pa.null())),
            pa.ListArray.from_arrays(offsets, pa.array([1, 2, 3, 4], pa.int64())),
        ],
        names=["n", "v"],
    )
    table = pa.table({"nested": struct})

    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    roundtrip = _combine(pa.array(df["nested"]))

    roundtrip.validate(full=True)
    assert roundtrip.field("n").values.null_count == 4


@pytest.mark.xfail(
    strict=True,
    reason="pyarrow limitation: StructBuilder can't AppendScalar into a null-typed child field",
    raises=pa.ArrowNotImplementedError,
)
def test_pyarrow_struct_array_from_null_field_scalars():
    """Building a struct array from per-row scalars should support a null-typed field.

    ``pa.array([scalar, scalar])`` works for a plain ``null``-typed scalar, and
    for a struct scalar whose fields are all non-null. But when a struct scalar
    has a ``null``-typed field, pyarrow raises
    ``ArrowNotImplementedError: AppendScalar for type null`` instead of building
    the (perfectly valid) array. No apache/arrow issue is filed for this yet --
    it's a missing feature, not corruption.

    Workarounds, both of which avoid ever appending a struct/list scalar with a
    null-typed child one at a time:

    * ``utils.scalars_to_pa_array`` -- rebuilds struct/list containers directly
      from each scalar's already-materialized child array/scalar instead of
      going through ``pa.array(scalars)``. Used by
      ``NestedExtensionArray._box_pa_array`` (its per-row-scalar fallback path,
      reached from ``_from_sequence``).
    * ``NestedExtensionArray.__setitem__`` -- broadcasts a single scalar via
      ``pa.repeat(scalar, n)`` instead of ``pa.array([scalar] * n)``.

    If this test starts passing (xpass), both call sites can revert to plain
    ``pa.array(...)``, and ``scalars_to_pa_array`` can likely be removed.
    """
    struct_type = pa.struct([("n", pa.null()), ("v", pa.int64())])
    row = pa.scalar({"n": None, "v": 1}, type=struct_type)

    result = pa.array([row, row])

    result.validate(full=True)
    assert result.to_pylist() == [{"n": None, "v": 1}, {"n": None, "v": 1}]
