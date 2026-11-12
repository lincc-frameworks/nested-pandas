"""Resolution of column names like ``"nested.field"`` against the columns of a frame.

A column of a ``NestedFrame`` is addressed either by its own label or, for a
sub-column of a nested column, by a dot-delimited path through the nested
columns containing it: ``"nested.field"``, ``"outer.inner.field"``, and so on
for any depth. :func:`resolve_column_path` is the single place that decides
what such a name refers to.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from nested_pandas.series.dtype import NestedDtype

__all__ = ["ColumnPath", "resolve_column_path"]


@dataclass(frozen=True)
class ColumnPath:
    """Location of a column: a column of the frame, or a sub-column of a nested column at any depth.

    Parameters
    ----------
    column : Hashable
        Label of the column of the frame the path starts at.
    fields : tuple of str
        Sub-column names leading from ``column`` down to the addressed
        sub-column, one per level of nesting. Empty for a column of the frame.
    dtypes : tuple
        The pandas dtype at every step of the path: of ``column`` first, then
        of each of ``fields``. The last one is None for a sub-column that
        does not exist yet. Not part of the path's identity.
    """

    column: Hashable
    fields: tuple[str, ...] = ()
    dtypes: tuple[Any, ...] = field(default=(), compare=False, repr=False)

    @property
    def is_base(self) -> bool:
        """Whether the path addresses a column of the frame itself, not a sub-column."""
        return not self.fields

    @property
    def leaf(self) -> Hashable:
        """Name of the addressed column within its parent: the last field, or the frame's column label."""
        return self.fields[-1] if self.fields else self.column

    @property
    def dtype(self) -> Any:
        """The pandas dtype of the addressed column, None if it does not exist yet."""
        return self.dtypes[-1] if self.dtypes else None

    @property
    def is_nested(self) -> bool:
        """Whether the addressed column is itself a nested column."""
        return isinstance(self.dtype, NestedDtype)

    @property
    def parent(self) -> ColumnPath:
        """Path of the nested column that directly contains this sub-column.

        Raises
        ------
        ValueError
            For a column of the frame, which is not contained in a nested column.
        """
        if self.is_base:
            raise ValueError(f"Column '{self.column}' is not a sub-column of a nested column")
        return ColumnPath(self.column, self.fields[:-1], self.dtypes[:-1])

    @property
    def name(self) -> str:
        """The dot-delimited name of the path, which resolves back to it."""
        return ".".join([str(self.column), *self.fields])


def resolve_column_path(
    frame: pd.DataFrame, components: Sequence[str], *, allow_new_field: bool = False
) -> ColumnPath | None:
    """Find the column that the components of a dot-delimited name refer to.

    Names may themselves contain dots, both column labels of the frame and
    sub-column names, so at every level the longest run of components that
    names an existing column is tried first. In particular, a column of the
    frame whose label is the whole name always wins over a sub-column.

    Parameters
    ----------
    frame : pd.DataFrame
        The frame whose columns are addressed.
    components : sequence of str
        The name split on its delimiter, with any aliases already replaced by
        the original names.
    allow_new_field : bool, default False
        Whether the last component may name a sub-column that does not exist
        yet, of an existing nested column. Used for assignment. The resulting
        path has None as its ``dtype``.

    Returns
    -------
    ColumnPath or None
        The resolved path, or None if the name does not refer to a column.
    """
    columns = frame.columns
    dtypes = None
    for n_column in range(len(components), 0, -1):
        column = ".".join(components[:n_column])
        if column not in columns:
            continue
        if dtypes is None:
            dtypes = frame.dtypes
        dtype = dtypes[column]
        rest = components[n_column:]
        if not rest:
            return ColumnPath(column, (), (dtype,))
        if isinstance(dtype, NestedDtype):
            resolved = _resolve_fields(dtype, rest, allow_new_field)
            if resolved is not None:
                fields, field_dtypes = resolved
                return ColumnPath(column, fields, (dtype, *field_dtypes))
    return None


def _resolve_fields(
    dtype: NestedDtype, components: Sequence[str], allow_new_field: bool
) -> tuple[tuple[str, ...], tuple[Any, ...]] | None:
    """Resolve components to sub-column names below a nested column of ``dtype``, with their dtypes."""
    known = dtype.column_dtypes
    for n_field in range(len(components), 0, -1):
        name = ".".join(components[:n_field])
        if name not in known:
            continue
        field_dtype = dtype.column_dtype(name)
        rest = components[n_field:]
        if not rest:
            return (name,), (field_dtype,)
        if isinstance(field_dtype, NestedDtype):
            resolved = _resolve_fields(field_dtype, rest, allow_new_field)
            if resolved is not None:
                return (name, *resolved[0]), (field_dtype, *resolved[1])
    if allow_new_field and len(components) == 1:
        return (components[0],), (None,)
    return None
