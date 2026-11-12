"""Registry of the kinds of columns nested-pandas gives special treatment: nested and tensor columns.

Such a column may be a column of a ``NestedFrame`` or a sub-column of a
nested column, at any depth. Wherever it appears, the same questions are
asked about it, and this module is the single place that answers them:

- which pandas extension dtype represents a pyarrow type
  (:func:`pandas_dtype_from_arrow`), and back (:func:`arrow_type_from_pandas_dtype`,
  :func:`arrow_type_from_string`),
- which ``pd.Series`` subclass a series of that dtype is returned as
  (:func:`wrap_series`),
- how its cells are rendered in HTML (:func:`html_cell_formatter`) and in
  text (:func:`text_cell_formatter`) reprs.

Each kind registers its own answers next to its implementation, so this
module imports none of them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pyarrow as pa
from pandas.api.extensions import ExtensionDtype

__all__ = [
    "arrow_type_from_pandas_dtype",
    "arrow_type_from_string",
    "has_special_dtype",
    "html_cell_formatter",
    "pandas_dtype_from_arrow",
    "register_arrow_dtype",
    "register_html_formatter",
    "register_series_class",
    "register_text_formatter",
    "text_cell_formatter",
    "wrap_series",
]

_ARROW_DTYPES: list[tuple[Callable[[pa.DataType], bool], Callable[[pa.DataType], ExtensionDtype]]] = []
_ARROW_TYPES: dict[type[ExtensionDtype], Callable[[Any], pa.DataType]] = {}
_SERIES_CLASSES: dict[type[ExtensionDtype], type[pd.Series]] = {}
_HTML_FORMATTERS: dict[type[ExtensionDtype], Callable[[], Callable[[Any], str]]] = {}
_TEXT_FORMATTERS: dict[type[ExtensionDtype], Callable[[Any], str]] = {}


def register_arrow_dtype(
    dtype_class: type[ExtensionDtype],
    matches: Callable[[pa.DataType], bool],
    to_arrow: Callable[[Any], pa.DataType],
) -> None:
    """Register the pandas extension dtype representing some pyarrow types, in both directions.

    Parameters
    ----------
    dtype_class : type[ExtensionDtype]
        The extension dtype. Called with a matching pyarrow type, it builds
        the dtype representing it. It may raise ``NotImplementedError`` or
        ``ValueError`` for a type it does not support, which then stays a
        ``pd.ArrowDtype``.
    matches : Callable
        Whether a pyarrow type is represented by the dtype.
    to_arrow : Callable
        The pyarrow type of the values of a column of a given dtype of
        ``dtype_class``, the inverse of building the dtype from it.
    """
    _ARROW_DTYPES.append((matches, dtype_class))
    _ARROW_TYPES[dtype_class] = to_arrow


def pandas_dtype_from_arrow(pa_type: pa.DataType) -> ExtensionDtype:
    """The pandas dtype of a column of values of a pyarrow type.

    Parameters
    ----------
    pa_type : pa.DataType
        The pyarrow type of the values.

    Returns
    -------
    ExtensionDtype
        The registered extension dtype for the type, such as ``NestedDtype``
        for a list of structs and ``TensorDtype`` for a fixed-shape tensor,
        otherwise ``pd.ArrowDtype``.
    """
    for matches, dtype_from_arrow in _ARROW_DTYPES:
        if matches(pa_type):
            try:
                return dtype_from_arrow(pa_type)
            except (NotImplementedError, ValueError):
                break
    return pd.ArrowDtype(pa_type)


def arrow_type_from_pandas_dtype(dtype: ExtensionDtype) -> pa.DataType:
    """The pyarrow type of the values of a column of a pandas extension dtype.

    The inverse of :func:`pandas_dtype_from_arrow`. Extension dtypes which
    are not registered here, such as those of pandas itself or of other
    packages, are asked through the arrow conversion of their arrays.

    Raises
    ------
    TypeError
        If the dtype has no pyarrow representation.
    """
    if isinstance(dtype, pd.ArrowDtype):
        return dtype.pyarrow_dtype
    for dtype_class, to_arrow in _ARROW_TYPES.items():
        if isinstance(dtype, dtype_class):
            return to_arrow(dtype)
    try:
        return pa.array(dtype.construct_array_type()._from_sequence([], dtype=dtype)).type
    except (TypeError, ValueError, NotImplementedError, pa.ArrowException) as e:
        raise TypeError(f"The pandas dtype '{dtype}' has no pyarrow type") from e


def arrow_type_from_string(string: str) -> pa.DataType:
    """Parse the type of the values of a column, as nested and tensor dtype names print it.

    The string is the name of a pyarrow type, such as ``double`` or
    ``timestamp[ns]``, or of a pandas extension dtype, such as
    ``nested<x: [int64]>`` or ``tensor[double, (2, 2)]``. Extension dtype
    names are parsed by pandas, so those of any package which registered its
    dtype with pandas are understood, and so are the parametric pyarrow types
    ``pd.ArrowDtype`` parses.

    Raises
    ------
    TypeError
        If the string is not understood.
    """
    # The plain pyarrow names first: pandas reads some of them as its own dtypes, "string" for one
    try:
        return pa.type_for_alias(string)
    except ValueError:
        pass
    # Then pandas' registry of extension dtypes, first for the name of one, then for a pyarrow type
    # with parameters, which pandas names with a "[pyarrow]" suffix
    for name in (string, f"{string}[pyarrow]"):
        try:
            dtype = pd.api.types.pandas_dtype(name)
        except (TypeError, NotImplementedError):
            continue
        if isinstance(dtype, ExtensionDtype):
            return arrow_type_from_pandas_dtype(dtype)
    raise TypeError(
        f"'{string}' is neither the name of a pyarrow type pandas can parse nor of a pandas extension dtype"
    )


def has_special_dtype(dtypes) -> bool:
    """Whether any of the dtypes is of a registered kind, so its columns need a ``NestedFrame`` to be
    returned as their Series subclass."""
    return any(isinstance(dtype, tuple(_SERIES_CLASSES)) for dtype in dtypes)


def register_series_class(dtype_class: type[ExtensionDtype], series_class: type[pd.Series]) -> None:
    """Register the Series subclass that series of an extension dtype are returned as."""
    _SERIES_CLASSES[dtype_class] = series_class


def wrap_series(series: pd.Series) -> pd.Series:
    """Return a series as the Series subclass registered for its dtype, or unchanged if there is none."""
    for dtype_class, series_class in _SERIES_CLASSES.items():
        if isinstance(series.dtype, dtype_class):
            if isinstance(series, series_class):
                return series
            return series_class(series, copy=False)
    return series


def register_html_formatter(
    dtype_class: type[ExtensionDtype], formatter_factory: Callable[[], Callable[[Any], str]]
) -> None:
    """Register the HTML rendering of the cells of columns of an extension dtype.

    ``formatter_factory`` makes a cell formatter, taking a cell value and
    returning its HTML. A fresh formatter is made per column and per repr,
    so it may keep state, such as a count of the cells it rendered.
    """
    _HTML_FORMATTERS[dtype_class] = formatter_factory


def html_cell_formatter(dtype: Any) -> Callable[[Any], str] | None:
    """A new HTML cell formatter for columns of ``dtype``, or None if pandas' own rendering is used."""
    for dtype_class, formatter_factory in _HTML_FORMATTERS.items():
        if isinstance(dtype, dtype_class):
            return formatter_factory()
    return None


def register_text_formatter(dtype_class: type[ExtensionDtype], formatter: Callable[[Any], str]) -> None:
    """Register the short text rendering of one value of an extension dtype, used when the value is
    shown inside the text repr of a nested column."""
    _TEXT_FORMATTERS[dtype_class] = formatter


def text_cell_formatter(dtype: Any) -> Callable[[Any], str] | None:
    """The text formatter for values of ``dtype``, or None if there is none registered."""
    for dtype_class, formatter in _TEXT_FORMATTERS.items():
        if isinstance(dtype, dtype_class):
            return formatter
    return None
