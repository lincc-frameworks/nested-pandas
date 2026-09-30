"""Shared pieces of the custom cell rendering in NestedFrame HTML reprs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["MAX_RENDERED", "PLACEHOLDER_HTML", "capped_column_formatter"]

MAX_RENDERED = 10
"""Number of cells rendered in full per column in a NestedFrame HTML repr; later cells show a placeholder."""

PLACEHOLDER_HTML = '<span style="color:#888;">&lt;not rendered in preview&gt;</span>'
"""HTML shown for cells past :data:`MAX_RENDERED`."""


def capped_column_formatter(
    cell_html: Callable[[Any], str], is_rendered: Callable[[Any], bool], max_rendered: int = MAX_RENDERED
) -> Callable[[Any], str]:
    """A cell formatter for one column that renders at most ``max_rendered`` cells in full.

    Cells for which ``is_rendered`` is True count towards the limit and show
    :data:`PLACEHOLDER_HTML` once it is reached; all other cells, such as
    missing values, always go through ``cell_html``. The formatter counts
    the cells it has rendered, so a new one is needed for every column of
    every repr.

    Parameters
    ----------
    cell_html : Callable
        Renders one cell value to HTML.
    is_rendered : Callable
        Whether a cell value is a full rendering that should count.
    max_rendered : int, default MAX_RENDERED
        Number of cells to render in full.

    Returns
    -------
    Callable
        Formatter taking a cell value and returning its HTML.
    """
    rendered = 0

    def format_cell(value: Any) -> str:
        nonlocal rendered
        if is_rendered(value):
            if rendered >= max_rendered:
                return PLACEHOLDER_HTML
            rendered += 1
        return cell_html(value)

    return format_cell
