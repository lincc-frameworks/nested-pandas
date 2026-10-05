"""Shared pieces of the custom cell rendering in NestedFrame HTML reprs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["MAX_RENDERED", "PLACEHOLDER_HTML", "CappedColumnFormatter"]

MAX_RENDERED = 10
"""Number of cells rendered in full per column in a NestedFrame HTML repr; later cells show a placeholder."""

PLACEHOLDER_HTML = '<span style="color:#888;">&lt;not rendered in preview&gt;</span>'
"""HTML shown for cells past :data:`MAX_RENDERED`."""


class CappedColumnFormatter:
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
    """

    def __init__(
        self,
        cell_html: Callable[[Any], str],
        is_rendered: Callable[[Any], bool],
        max_rendered: int = MAX_RENDERED,
    ) -> None:
        self.cell_html = cell_html
        self.is_rendered = is_rendered
        self.max_rendered = max_rendered
        self.rendered = 0
        """Number of cells rendered in full so far."""

    def __call__(self, value: Any) -> str:
        """HTML for one cell: its full rendering, or the placeholder once the limit is reached."""
        if self.is_rendered(value):
            if self.rendered >= self.max_rendered:
                return PLACEHOLDER_HTML
            self.rendered += 1
        return self.cell_html(value)
