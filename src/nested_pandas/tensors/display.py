"""HTML rendering for tensor columns.

Small tensors, up to :data:`TENSOR_FORMATTING_MAX_ELEMENTS` elements, show
their values, one line per row. Larger 2-d tensors render as inline PNG
thumbnails through the viridis colormap, with a colorbar beside each
thumbnail labelled with the displayed value range, and larger tensors of
any other dimensionality show the compact ``[h×w] dtype`` descriptor.
:func:`tensor_cell_html` formats one cell that way. A ``NestedFrame`` HTML
repr formats each tensor column with :func:`tensor_column_formatter`, which
renders thumbnails for the first :data:`~nested_pandas.display.MAX_RENDERED`
cells that would get one and shows a placeholder for the rest, so a long
repr stays small.
Thumbnails need matplotlib; without it, those cells degrade to the
descriptor text.
"""

from __future__ import annotations

import base64
import html as html_module
import io
from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np

from nested_pandas.display import MAX_RENDERED, capped_column_formatter
from nested_pandas.tensors.ext_array import TENSOR_FORMATTING_MAX_ELEMENTS

__all__ = [
    "TENSOR_CMAP",
    "render_png_base64",
    "tensor_cell_html",
    "tensor_column_formatter",
]

TENSOR_CMAP = "viridis"
"""Matplotlib colormap of the thumbnails."""

_THUMBNAIL_SIZE = 64
_THUMBNAIL_STYLE = f"width:{_THUMBNAIL_SIZE}px;image-rendering:pixelated;"

# Colorbar: a thumbnail-height gradient strip with the top/bottom values beside it
_COLORBAR_STEPS = 64
_COLORBAR_STYLE = f"width:8px;height:{_THUMBNAIL_SIZE}px;"
_COLORBAR_LABELS_STYLE = (
    f"display:inline-flex;flex-direction:column;justify-content:space-between;"
    f"height:{_THUMBNAIL_SIZE}px;font-size:9px;line-height:1;font-family:monospace;"
)
_CELL_STYLE = "display:inline-flex;align-items:flex-start;gap:3px;"
_VALUES_STYLE = "margin:0;font-family:monospace;text-align:left;"


def _descriptor_text(value: np.ndarray) -> str:
    return f"[{'×'.join(str(size) for size in value.shape)}] {value.dtype}"


def _display_range(data: np.ndarray) -> tuple[float, float]:
    """Value range shown for ``data``: the 1st-99th percentile of its finite values."""
    finite = data[np.isfinite(data)]
    if finite.size:
        vmin, vmax = (float(v) for v in np.percentile(finite, [1, 99]))
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _imsave_png_base64(data: np.ndarray, cmap: str, vmin: float, vmax: float) -> str | None:
    """Base64 PNG of a 2-d array through a matplotlib colormap, or None without matplotlib."""
    try:
        from matplotlib import image as mpl_image
    except ImportError:
        return None
    buffer = io.BytesIO()
    mpl_image.imsave(buffer, data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", format="png")
    return base64.b64encode(buffer.getvalue()).decode()


def render_png_base64(data: np.ndarray, cmap: str = TENSOR_CMAP) -> str | None:
    """Render a 2-d array as a base64-encoded PNG.

    Pixel values are clipped to the 1st-99th percentile before rendering.

    Parameters
    ----------
    data : np.ndarray
        2-d array.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    str or None
        Base64-encoded PNG bytes, or None if matplotlib is unavailable.
    """
    data = np.asarray(data, dtype=float)
    vmin, vmax = _display_range(data)
    return _imsave_png_base64(data, cmap, vmin, vmax)


@cache
def _colorbar_png_base64(cmap: str) -> str | None:
    """Base64 PNG of a vertical colormap gradient (high values at the top); cached per colormap."""
    gradient = np.linspace(0.0, 1.0, _COLORBAR_STEPS)[:, np.newaxis]
    return _imsave_png_base64(gradient, cmap, 0.0, 1.0)


def _format_label(value: float) -> str:
    return html_module.escape(f"{value:.3g}")


def _colorbar_html(cmap: str, vmin: float, vmax: float) -> str:
    """HTML for a colorbar strip labelled with the top and bottom of the displayed range."""
    png = _colorbar_png_base64(cmap)
    if png is None:
        return ""
    return (
        f'<img src="data:image/png;base64,{png}" style="{_COLORBAR_STYLE}" title="colorbar"/>'
        f'<span style="{_COLORBAR_LABELS_STYLE}">'
        f"<span>{_format_label(vmax)}</span><span>{_format_label(vmin)}</span></span>"
    )


def _values_html(value: np.ndarray) -> str:
    """The values of a small tensor, one line per row as numpy prints them."""
    return f'<pre style="{_VALUES_STYLE}">{html_module.escape(np.array2string(value))}</pre>'


def _wants_thumbnail(value: np.ndarray) -> bool:
    """Whether a tensor is rendered as a thumbnail: 2-d and too large to show its values."""
    return value.ndim == 2 and value.size > TENSOR_FORMATTING_MAX_ELEMENTS


def _cell_html(value: np.ndarray) -> str:
    """HTML for a single tensor: its values if small, else a thumbnail with a colorbar if 2-d, else
    descriptor text."""
    if value.size <= TENSOR_FORMATTING_MAX_ELEMENTS:
        return _values_html(value)
    descriptor = html_module.escape(_descriptor_text(value), quote=True)
    if value.ndim != 2:
        return descriptor
    data = np.asarray(value, dtype=float)
    vmin, vmax = _display_range(data)
    png = _imsave_png_base64(data, TENSOR_CMAP, vmin, vmax)
    if png is None:  # matplotlib unavailable
        return descriptor
    thumbnail = f'<img src="data:image/png;base64,{png}" style="{_THUMBNAIL_STYLE}" title="{descriptor}"/>'
    return f'<span style="{_CELL_STYLE}">{thumbnail}{_colorbar_html(TENSOR_CMAP, vmin, vmax)}</span>'


def tensor_cell_html(value) -> str:
    """Cell HTML formatter for tensor columns in ``NestedFrame`` HTML reprs.

    Applied to every *displayed* tensor cell; pandas' own row truncation
    (``display.max_rows``/``display.min_rows``) bounds how many are rendered.

    Parameters
    ----------
    value : np.ndarray or object
        The cell value; anything that is not an ndarray renders as NA.

    Returns
    -------
    str
        HTML for the cell.
    """
    if not isinstance(value, np.ndarray):
        return "&lt;NA&gt;"
    return _cell_html(value)


def tensor_column_formatter(max_rendered: int = MAX_RENDERED) -> Callable[[Any], str]:
    """Cell HTML formatter for one tensor column of a ``NestedFrame`` HTML repr.

    Like :func:`tensor_cell_html`, but only the first ``max_rendered``
    cells that would get a thumbnail are rendered; later ones show a
    placeholder instead, which keeps a long repr from embedding one PNG per
    row. Cells that show their values or a descriptor are not counted. A
    new formatter is needed for every repr, since it counts the cells it
    has rendered.

    Parameters
    ----------
    max_rendered : int, default MAX_RENDERED
        Number of thumbnails to render.

    Returns
    -------
    Callable
        Formatter taking a cell value and returning its HTML.
    """

    def is_thumbnail(value: Any) -> bool:
        return isinstance(value, np.ndarray) and _wants_thumbnail(value)

    return capped_column_formatter(tensor_cell_html, is_thumbnail, max_rendered)
