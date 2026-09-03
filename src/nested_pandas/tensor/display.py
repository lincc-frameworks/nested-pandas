"""Text and HTML rendering for tensor and image columns.

HTML reprs render 2-d cells as inline PNG thumbnails (the first
:data:`MAX_RENDERED` rows of a series repr; every displayed cell in
``NestedFrame`` reprs, where pandas' own row truncation bounds the count).
Image columns render in grayscale; plain tensor columns render with the
viridis colormap and a colorbar beside each thumbnail, labelled with the
displayed value range. Text reprs and non-2-d cells fall back to the compact
``[h×w] dtype`` descriptor. Rendering needs matplotlib; without it, cells
degrade to the descriptor text.
"""

from __future__ import annotations

import base64
import html as html_module
import io
from functools import cache

import numpy as np
import pandas as pd

__all__ = [
    "MAX_RENDERED",
    "IMAGE_CMAP",
    "TENSOR_CMAP",
    "render_png_base64",
    "image_series_html",
    "image_cell_html",
    "tensor_series_html",
    "tensor_cell_html",
]

# Number of rows rendered as actual images in series HTML previews.
MAX_RENDERED = 10

# Colormaps used for image and plain tensor thumbnails.
IMAGE_CMAP = "gray"
TENSOR_CMAP = "viridis"

_THUMBNAIL_SIZE = 64
_THUMBNAIL_STYLE = f"width:{_THUMBNAIL_SIZE}px;image-rendering:pixelated;"
_PLACEHOLDER_HTML = '<span style="color:#888;">&lt;not rendered in preview&gt;</span>'

# Colorbar: a thumbnail-height gradient strip with the top/bottom values beside it.
_COLORBAR_STEPS = 64
_COLORBAR_STYLE = f"width:8px;height:{_THUMBNAIL_SIZE}px;"
_COLORBAR_LABELS_STYLE = (
    f"display:inline-flex;flex-direction:column;justify-content:space-between;"
    f"height:{_THUMBNAIL_SIZE}px;font-size:9px;line-height:1;font-family:monospace;"
)
_CELL_STYLE = "display:inline-flex;align-items:flex-start;gap:3px;"


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
        from matplotlib import image as mpl_image  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    buffer = io.BytesIO()
    mpl_image.imsave(buffer, data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", format="png")
    return base64.b64encode(buffer.getvalue()).decode()


def render_png_base64(data: np.ndarray, cmap: str = IMAGE_CMAP) -> str | None:
    """Render a 2D array as a base64-encoded PNG.

    Pixel values are clipped to the 1st-99th percentile before rendering.

    Parameters
    ----------
    data : np.ndarray
        2D pixel array.
    cmap : str
        Matplotlib colormap name; grayscale by default.

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


def _cell_html(value, rendered: bool, cmap: str = IMAGE_CMAP, colorbar: bool = False) -> str:
    """HTML for a single cell: a thumbnail (with optional colorbar), a placeholder, or descriptor text."""
    if value is pd.NA or value is None:
        return "&lt;NA&gt;"
    descriptor = html_module.escape(_descriptor_text(value), quote=True)
    if not rendered:
        return _PLACEHOLDER_HTML
    if value.ndim != 2:
        return descriptor
    data = np.asarray(value, dtype=float)
    vmin, vmax = _display_range(data)
    png = _imsave_png_base64(data, cmap, vmin, vmax)
    if png is None:  # matplotlib unavailable
        return descriptor
    thumbnail = f'<img src="data:image/png;base64,{png}" style="{_THUMBNAIL_STYLE}" title="{descriptor}"/>'
    if not colorbar:
        return thumbnail
    return f'<span style="{_CELL_STYLE}">{thumbnail}{_colorbar_html(cmap, vmin, vmax)}</span>'


def _series_html(series: pd.Series, cmap: str, colorbar: bool) -> str:
    """HTML table for a tensor-like series: thumbnails for the first rows, descriptors for all."""
    rows = []
    for position in range(len(series)):
        value = series.array[position]
        preview = _cell_html(value, rendered=position < MAX_RENDERED, cmap=cmap, colorbar=colorbar)
        descriptor = "" if value is pd.NA else html_module.escape(_descriptor_text(value))
        rows.append(
            f"<tr><th>{html_module.escape(str(series.index[position]))}</th>"
            f"<td>{preview}</td><td>{descriptor}</td></tr>"
        )
    name = html_module.escape(str(series.name)) if series.name is not None else ""
    header = f"<tr><th></th><th>{name}</th><th></th></tr>"
    footer = f"<p>Length: {len(series)}, dtype: {series.dtype}</p>"
    return f"<table>{header}{''.join(rows)}</table>{footer}"


def image_series_html(series: pd.Series) -> str:
    """HTML repr for an image series: grayscale thumbnails for the first rows.

    Parameters
    ----------
    series : pd.Series
        A series of image dtype.

    Returns
    -------
    str
    """
    return _series_html(series, cmap=IMAGE_CMAP, colorbar=False)


def tensor_series_html(series: pd.Series) -> str:
    """HTML repr for a tensor series: viridis thumbnails with colorbars for the first 2-d rows.

    Parameters
    ----------
    series : pd.Series
        A series of tensor dtype.

    Returns
    -------
    str
    """
    return _series_html(series, cmap=TENSOR_CMAP, colorbar=True)


def image_cell_html(value) -> str:
    """Cell HTML formatter for image columns in NestedFrame reprs.

    Registered with nested-pandas (`register_html_formatter`), which applies
    it to every *displayed* image cell — pandas' own row truncation
    (``display.max_rows``/``display.min_rows``) bounds how many thumbnails
    are rendered.

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
    return _cell_html(value, rendered=True, cmap=IMAGE_CMAP, colorbar=False)


def tensor_cell_html(value) -> str:
    """Cell HTML formatter for plain tensor columns in NestedFrame reprs.

    Like :func:`image_cell_html`, but 2-d cells render through the viridis
    colormap with a colorbar labelled with the displayed value range.

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
    return _cell_html(value, rendered=True, cmap=TENSOR_CMAP, colorbar=True)
