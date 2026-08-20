"""Text and HTML rendering for image columns.

HTML reprs render 2-d image cells as inline PNG thumbnails (the first
:data:`MAX_RENDERED` rows of a series repr; every displayed cell in
``NestedFrame`` reprs, where pandas' own row truncation bounds the count).
Text reprs and non-2-d cells fall back to the compact ``[h×w] dtype``
descriptor. Rendering needs matplotlib; without it, cells degrade to the
descriptor text.
"""

from __future__ import annotations

import base64
import html as html_module
import io

import numpy as np
import pandas as pd

__all__ = ["MAX_RENDERED", "render_png_base64", "image_series_html", "image_cell_html"]

# Number of rows rendered as actual images in series HTML previews.
MAX_RENDERED = 10

_THUMBNAIL_STYLE = "width:64px;image-rendering:pixelated;"
_PLACEHOLDER_HTML = '<span style="color:#888;">&lt;not rendered in preview&gt;</span>'


def _descriptor_text(value: np.ndarray) -> str:
    return f"[{'×'.join(str(size) for size in value.shape)}] {value.dtype}"


def render_png_base64(data: np.ndarray) -> str | None:
    """Render a 2D array as a base64-encoded grayscale PNG.

    Pixel values are clipped to the 1st-99th percentile before rendering.

    Parameters
    ----------
    data : np.ndarray
        2D pixel array.

    Returns
    -------
    str or None
        Base64-encoded PNG bytes, or None if matplotlib is unavailable.
    """
    try:
        from matplotlib import image as mpl_image  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    data = np.asarray(data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [1, 99])
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    buffer = io.BytesIO()
    mpl_image.imsave(buffer, data, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", format="png")
    return base64.b64encode(buffer.getvalue()).decode()


def _cell_html(value, rendered: bool) -> str:
    """HTML for a single image cell: a thumbnail, a placeholder, or descriptor text."""
    if value is pd.NA or value is None:
        return "&lt;NA&gt;"
    descriptor = html_module.escape(_descriptor_text(value), quote=True)
    if not rendered:
        return _PLACEHOLDER_HTML
    if value.ndim != 2:
        return descriptor
    png = render_png_base64(value)
    if png is None:  # matplotlib unavailable
        return descriptor
    return f'<img src="data:image/png;base64,{png}" style="{_THUMBNAIL_STYLE}" title="{descriptor}"/>'


def image_series_html(series: pd.Series) -> str:
    """HTML repr for an image series: thumbnails for the first rows.

    Parameters
    ----------
    series : pd.Series
        A series of image dtype.

    Returns
    -------
    str
    """
    rows = []
    for position in range(len(series)):
        value = series.array[position]
        preview = _cell_html(value, rendered=position < MAX_RENDERED)
        descriptor = "" if value is pd.NA else html_module.escape(_descriptor_text(value))
        rows.append(
            f"<tr><th>{html_module.escape(str(series.index[position]))}</th>"
            f"<td>{preview}</td><td>{descriptor}</td></tr>"
        )
    name = html_module.escape(str(series.name)) if series.name is not None else ""
    header = f"<tr><th></th><th>{name}</th><th></th></tr>"
    footer = f"<p>Length: {len(series)}, dtype: {series.dtype}</p>"
    return f"<table>{header}{''.join(rows)}</table>{footer}"


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
    return _cell_html(value, rendered=True)
