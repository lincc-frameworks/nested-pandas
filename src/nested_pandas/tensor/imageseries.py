"""Image columns: the shared image series API and the tensor-backed implementation.

An image column is a column whose cells are image pixel arrays (2-d cutout
stamps, or 3-d stacks with a leading plane/band axis). Two backings share one
user-facing API through :class:`BaseImageSeries`:

- :class:`ImageSeries` — tensor-backed: pixels materialized in the column
  (:class:`~nested_pandas.tensor.dtype.TensorDtype` machinery underneath).
- :class:`~nested_pandas.tensor.cutouts.CutoutSeries` — store-backed:
  the column holds lightweight cutout descriptors resolved lazily against a
  shared :class:`~nested_pandas.tensor.cutouts.ImageStore`.

Both serialize to the same thing: the ``nested_pandas.image`` arrow extension
type (store-backed columns materialize their pixels on write), so any saved
image column reads back as a tensor-backed :class:`ImageSeries`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.extensions import register_extension_dtype

from nested_pandas.series.registry import register_html_formatter, register_series_class
from nested_pandas.tensor.arrow_ext import ImageType
from nested_pandas.tensor.display import image_cell_html, image_series_html
from nested_pandas.tensor.dtype import TensorDtype
from nested_pandas.tensor.tensorseries import TensorSeries

__all__ = ["BaseImageSeries", "ImageDtype", "ImageSeries"]


class BaseImageSeries(pd.Series):
    """Shared API for image columns, whatever the pixel backing.

    Subclasses pair with a dtype whose extension array exposes ``to_stack``,
    ``shapes``, and ndarray scalar access; everything here delegates to the
    array, so tensor-backed and store-backed columns behave the same.
    """

    def _is_image_column(self) -> bool:
        """Whether the underlying array supports the image column API."""
        return hasattr(self.array, "to_stack") and hasattr(self.array, "shapes")

    def _require_image_column(self):
        if not self._is_image_column():
            raise TypeError(f"not an image column: dtype is '{self.dtype}'")

    def to_stack(self, na_value=np.nan) -> np.ndarray:
        """Convert to a single (n, height, width, ...) numpy pixel block.

        Missing rows are filled with ``na_value``. Only available when all
        rows share one shape (always true for fixed-shape image columns).
        """
        self._require_image_column()
        return self.array.to_stack(na_value=na_value)

    def to_image_stack(self, na_value=np.nan) -> np.ndarray:
        """Alias of :meth:`to_stack`."""
        return self.to_stack(na_value=na_value)

    @property
    def shapes(self) -> np.ndarray:
        """Per-row image shapes as an (n, ndim) integer array."""
        self._require_image_column()
        return self.array.shapes

    def _repr_html_(self) -> str | None:
        """HTML repr with image thumbnails for the first rows (used by notebooks)."""
        if not self._is_image_column():
            return None
        return image_series_html(self)


@register_extension_dtype
class ImageDtype(TensorDtype):
    """Data type for columns of materialized image pixel arrays.

    Behaves exactly like ``TensorDtype`` (fixed shape via ``shape``, ragged
    via ``ndim``), but marks the column as image data: it is returned as an
    :class:`ImageSeries`, renders cells as thumbnails in HTML reprs, and its
    dtype string uses the ``image`` prefix, e.g. ``"image[float, (25, 25)]"``.

    Image columns are serialized as the ``nested_pandas.image`` extension
    type (never the canonical ``arrow.fixed_shape_tensor``, whose name cannot
    carry the image identity), so they read back as image columns.
    """

    _name_prefix = "image"
    _arrow_type_class = ImageType


class ImageSeries(TensorSeries, BaseImageSeries):
    """A Series of materialized image pixel arrays, one numpy ndarray per row.

    The tensor-backed image column: all of :class:`TensorSeries` plus the
    shared :class:`BaseImageSeries` image API.
    """


register_series_class(ImageDtype, ImageSeries)
register_html_formatter(ImageDtype, image_cell_html)
