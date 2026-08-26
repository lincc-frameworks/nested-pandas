"""Image columns: tensor-backed image dtype and series.

An image column is a tensor column whose cells are image pixel arrays
(2-d cutout stamps, or 3-d stacks with a leading plane/band axis).
:class:`ImageDtype` subclasses :class:`TensorDtype` and :class:`ImageSeries`
subclasses :class:`TensorSeries`; the series-class registry dispatches on the
dtype's MRO, so image columns come back as ``ImageSeries`` while plain tensor
columns stay ``TensorSeries``. Image columns serialize as their own arrow
extension type (``nested_pandas.image``), so the identity survives
serialization in the type name itself.

This is the tensor-backed (materialized pixels) entry point of the image
column design. A store-backed variant — descriptor rows resolved against a
shared image store — can later be added as a sibling series class registered
for its own dtype, sharing the ``ImageSeries`` API.
"""

from __future__ import annotations

import numpy as np
from pandas.api.extensions import register_extension_dtype

from nested_pandas.series.registry import register_html_formatter, register_series_class
from nested_pandas.tensor.arrow_ext import ImageType
from nested_pandas.tensor.display import image_cell_html, image_series_html
from nested_pandas.tensor.dtype import TensorDtype
from nested_pandas.tensor.tensorseries import TensorSeries

__all__ = ["ImageDtype", "ImageSeries"]


@register_extension_dtype
class ImageDtype(TensorDtype):
    """Data type for columns of image pixel arrays.

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


class ImageSeries(TensorSeries):
    """A Series of image pixel arrays, one numpy array per row."""

    def to_image_stack(self, na_value=np.nan) -> np.ndarray:
        """Convert to a single (n, height, width, ...) numpy pixel block.

        Missing rows are filled with ``na_value``. Only available when all
        rows share one shape (always true for fixed-shape image columns).
        """
        return self.to_stack(na_value=na_value)

    def _repr_html_(self) -> str | None:
        """HTML repr with image thumbnails for the first rows (used by notebooks)."""
        if not isinstance(self.dtype, TensorDtype):
            return None
        return image_series_html(self)


register_series_class(ImageDtype, ImageSeries)
register_html_formatter(ImageDtype, image_cell_html)
