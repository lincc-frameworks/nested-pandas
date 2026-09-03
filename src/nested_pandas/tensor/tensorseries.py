"""Series subclass returned for tensor columns."""

from functools import wraps

import numpy as np
import pandas as pd

from nested_pandas.series.registry import register_html_formatter, register_series_class
from nested_pandas.tensor.display import tensor_cell_html, tensor_series_html
from nested_pandas.tensor.dtype import TensorDtype
from nested_pandas.tensor.ext_array import TensorArray

__all__ = ["TensorSeries"]


def tensor_only(func):
    """Decorator to designate certain functions can only be used with TensorDtype."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].dtype, TensorDtype):
            raise TypeError(f"'{func.__name__}' can only be used with a TensorDtype, not '{args[0].dtype}'.")
        return func(*args, **kwargs)

    return wrapper


class TensorSeries(pd.Series):
    """A Series of tensor values, one numpy ndarray per row.

    This class is a stateless view over a :class:`TensorArray`-backed series;
    ``NestedFrame`` column access returns tensor columns wrapped in it. It
    works for both fixed-shape and variable-shape tensor dtypes.
    """

    @property
    @tensor_only
    def tensor(self) -> TensorArray:
        """The backing TensorArray."""
        return self.array

    @property
    @tensor_only
    def tensor_shape(self) -> tuple[int, ...] | None:
        """The common tensor shape, or None for variable-shape columns."""
        return self.dtype.shape

    @property
    @tensor_only
    def tensor_ndim(self) -> int:
        """The common number of tensor dimensions."""
        return self.dtype.ndim

    @property
    @tensor_only
    def value_dtype(self) -> np.dtype:
        """The tensor element type as a numpy dtype."""
        return self.dtype.np_value_dtype

    @property
    @tensor_only
    def shapes(self) -> np.ndarray:
        """Per-row tensor shapes as an (n, ndim) integer array."""
        return self.array.shapes

    @tensor_only
    def to_stack(self, na_value=np.nan) -> np.ndarray:
        """Convert to a single (n, ...) numpy block; see TensorArray.to_stack."""
        return self.array.to_stack(na_value=na_value)

    def _repr_html_(self) -> str | None:
        """HTML repr with viridis thumbnails and colorbars for the first 2-d rows (used by notebooks)."""
        if not isinstance(self.dtype, TensorDtype):
            return None
        return tensor_series_html(self)


register_series_class(TensorDtype, TensorSeries)
register_html_formatter(TensorDtype, tensor_cell_html)
