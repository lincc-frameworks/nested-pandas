"""Pandas Series subclass for fixed-shape tensor columns."""

from __future__ import annotations  # Self is not available in python 3.10

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from nested_pandas.tensors.dtype import TensorDtype
from nested_pandas.tensors.ext_array import TensorExtensionArray

__all__ = ["TensorSeries"]


def tensor_only(func):
    """Decorator to designate certain functions can only be used with TensorDtype."""

    @wraps(func)  # This ensures the original function's metadata is preserved
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].dtype, TensorDtype):
            raise TypeError(f"'{func.__name__}' can only be used with a TensorDtype, not '{args[0].dtype}'.")

        result = func(*args, **kwargs)
        return result

    return wrapper


def _tensor_series_or_series(data=None, *args, **kwargs) -> pd.Series:
    """Build a Series, as TensorSeries if it is of TensorDtype, see ``TensorSeries._constructor``"""
    series = pd.Series(data, *args, **kwargs)
    if isinstance(series.dtype, TensorDtype):
        return TensorSeries(series)
    return series


class TensorSeries(pd.Series):
    """A Series of fixed-shape tensors, one numpy array per row.

    This class extends a standard pandas Series with conveniences for columns
    of :class:`TensorDtype`, whose values are stored as a
    :class:`TensorExtensionArray`. Scalar access returns the tensor as a
    read-only numpy view, and :meth:`to_stack` returns the whole column as
    a single ``(n, *shape)`` numpy array.

    Examples
    --------
    >>> import numpy as np
    >>> from nested_pandas import TensorSeries
    >>> series = TensorSeries.from_stack(np.arange(12.0).reshape(3, 2, 2), name="t")
    >>> series.dtype
    tensor[double, (2, 2)]
    >>> series[1]
    array([[4., 5.],
           [6., 7.]])
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        """Constructor pandas uses for the results of Series operations.

        Results that are still of TensorDtype, such as slices, ``take``,
        ``copy`` or boolean filtering, stay a TensorSeries. Results of any
        other dtype, such as ``isna()``, become a plain pandas Series, so a
        TensorSeries is never held over data its methods cannot handle.
        """
        return _tensor_series_or_series

    @classmethod
    def from_stack(
        cls,
        stack: np.ndarray,
        *,
        dtype: TensorDtype | pa.FixedShapeTensorType | pd.ArrowDtype | str | None = None,
        copy: bool = False,
        index=None,
        name=None,
    ) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Construct a TensorSeries from a numpy array of shape ``(n, *shape)``.

        The first axis of ``stack`` enumerates the rows. See
        :meth:`TensorExtensionArray.from_stack` for details, including the
        memory aliasing of a C-contiguous ``stack`` when ``copy`` is False.

        Parameters
        ----------
        stack : np.ndarray
            Array whose first axis enumerates the tensors.
        dtype : TensorDtype, pa.FixedShapeTensorType, pd.ArrowDtype or str, optional
            dtype of the resulting series. Inferred from the numpy dtype and
            ``stack.shape[1:]`` if not given.
        copy : bool, default False
            Whether to copy ``stack``. Layouts other than C-contiguous are
            always copied into C order.
        index : array-like or Index, optional
            Index of the resulting series, a RangeIndex if not given.
        name : Hashable, optional
            Name of the resulting series.

        Returns
        -------
        TensorSeries
            The constructed series.

        Examples
        --------
        >>> import numpy as np
        >>> from nested_pandas import TensorSeries
        >>> TensorSeries.from_stack(np.zeros((2, 3)), index=["a", "b"], name="t")
        a    [0.0, 0.0, 0.0]
        b    [0.0, 0.0, 0.0]
        Name: t, dtype: tensor[double, (3,)]
        """
        array = TensorExtensionArray.from_stack(stack, dtype=dtype, copy=copy)
        return cls(array, index=index, name=name)

    def _repr_html_(self) -> str | None:
        """HTML repr of a tensor series: its cells rendered as in a one-column NestedFrame.

        Plain pandas Series have no HTML repr, so for any other dtype this
        returns None and notebooks show the text repr.
        """
        if not isinstance(self.dtype, TensorDtype):
            return None
        # Imported here because nestedframe.core imports this module
        from nested_pandas.nestedframe.core import NestedFrame

        return NestedFrame(self.to_frame())._repr_html_()

    @property
    @tensor_only
    def tensor_shape(self) -> tuple[int, ...]:
        """Shape of every tensor in the series.

        This is ``dtype.shape``. Note that ``Series.shape`` is the shape of
        the series itself, ``(len(self),)``, as for any pandas Series.
        """
        return self.dtype.shape

    @tensor_only
    def to_stack(self, na_value: Any = np.nan) -> np.ndarray:
        """Convert the series to a single numpy array of shape ``(n, *shape)``.

        Without missing values the result is a read-only zero-copy view over
        the arrow buffers. With missing tensors the data is copied, the result
        dtype is widened as needed to hold ``na_value``, and the missing
        tensors are filled with it. See :meth:`TensorExtensionArray.to_stack`.

        Parameters
        ----------
        na_value : Any, default np.nan
            Value to fill missing tensors with.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(self), *self.tensor_shape)``.

        Examples
        --------
        >>> import numpy as np
        >>> from nested_pandas import TensorSeries
        >>> series = TensorSeries.from_stack(np.arange(6.0).reshape(3, 2))
        >>> series.to_stack()
        array([[0., 1.],
               [2., 3.],
               [4., 5.]])
        """
        return self.array.to_stack(na_value=na_value)

    @tensor_only
    def to_lists(self) -> pd.Series:
        """Convert the series to a pandas Series of pyarrow fixed-size lists.

        The result has a ``pd.ArrowDtype`` of the ``fixed_size_list`` storage
        type of the tensors, flattening each tensor to one dimension, and
        shares the index and name of this series. No data is copied.

        Returns
        -------
        pd.Series
            Series of ``pd.ArrowDtype(pa.list_(value_type, size))``.

        Examples
        --------
        >>> import numpy as np
        >>> from nested_pandas import TensorSeries
        >>> series = TensorSeries.from_stack(np.arange(8.0).reshape(2, 2, 2), name="t")
        >>> series.to_lists()
        0    [0. 1. 2. 3.]
        1    [4. 5. 6. 7.]
        Name: t, dtype: fixed_size_list<item: double>[4][pyarrow]
        """
        return pd.Series(self.array.to_arrow_ext_array(storage=True), index=self.index, name=self.name)
