"""Registry mapping extension dtypes to pandas Series subclasses.

This allows ``NestedFrame`` column access (``df["col"]``) to return a custom
``pd.Series`` subclass for columns of a registered extension dtype, the same
way nested columns are returned as :class:`NestedSeries`. Downstream packages
can register their own dtype/series pairs::

    from nested_pandas import register_series_class

    register_series_class(MyDtype, MySeries)

Registered series classes should be stateless views over the underlying
extension array: they must be constructible from a plain ``pd.Series``
(``MySeries(series)``) without losing information, since pandas operations
may return plain ``pd.Series`` objects that get re-wrapped on access.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.extensions import ExtensionDtype

__all__ = ["register_series_class", "unregister_series_class", "get_series_class", "wrap_series"]

_SERIES_CLASSES: dict[type[ExtensionDtype], type[pd.Series]] = {}


def register_series_class(dtype_class: type[ExtensionDtype], series_class: type[pd.Series]) -> None:
    """Register a Series subclass to be returned for columns of the given dtype.

    Parameters
    ----------
    dtype_class : type[ExtensionDtype]
        The extension dtype class to associate with the series class.
    series_class : type[pd.Series]
        The ``pd.Series`` subclass to wrap columns of this dtype in. It must
        be constructible from a plain ``pd.Series`` without copying or losing
        information, i.e. it should carry no state of its own.

    Examples
    --------
    >>> import pandas as pd
    >>> from nested_pandas import NestedDtype, NestedSeries
    >>> from nested_pandas.series.registry import get_series_class
    >>> get_series_class(NestedDtype.from_fields({"a": pd.ArrowDtype("int64")}))
    <class 'nested_pandas.series.nestedseries.NestedSeries'>
    """
    _SERIES_CLASSES[dtype_class] = series_class


def unregister_series_class(dtype_class: type[ExtensionDtype]) -> None:
    """Remove a previously registered dtype/series association.

    Parameters
    ----------
    dtype_class : type[ExtensionDtype]
        The extension dtype class to unregister. No-op if not registered.
    """
    _SERIES_CLASSES.pop(dtype_class, None)


def get_series_class(dtype) -> type[pd.Series] | None:
    """Return the registered series class for a dtype instance, if any.

    Walks the dtype's class MRO, so subclasses of a registered dtype resolve
    to the same series class unless they register their own.

    Parameters
    ----------
    dtype
        A dtype instance (e.g. ``series.dtype``).

    Returns
    -------
    type[pd.Series] or None
        The registered series class, or None if no class is registered.
    """
    for klass in type(dtype).__mro__:
        if klass in _SERIES_CLASSES:
            return _SERIES_CLASSES[klass]
    return None


def wrap_series(series: pd.Series) -> pd.Series:
    """Wrap a series in the series class registered for its dtype, if any.

    Parameters
    ----------
    series : pd.Series
        The series to wrap.

    Returns
    -------
    pd.Series
        The wrapped series, or the input unchanged if no class is registered
        for its dtype or it is already of the registered class.
    """
    series_class = get_series_class(series.dtype)
    if series_class is None or type(series) is series_class:
        return series
    return series_class(series)
