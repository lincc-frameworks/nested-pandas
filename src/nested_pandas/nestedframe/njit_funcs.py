import numpy as np
from numba import njit


@njit
def _map_rows_njit2_nest_nest(func, offsets1, offsets2, col1, col2):
    """
    Apply a JIT-compiled row-wise function to two nested columns.

    This helper function operates on two flattened nested columns of same length.
    Each row's nested values are reconstructed using the provided offset
    arrays and passed as 1D slices to func.

    Parameters
    ----------
    func : callable
        A Numba-compiled function that takes two 1D NumPy arrays
        and returns a scalar result.
    offsets1 : ndarray of shape (n_rows + 1,)
        Offset array defining row boundaries for the first nested column.
        The slice for row i is col1[offsets1[i]:offsets1[i + 1]].
    offsets2 : ndarray of shape (n_rows + 1,)
        Offset array defining row boundaries for the second nested column.
        The slice for row i is col2[offsets2[i]:offsets2[i + 1]].
    col1 : ndarray
        Flattened data for the first nested column.
    col2 : ndarray
        Flattened data for the second nested column.

    Returns
    -------
    out : ndarray
        Array containing the scalar result of func applied to
        each pair of nested row slices.

    Notes
    -----
    This function assumes that two offset arrays define the same number of rows.
    """
    out = np.empty(offsets1.size - 1)

    for i in range(out.size):
        start1, end1 = offsets1[i], offsets1[i + 1]
        start2, end2 = offsets2[i], offsets2[i + 1]
        out[i] = func(col1[start1:end1], col2[start2:end2])

    return out


@njit
def _map_rows_njit2_base_nest(func, base_col1, offsets, col2):
    """
    Apply a JIT-compiled row-wise function to a base column as the first argument
    and a nested column as the second argument.

    Parameters
    ----------
    func : callable
        A Numba-compiled function that takes a scalar and a 1D NumPy array
        and returns a scalar result.
    base_col1 : ndarray
        Base column data for the first argument of func.
    offsets : ndarray of shape (n_rows + 1,)
        Offset array defining row boundaries for the nested column.
        The slice for row i is col2[offsets[i]:offsets[i + 1]].
    col2 : ndarray
        Flattened data for the second nested column.

    Returns
    -------
    out : ndarray
        Array containing the scalar result of func applied to
        each pair of base column scalar and nested row slice.
    """
    out = np.empty(offsets.size - 1)

    for i in range(out.size):
        start, end = offsets[i], offsets[i + 1]
        out[i] = func(base_col1[i], col2[start:end])

    return out


@njit
def _map_rows_njit2_nest_base(func, offsets, col1, base_col2):
    """
    Apply a JIT-compiled row-wise function to a nested column as the first argument
    and a base column as the second argument.

    Parameters
    ----------
    func : callable
        A Numba-compiled function that takes a 1D NumPy array and a scalar
        and returns a scalar result.
    offsets : ndarray of shape (n_rows + 1,)
        Offset array defining row boundaries for the nested column.
        The slice for row i is col1[offsets[i]:offsets[i + 1]].
    col1 : ndarray
        Flattened data for the first nested column.
    base_col2 : ndarray
        Base column data for the second argument of func.

    Returns
    -------
    out : ndarray
        Array containing the scalar result of func applied to
        each pair of base column scalar and nested row slice.
    """
    out = np.empty(offsets.size - 1)

    for i in range(out.size):
        start, end = offsets[i], offsets[i + 1]
        out[i] = func(col1[start:end], base_col2[i])

    return out


@njit
def _map_rows_njit2_base_base(func, base_col1, base_col2):
    """
    Apply a JIT-compiled row-wise function to two base columns.

    This helper function operates on two base columns of same length.
    Each row's values are passed as scalars to func.

    Parameters
    ----------
    func : callable
        A Numba-compiled function that takes two scalars and returns a scalar result.
    base_col1 : ndarray
        Base column data for the first argument of func.
    base_col2 : ndarray
        Base column data for the second argument of func.

    Returns
    -------
    out : ndarray
        Array containing the scalar result of func applied to
        each pair of base column values.

    Notes
    -----
    This function assumes that two base columns have the same number of rows.
    """
    out = np.empty(base_col1.size)

    for i in range(out.size):
        out[i] = func(base_col1[i], base_col2[i])

    return out


@njit
def _map_rows_njit1_nested(func, offsets, col):
    """
    Apply a JIT-compiled row-wise function to one nested column.

    Parameters
    ----------
    func : callable
        A Numba-compiled function that takes a 1D NumPy array and returns a scalar result.
    offsets : ndarray of shape (n_rows + 1,)
        Offset array defining row boundaries for the nested column.
        The slice for row i is col[offsets[i]:offsets[i + 1]].
    col : ndarray
        Flattened data for the nested column.

    Returns
    -------
    out : ndarray
        Array containing the scalar result of func applied to each nested row slice.
    """
    out = np.empty(offsets.size - 1)

    for i in range(out.size):
        start, end = offsets[i], offsets[i + 1]
        out[i] = func(col[start:end])

    return out


@njit
def _map_rows_njit1_base(func, base_col):
    """
    Apply a JIT-compiled row-wise function to one base column.

    Parameters
    ----------
    func : callable
        A Numba-compiled function that takes a scalar and returns a scalar result.
    base_col : ndarray
        Base column data.

    Returns
    -------
    out : ndarray
        Array containing the scalar result of func applied to each base column value.
    """
    out = np.empty(base_col.size)

    for i in range(out.size):
        out[i] = func(base_col[i])

    return out
