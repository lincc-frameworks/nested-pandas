import numpy as np
from numba import njit

@njit
def _map_rows_njit2_nest_nest(func, offsets1, offsets2, col1, col2):
    """
    func: numba-jit function taking 2 scalar args from nested column (must be same length)
    offsets1, offsets2: 1D numpy arrays of offsets for the two nested columns
    col1, col2: 1D numpy arrays corresponding to flattened nested columns
    """
    out = np.empty(offsets1.size - 1)

    for i in range(out.size):
        start1, end1 = offsets1[i], offsets1[i + 1]
        start2, end2 = offsets2[i], offsets2[i + 1]
        out[i] = func(col1[start1:end1], col2[start2:end2])

    return out

@njit
def _map_rows_njit2_base_nest(func, offsets, base_col1, col2):
    """
    func: numba-jit function taking 1 scalar arg from base column
    offsets: 1D numpy array of offsets for the nested column
    base_col1: 1D numpy array of the base column for the first argument
    col2: 1D numpy array corresponding to flattened nested column for the second argument
    """
    out = np.empty(offsets.size - 1)

    for i in range(out.size):
        start, end = offsets[i], offsets[i + 1]
        out[i] = func(base_col1[i], col2[start:end])

    return out

@njit
def _map_rows_njit2_nest_base(func, offsets, col1, base_col2):
    """
    func: numba-jit function taking 1 scalar arg from base column
    offsets: 1D numpy array of offsets for the nested column
    col1: 1D numpy array corresponding to flattened nested column for the first argument
    base_col2: 1D numpy array of the base column for the second argument
    """
    out = np.empty(offsets.size - 1)

    for i in range(out.size):
        start, end = offsets[i], offsets[i + 1]
        out[i] = func(col1[start:end], base_col2[i])

    return out

@njit
def _map_rows_njit2_base_base(func, base_col1, base_col2):
    """
    func: numba-jit function taking 1 scalar arg from base column
    base_col1: 1D numpy array of the base column for the first argument
    base_col2: 1D numpy array of the base column for the second argument
    """
    out = np.empty(base_col1.size)

    for i in range(out.size):
        out[i] = func(base_col1[i], base_col2[i])

    return out

@njit
def _map_rows_njit1_nested(func, offsets, col):
    """
    func: numba-jit function taking 1 scalar arg from nested column
    offsets: 1D numpy array of offsets for the nested column
    col: 1D numpy array corresponding to flattened nested column
    """
    out = np.empty(offsets.size - 1)

    for i in range(out.size):
        start, end = offsets[i], offsets[i + 1]
        out[i] = func(col[start:end])

    return out

@njit
def _map_rows_njit1_base(func, base_col):
    """
    func: numba-jit function taking 1 scalar arg from base column
    base_col: 1D numpy array of the base column
    """

    out = np.empty(base_col.size)

    for i in range(out.size):
        out[i] = func(base_col[i])
    
    return out