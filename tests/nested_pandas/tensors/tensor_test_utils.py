"""Helpers for building tensor test data, shared by conftest.py and the test modules."""

import numpy as np

TENSOR_SHAPE = (2, 3)
"""Shape of the tensors in the shared fixtures"""


def tensor(value: float) -> np.ndarray:
    """A distinct ``TENSOR_SHAPE`` tensor for a scalar, used to build fixture data."""
    return value + np.arange(np.prod(TENSOR_SHAPE), dtype=np.float64).reshape(TENSOR_SHAPE) / 10


def tensor_stack(values) -> np.ndarray:
    """A ``(len(values), *TENSOR_SHAPE)`` stack of :func:`tensor` for each value."""
    return np.stack([tensor(value) for value in values])
