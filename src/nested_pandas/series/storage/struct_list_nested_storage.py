from __future__ import annotations  # Python 3.9 requires it for X | Y type hints

import pyarrow as pa

from nested_pandas.series.utils import validate_struct_list_array_for_equal_lengths


class StructListNestedStorage:
    """Store nested data as PyArrow struct-list array."""

    def __init__(self, array: pa.StructArray | pa.ChunkedArray, *, validate: bool = True) -> None:
        if isinstance(array, pa.StructArray):
            array = pa.chunked_array([array])
        if not isinstance(array, pa.ChunkedArray):
            raise ValueError("array must be a StructArray or ChunkedArray")

        if validate:
            for chunk in array.chunks:
                validate_struct_list_array_for_equal_lengths(chunk)

        self.data = array

    def from_list_struct_array(self, list_array: pa.ListArray | pa.ChunkedArray) -> Self:  # type: ignore # noqa: UP034
        """Construct from"""
