from __future__ import annotations  # Python 3.9 requires it for X | Y type hints

from typing import TYPE_CHECKING

import pyarrow as pa

from nested_pandas.series.utils import (
    transpose_list_struct_chunked,
    validate_struct_list_array_for_equal_lengths,
)

if TYPE_CHECKING:
    from nested_pandas.series.storage.list_struct_storage import ListStructStorage
    from nested_pandas.series.storage.table_storage import TableStorage


class StructListStorage:
    """Store nested data as a PyArrow struct-list array.

    Parameters
    ----------
    array : pa.StructArray or pa.ChunkedArray
        Pyarrow struct-array with all fields to be list-arrays.
        All list-values must be "aligned", e.g., have the same length.
    validate : bool (default True)
        Check that all the lists have the same lengths for each struct-value.
    """

    data: pa.ChunkedArray

    def __init__(self, array: pa.StructArray | pa.ChunkedArray, *, validate: bool = True) -> None:
        if isinstance(array, pa.StructArray):
            array = pa.chunked_array([array])
        if not isinstance(array, pa.ChunkedArray):
            raise ValueError("array must be a StructArray or ChunkedArray")

        if validate:
            for chunk in array.chunks:
                validate_struct_list_array_for_equal_lengths(chunk)

        self.data = array

    @classmethod
    def from_list_struct_storage(cls, list_struct_storage: ListStructStorage) -> Self:  # type: ignore # noqa: F821
        """Construct from a ListStructStorage object.

        Parameters
        ----------
        list_struct_storage : ListStructStorage
            ListStructStorage object.
        """
        data = transpose_list_struct_chunked(list_struct_storage.data)
        return cls(data, validate=False)

    @classmethod
    def from_table_storage(cls, table_storage: TableStorage) -> Self:  # type: ignore # noqa: F821
        """Construct from a TableStorage object.

        Parameters
        ----------
        table_storage : TableStorage
            TableStorage object.
        """
        data = table_storage.data.to_struct_array()
        return cls(data, validate=False)
