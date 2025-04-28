from __future__ import annotations  # Python 3.9 requires it for X | Y type hints

from typing import TYPE_CHECKING

import pyarrow as pa

from nested_pandas.series.utils import transpose_struct_list_chunked

if TYPE_CHECKING:
    from nested_pandas.series.storage.struct_list_storage import StructListStorage
    from nested_pandas.series.storage.table_storage import TableStorage


class ListStructStorage:
    """Store nested data as a PyArrow list-struct array.

    Parameters
    ----------
    array : pa.ListArray or pa.ChunkedArray
        Pyarrow list-array with a struct value type. An array or a chunk-array
    """

    data: pa.ChunkedArray

    def __init__(self, array: pa.ListArray | pa.ChunkedArray) -> None:
        if isinstance(array, pa.ListArray):
            array = pa.chunked_array([array])
        if not isinstance(array, pa.ChunkedArray):
            raise ValueError("array must be of type pa.ChunkedArray")
        self.data = array

    @classmethod
    def from_struct_list_storage(cls, struct_list_storage: StructListStorage) -> Self:  # type: ignore # noqa: F821
        """Construct from a StructListStorage object.

        Parameters
        ----------
        struct_list_storage : StructListStorage
            StructListStorage object.
        """
        data = transpose_struct_list_chunked(struct_list_storage.data, validate=False)
        return cls(data)

    @classmethod
    def from_table_storage(cls, table_storage: TableStorage) -> Self:  # type: ignore # noqa: F821
        """Construct from a TableStorage object.

        Parameters
        ----------
        table_storage : TableStorage
            TableStorage object.
        """
        from nested_pandas.series.storage import StructListStorage

        struct_list_storage = StructListStorage.from_table_storage(table_storage)
        return cls.from_struct_list_storage(struct_list_storage)
