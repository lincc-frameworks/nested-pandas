from __future__ import annotations  # Python 3.9 requires it for X | Y type hints

from typing import TYPE_CHECKING, Any

import pyarrow as pa

from nested_pandas.series.utils import transpose_struct_list_chunked, validate_list_struct_type

if TYPE_CHECKING:
    from nested_pandas.series._storage.struct_list_storage import StructListStorage
    from nested_pandas.series._storage.table_storage import TableStorage


class ListStructStorage:
    """Store nested data as a PyArrow list-struct array.

    Parameters
    ----------
    array : pa.ListArray or pa.ChunkedArray
        Pyarrow list-array with a struct value type. An array or a chunk-array
    """

    _data: pa.ChunkedArray

    def __init__(self, array: pa.ListArray | pa.ChunkedArray) -> None:
        if isinstance(array, pa.ListArray):
            array = pa.chunked_array([array])
        if not isinstance(array, pa.ChunkedArray):
            raise ValueError("array must be of type pa.ChunkedArray")
        validate_list_struct_type(array.type)
        self._data = array

    @property
    def data(self) -> pa.ChunkedArray:
        return self._data

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
        from nested_pandas.series._storage import StructListStorage

        struct_list_storage = StructListStorage.from_table_storage(table_storage)
        return cls.from_struct_list_storage(struct_list_storage)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self._data == other._data

    @property
    def nbytes(self) -> int:
        """Number of bytes consumed by the data in memory."""
        return self._data.nbytes

    @property
    def type(self) -> pa.ListType:
        """Pyarrow type of the underlying array."""
        return self._data.type

    @property
    def num_chunks(self) -> int:
        """Number of chunk_lens in the underlying array."""
        return self._data.num_chunks
