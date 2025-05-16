from __future__ import annotations  # Python 3.9 requires it for X | Y type hints

from typing import TYPE_CHECKING

import pyarrow as pa

from nested_pandas.series.utils import (
    table_from_struct_array,
    table_to_struct_array,
    validate_struct_list_array_for_equal_lengths,
)

if TYPE_CHECKING:
    from nested_pandas.series._storage.list_struct_storage import ListStructStorage
    from nested_pandas.series._storage.struct_list_storage import StructListStorage


class TableStorage:
    """Store nested data as a PyArrow table with list-columns.

    Parameters
    ----------
    table : pa.Table
        PyArrow table, all columns must be list-columns.
        All list-values must be "aligned", e.g., have the same length.
    """

    _data: pa.Table

    def __init__(self, table: pa.Table, validate: bool = True) -> None:
        if validate:
            struct_array = table_to_struct_array(table)
            for chunk in struct_array.iterchunks():
                validate_struct_list_array_for_equal_lengths(chunk)

        self._data = table

    @property
    def data(self) -> pa.Table:
        return self._data

    @classmethod
    def from_list_struct_storage(cls, list_storage: ListStructStorage) -> Self:  # type: ignore # noqa: F821
        """Construct from a StructListStorage object.

        Parameters
        ----------
        list_storage : ListStructStorage
            StructListStorage object.
        """
        from nested_pandas.series._storage import StructListStorage

        struct_list_storage = StructListStorage.from_list_struct_storage(list_storage)
        return cls.from_struct_list_storage(struct_list_storage)

    @classmethod
    def from_struct_list_storage(cls, struct_list_storage: StructListStorage) -> Self:  # type: ignore # noqa: F821
        """Construct from a StructListStorage object.

        Parameters
        ----------
        struct_list_storage : StructListStorage
            StructListStorage object.
        """
        table = table_from_struct_array(struct_list_storage.data)
        return cls(table, validate=False)
