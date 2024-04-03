from collections.abc import Generator

import pyarrow as pa


def is_pa_type_a_list(pa_type: type[pa.Array]) -> bool:
    """Check if the given pyarrow type is a list type.

    I.e. one of the following types: ListArray, LargeListArray,
    FixedSizeListArray.

    Returns
    -------
    bool
        True if the given type is a list type, False otherwise.
    """
    return (
        pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type) or pa.types.is_fixed_size_list(pa_type)
    )


def enumerate_chunks(array: pa.ChunkedArray) -> Generator[tuple[slice, pa.Array], None, None]:
    """Iterate over pyarrow.ChunkedArray chunks with their slice indices.

    Parameters
    ----------
    array : pa.ChunkedArray
        Input chunked array.

    Yields
    ------
    slice
        `slice(index_start, index_stop)` for the current chunk.
    pa.Array
        The current chunk.
    """
    index_start = 0
    for chunk in array.iterchunks():
        index_stop = index_start + len(chunk)
        yield slice(index_start, index_stop), chunk
        index_start = index_stop
