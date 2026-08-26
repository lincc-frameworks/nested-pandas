"""Store-backed image columns: cutout descriptors resolved against an image store.

A cutout column stores one ~20-byte descriptor per row — an image id plus a
pixel bounding box — and resolves pixels lazily through a single
:class:`ImageStore` shared by the whole column. Scalar access returns numpy
*views* into the store's cached planes, so overlapping cutouts share memory.

The store is array-level state (not per row, not frame metadata): it rides
along through ``take``/``filter``/``concat``/slicing via the extension array
hooks, exactly like the earlier cutout prototype.

Serialization converts to the tensor representation: writing a cutout column
materializes its pixels into the ``nested_pandas.image`` extension type, so
the file holds an ordinary image column and reads back as a tensor-backed
:class:`~nested_pandas.tensor.imageseries.ImageSeries`. The store reference
itself is never serialized.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.extensions import ExtensionArray, register_extension_dtype
from pandas.api.indexers import check_array_indexer
from pandas.core.dtypes.base import ExtensionDtype

from nested_pandas.series.registry import register_html_formatter, register_series_class
from nested_pandas.tensor.display import image_cell_html
from nested_pandas.tensor.dtype import TensorDtype
from nested_pandas.tensor.ext_array import TensorArray
from nested_pandas.tensor.imageseries import BaseImageSeries, ImageDtype

__all__ = [
    "ImageStore",
    "DictImageStore",
    "ChainImageStore",
    "merge_stores",
    "CutoutDtype",
    "CutoutArray",
    "CutoutSeries",
    "CUTOUT_DESCRIPTOR_TYPE",
]

CUTOUT_DESCRIPTOR_TYPE = pa.struct(
    [
        pa.field("image_id", pa.string()),
        pa.field("x0", pa.int32()),
        pa.field("y0", pa.int32()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
    ]
)


class ImageStore(ABC):
    """Minimal interface a cutout column needs to resolve pixels.

    Implementations map image ids to 2-d pixel planes; richer stores (lazy
    readers, region reads, caching) implement the same interface downstream.
    """

    @abstractmethod
    def get_image(self, image_id: str) -> np.ndarray:
        """Return the full pixel plane for an image id."""

    @abstractmethod
    def __contains__(self, image_id: str) -> bool:
        """Whether this store can resolve the image id."""

    def get_region(self, image_id: str, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Return one rectangular region of an image.

        The base implementation slices the full plane; stores with
        chunked/tiled backing may override it to read only what is needed.
        """
        return self.get_image(image_id)[y0:y1, x0:x1]

    def plan_reads(self, image_ids) -> None:
        """Announce the images an upcoming batch of reads will touch.

        Called with one image id per cutout (repetitions carry multiplicity)
        before a batch render such as ``to_stack``. Stores may use the counts
        to pick a read strategy upfront; the base implementation does nothing.
        """

    def with_read_mode(self, **kwargs) -> ImageStore:
        """Return a store with adjusted read behavior; base implementation is a no-op."""
        del kwargs
        return self


class DictImageStore(ImageStore):
    """An in-memory image store over a mapping of id -> 2-d numpy array."""

    def __init__(self, images: Mapping[str, np.ndarray]):
        self._images = dict(images)

    def get_image(self, image_id: str) -> np.ndarray:
        return self._images[image_id]

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._images


class ChainImageStore(ImageStore):
    """An image store resolving ids by querying a sequence of stores in order.

    Used when concatenating cutout columns whose stores differ: the result
    references all of them without copying pixel data.
    """

    def __init__(self, stores: Sequence[ImageStore]):
        self.stores = list(stores)

    def get_image(self, image_id: str) -> np.ndarray:
        for store in self.stores:
            if image_id in store:
                return store.get_image(image_id)
        raise KeyError(image_id)

    def get_region(self, image_id: str, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        for store in self.stores:
            if image_id in store:
                return store.get_region(image_id, y0, y1, x0, x1)
        raise KeyError(image_id)

    def plan_reads(self, image_ids) -> None:
        for store in self.stores:
            store.plan_reads([image_id for image_id in image_ids if image_id in store])

    def __contains__(self, image_id: str) -> bool:
        return any(image_id in store for store in self.stores)


def merge_stores(stores: Sequence[ImageStore | None]) -> ImageStore | None:
    """Merge the stores of multiple cutout columns into a single store.

    Identical store objects are deduplicated; distinct stores are combined
    into a :class:`ChainImageStore` (flattening nested chains).
    """
    unique: list[ImageStore] = []
    for store in stores:
        if store is None:
            continue
        candidates = store.stores if isinstance(store, ChainImageStore) else [store]
        for candidate in candidates:
            if not any(candidate is seen for seen in unique):
                unique.append(candidate)
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    return ChainImageStore(unique)


@register_extension_dtype
class CutoutDtype(ExtensionDtype):
    """Data type for store-backed image cutout columns.

    Rows are cutout descriptors (image id and pixel bounding box); pixels are
    resolved lazily through the column's attached :class:`ImageStore`.
    """

    _metadata = ()
    na_value = pd.NA
    type = np.ndarray

    @property
    def name(self) -> str:
        return "cutout"

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def construct_array_type(cls):
        return CutoutArray

    @classmethod
    def construct_from_string(cls, string: str) -> CutoutDtype:
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        if string != "cutout":
            raise TypeError(f"Cannot construct a 'CutoutDtype' from '{string}'")
        return cls()


class CutoutArray(ExtensionArray):
    """Pandas extension array of image cutouts, stored as descriptors.

    Parameters
    ----------
    values : pa.Array or pa.ChunkedArray
        Struct-typed descriptors ``(image_id, x0, y0, width, height)``.
    store : ImageStore or None
        The store used to resolve pixels; without one, only descriptor-level
        operations (shapes, repr, isna, take, ...) are available.
    """

    def __init__(self, values: pa.Array | pa.ChunkedArray, *, store: ImageStore | None = None):
        if isinstance(values, pa.Array):
            values = pa.chunked_array([values])
        if not isinstance(values, pa.ChunkedArray):
            raise TypeError(f"values must be a pyarrow Array or ChunkedArray, got {type(values)}")
        if not values.type.equals(CUTOUT_DESCRIPTOR_TYPE):
            values = values.cast(CUTOUT_DESCRIPTOR_TYPE)
        self._storage = values
        self._store = store
        self._dtype = CutoutDtype()

    @classmethod
    def from_descriptors(
        cls, image_id, x0, y0, width, height, *, store: ImageStore | None = None
    ) -> CutoutArray:
        """Build a CutoutArray from per-row descriptor sequences."""
        storage = pa.StructArray.from_arrays(
            [
                pa.array([str(value) for value in image_id], type=pa.string()),
                pa.array(np.asarray(x0, dtype=np.int32)),
                pa.array(np.asarray(y0, dtype=np.int32)),
                pa.array(np.asarray(width, dtype=np.int32)),
                pa.array(np.asarray(height, dtype=np.int32)),
            ],
            names=["image_id", "x0", "y0", "width", "height"],
        )
        return cls(storage, store=store)

    def with_store(self, store: ImageStore | None) -> CutoutArray:
        """Return the same descriptors with a different store attached."""
        return type(self)(self._storage, store=store)

    @property
    def store(self) -> ImageStore | None:
        """The attached image store, or None."""
        return self._store

    def _require_store(self) -> ImageStore:
        if self._store is None:
            raise ValueError(
                "this cutout column has no image store attached; use .with_store(store) first"
            )
        return self._store

    # ExtensionArray overrides #

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy: bool = False) -> CutoutArray:
        del dtype, copy
        storage = pa.array(list(scalars), type=CUTOUT_DESCRIPTOR_TYPE)
        return cls(storage)

    @classmethod
    def _from_factorized(cls, values, original):
        raise NotImplementedError("CutoutArray does not support factorization")

    def __getitem__(self, item):
        if isinstance(item, int | np.integer):
            index = int(item)
            if index < 0:
                index += len(self)
            if not 0 <= index < len(self):
                raise IndexError(f"index {item} is out of bounds for array of length {len(self)}")
            return self._pixels_at(index)

        item = check_array_indexer(self, item)
        if isinstance(item, slice):
            if item.step is None or item.step == 1:
                start, stop, _ = item.indices(len(self))
                return type(self)(self._storage.slice(start, max(stop - start, 0)), store=self._store)
            item = np.arange(len(self))[item]

        item = np.asarray(item)
        if item.dtype == bool:
            return type(self)(self._storage.filter(pa.array(item)), store=self._store)
        return self.take(item)

    def __setitem__(self, key, value) -> None:
        raise NotImplementedError("CutoutArray is immutable; build a new column instead")

    def __len__(self) -> int:
        return len(self._storage)

    def __iter__(self):
        for index in range(len(self)):
            yield self._pixels_at(index)

    @property
    def dtype(self) -> CutoutDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._storage.nbytes

    def isna(self) -> np.ndarray:
        if self._storage.null_count == 0:
            return np.zeros(len(self), dtype=bool)
        return self._storage.is_null().to_numpy(zero_copy_only=False)

    @property
    def _hasna(self) -> bool:
        return self._storage.null_count > 0

    def take(self, indices, *, allow_fill: bool = False, fill_value: Any = None) -> CutoutArray:
        indices_array = np.asanyarray(indices)
        if len(self) == 0 and (indices_array >= 0).any():
            raise IndexError("cannot do a non-empty take from the empty array")
        if indices_array.size > 0 and indices_array.max() >= len(self):
            raise IndexError("out of bounds value in 'indices'.")

        if allow_fill:
            if not (fill_value is None or fill_value is pd.NA):
                raise NotImplementedError("CutoutArray.take only supports NA fill values")
            fill_mask = indices_array < 0
            if (indices_array < -1).any():
                raise ValueError("Invalid value in 'indices'. Must be all >= -1 for 'allow_fill'")
            if fill_mask.any():
                indices_array = pa.array(indices_array, mask=fill_mask)
            return type(self)(self._storage.take(indices_array), store=self._store)

        if (indices_array < 0).any():
            indices_array = np.copy(indices_array)
            indices_array[indices_array < 0] += len(self)
        return type(self)(self._storage.take(indices_array), store=self._store)

    def copy(self) -> CutoutArray:
        return type(self)(self._storage, store=self._store)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[CutoutArray]) -> CutoutArray:
        store = merge_stores([array._store for array in to_concat])
        chunks = [chunk for array in to_concat for chunk in array._storage.chunks]
        return cls(pa.chunked_array(chunks, type=CUTOUT_DESCRIPTOR_TYPE), store=store)

    def dropna(self) -> CutoutArray:
        return type(self)(self._storage.drop_null(), store=self._store)

    def to_numpy(self, dtype=None, copy: bool = False, na_value=None) -> np.ndarray:
        del dtype, copy
        result = np.empty(len(self), dtype=object)
        for index in range(len(self)):
            value = self._pixels_at(index)
            result[index] = na_value if value is pd.NA else value
        return result

    def __array__(self, dtype=None, copy=None):
        del copy
        return self.to_numpy(dtype=dtype)

    def equals(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self._storage == other._storage and self._store is other._store

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        del boxed

        def formatter(value):
            if value is pd.NA or value is None:
                return str(pd.NA)
            shape = "×".join(str(size) for size in value.shape)
            return f"[{shape}] {value.dtype}"

        return formatter

    # End of ExtensionArray overrides #

    def __arrow_array__(self, type=None):  # noqa: A002
        """Serialize by materializing into the tensor (image) representation.

        The written column is an ordinary ``nested_pandas.image`` extension
        column; the store reference is not serialized, so it reads back as a
        tensor-backed image column.
        """
        return self.to_tensor().__arrow_array__(type=type)

    # Image column API (shared with TensorArray) #

    def _plan_reads(self) -> None:
        """Announce the whole column's image ids to the store before a batch read."""
        if self._store is None:
            return
        combined = (
            self._storage.combine_chunks() if self._storage.num_chunks != 1 else self._storage.chunk(0)
        )
        ids = combined.field("image_id").drop_null().to_pylist()
        self._store.plan_reads(ids)

    def _descriptor_at(self, index: int) -> dict | Any:
        """The descriptor dict at a flat position, or NA — no pixel access."""
        scalar = self._storage[index]
        if not scalar.is_valid:
            return pd.NA
        return scalar.as_py()

    def _pixels_at(self, index: int) -> np.ndarray | Any:
        """The cutout pixels at a flat position, as a view into the store's plane."""
        descriptor = self._descriptor_at(index)
        if descriptor is pd.NA:
            return pd.NA
        store = self._require_store()
        return store.get_region(
            descriptor["image_id"],
            descriptor["y0"],
            descriptor["y0"] + descriptor["height"],
            descriptor["x0"],
            descriptor["x0"] + descriptor["width"],
        )

    @property
    def shapes(self) -> np.ndarray:
        """Per-row cutout shapes (height, width) from the descriptors — no pixel access."""
        combined = self._storage.combine_chunks() if self._storage.num_chunks != 1 else self._storage.chunk(0)
        heights = combined.field("height").to_numpy(zero_copy_only=False)
        widths = combined.field("width").to_numpy(zero_copy_only=False)
        result = np.stack([heights, widths], axis=1)
        result[self.isna()] = 0
        return result.astype(np.int32)

    def to_stack(self, na_value=np.nan) -> np.ndarray:
        """Gather all cutouts into one (n, height, width) block (always a copy).

        Missing rows are filled with ``na_value``; mixed cutout sizes raise,
        matching tensor-column semantics.
        """
        valid_shapes = {tuple(shape) for shape in self.shapes[~self.isna()]}
        if len(valid_shapes) > 1:
            raise ValueError(f"Cannot stack cutouts with mixed shapes {sorted(valid_shapes)}")
        if not valid_shapes:
            raise ValueError("Cannot stack an all-missing cutout column")
        shape = valid_shapes.pop()

        self._plan_reads()
        first = next(pixels for pixels in self if pixels is not pd.NA)
        result_dtype = np.result_type(first.dtype, np.asarray(na_value).dtype) if self._hasna else first.dtype
        result = np.full((len(self), *shape), na_value, dtype=result_dtype)
        for index in range(len(self)):
            pixels = self._pixels_at(index)
            if pixels is not pd.NA:
                result[index] = pixels
        return result

    def to_tensor(self, dtype: TensorDtype | None = None) -> TensorArray:
        """Materialize into a tensor-backed image array.

        Parameters
        ----------
        dtype : TensorDtype or None
            Target dtype; by default an :class:`ImageDtype` is inferred —
            fixed-shape when all cutouts share one size, ragged otherwise.
        """
        self._plan_reads()
        pixels = [self._pixels_at(index) for index in range(len(self))]
        if dtype is None:
            valid = [p for p in pixels if p is not pd.NA]
            if not valid:
                # Empty/all-missing columns (e.g. dask meta) have no shape to
                # infer; default to a ragged float32 image column.
                return TensorArray._from_sequence(pixels, dtype=ImageDtype(pa.float32(), ndim=2))
            value_type = pa.from_numpy_dtype(np.result_type(*(p.dtype for p in valid)))
            sizes = {p.shape for p in valid}
            if len(sizes) == 1:
                dtype = ImageDtype(value_type, shape=sizes.pop())
            else:
                dtype = ImageDtype(value_type, ndim=2)
        return TensorArray._from_sequence(pixels, dtype=dtype)

    def descriptors(self) -> pd.DataFrame:
        """The cutout descriptors as a plain DataFrame — no pixel access."""
        combined = self._storage.combine_chunks()
        table = pa.table({field.name: combined.field(field.name) for field in CUTOUT_DESCRIPTOR_TYPE})
        return table.to_pandas(types_mapper=pd.ArrowDtype)


class CutoutSeries(BaseImageSeries):
    """A Series of image cutouts: descriptor rows over a shared image store.

    Shares the :class:`BaseImageSeries` API with tensor-backed image columns
    (``to_stack``, ``to_image_stack``, ``shapes``, thumbnail HTML reprs);
    scalar access returns numpy views into the store's pixel planes.
    """

    @property
    def store(self) -> ImageStore | None:
        """The attached image store, or None."""
        self._require_image_column()
        return self.array.store

    def with_store(self, store: ImageStore | None) -> CutoutSeries:
        """Return the same series with a different store attached."""
        self._require_image_column()
        return CutoutSeries(
            pd.Series(self.array.with_store(store), index=self.index, name=self.name)
        )

    def to_tensor(self, dtype: TensorDtype | None = None):
        """Materialize into a tensor-backed :class:`ImageSeries`."""
        from nested_pandas.tensor.imageseries import ImageSeries

        self._require_image_column()
        return ImageSeries(pd.Series(self.array.to_tensor(dtype), index=self.index, name=self.name))

    def descriptors(self) -> pd.DataFrame:
        """The cutout descriptors as a plain DataFrame — no pixel access."""
        self._require_image_column()
        frame = self.array.descriptors()
        frame.index = self.index
        return frame

    def __repr__(self) -> str:
        """Text repr from descriptors only — never fetches pixels."""
        if not self._is_image_column():
            return super().__repr__()
        lines = []
        for position in range(len(self)):
            descriptor = self.array._descriptor_at(position)
            label = f"{self.index[position]}"
            if descriptor is pd.NA:
                lines.append(f"{label}    <NA>")
                continue
            lines.append(
                f"{label}    image {descriptor['image_id']} "
                f"[{descriptor['y0']}:{descriptor['y0'] + descriptor['height']}, "
                f"{descriptor['x0']}:{descriptor['x0'] + descriptor['width']}]"
            )
        store_name = type(self.array.store).__name__ if self.array.store is not None else "None"
        lines.append(f"Length: {len(self)}, dtype: {self.dtype}, store: {store_name}")
        return "\n".join(lines)


register_series_class(CutoutDtype, CutoutSeries)
register_html_formatter(CutoutDtype, image_cell_html)
