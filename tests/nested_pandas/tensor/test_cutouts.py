import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow.parquet as pq
import pytest

import nested_pandas as npd
from nested_pandas import (
    BaseImageSeries,
    CutoutArray,
    CutoutDtype,
    CutoutSeries,
    DictImageStore,
    ImageDtype,
    ImageSeries,
    TensorArray,
)
from nested_pandas.tensor.arrow_ext import ImageType


class CountingStore(DictImageStore):
    """DictImageStore that counts pixel accesses and read plans."""

    def __init__(self, images):
        super().__init__(images)
        self.reads = 0
        self.planned = []

    def get_image(self, image_id):
        self.reads += 1
        return super().get_image(image_id)

    def plan_reads(self, image_ids):
        self.planned.extend(image_ids)


@pytest.fixture
def store():
    rng = np.random.default_rng(0)
    return CountingStore(
        {
            "img1": rng.normal(size=(100, 100)).astype(np.float32),
            "img2": rng.normal(size=(80, 80)).astype(np.float32),
        }
    )


@pytest.fixture
def cutouts(store):
    return CutoutArray.from_descriptors(
        image_id=["img1", "img1", "img2"],
        x0=[10, 12, 5],
        y0=[20, 22, 5],
        width=[8, 8, 8],
        height=[8, 8, 8],
        store=store,
    )


def test_dtype_roundtrip():
    dtype = CutoutDtype()
    assert dtype.name == "cutout"
    assert CutoutDtype.construct_from_string("cutout") == dtype
    assert pd.api.types.pandas_dtype("cutout") == dtype
    with pytest.raises(TypeError):
        CutoutDtype.construct_from_string("tensor[float, (2, 2)]")


def test_scalar_access_is_view(cutouts, store):
    pixels = cutouts[0]
    assert pixels.shape == (8, 8)
    npt.assert_array_equal(pixels, store.get_image("img1")[20:28, 10:18])
    assert np.shares_memory(pixels, store.get_image("img1"))
    # Overlapping cutouts share memory rather than copying.
    assert np.shares_memory(cutouts[0], cutouts[1])


def test_requires_store_for_pixels(cutouts):
    detached = cutouts.with_store(None)
    with pytest.raises(ValueError, match="no image store attached"):
        detached[0]
    # Descriptor-level operations still work without a store.
    npt.assert_array_equal(detached.shapes, np.tile([8, 8], (3, 1)))
    assert len(detached.descriptors()) == 3


def test_shapes_and_descriptors_do_not_read_pixels(cutouts, store):
    baseline = store.reads
    cutouts.shapes
    cutouts.descriptors()
    cutouts.isna()
    assert store.reads == baseline


def test_to_stack(cutouts, store):
    stack = cutouts.to_stack()
    assert store.planned == ["img1", "img1", "img2"]
    assert stack.shape == (3, 8, 8)
    npt.assert_array_equal(stack[2], store.get_image("img2")[5:13, 5:13])


def test_to_stack_mixed_sizes_raises(store):
    mixed = CutoutArray.from_descriptors(
        image_id=["img1", "img2"], x0=[0, 0], y0=[0, 0], width=[8, 4], height=[8, 4], store=store
    )
    with pytest.raises(ValueError, match="mixed shapes"):
        mixed.to_stack()


def test_missing_rows(cutouts):
    with_na = cutouts.take([0, -1, 2], allow_fill=True)
    npt.assert_array_equal(with_na.isna(), [False, True, False])
    assert with_na[1] is pd.NA
    npt.assert_array_equal(with_na.shapes[1], [0, 0])
    stack = with_na.to_stack()
    assert np.isnan(stack[1]).all()
    npt.assert_array_equal(stack[0], cutouts[0])


def test_store_survives_operations(cutouts, store):
    assert cutouts.take([2, 0]).store is store
    assert cutouts[1:3].store is store
    assert cutouts[np.array([True, False, True])].store is store
    assert cutouts.copy().store is store
    assert CutoutArray._concat_same_type([cutouts, cutouts]).store is store


def test_concat_different_stores_chains(cutouts, store):
    other_store = DictImageStore({"img3": np.ones((10, 10), np.float32)})
    other = CutoutArray.from_descriptors(
        image_id=["img3"], x0=[0], y0=[0], width=[8], height=[8], store=other_store
    )
    combined = CutoutArray._concat_same_type([cutouts, other])
    # Distinct stores merge into a chain that resolves ids from both.
    npt.assert_array_equal(combined[0], cutouts[0])
    npt.assert_array_equal(combined[3], np.ones((8, 8)))
    assert "img1" in combined.store and "img3" in combined.store


def test_series_class_and_shared_api(cutouts):
    nf = npd.NestedFrame({"a": [1, 2, 3]})
    nf["cut"] = cutouts
    column = nf["cut"]
    assert isinstance(column, CutoutSeries)
    assert isinstance(column, BaseImageSeries)
    assert not isinstance(column, ImageSeries)
    npt.assert_array_equal(column.to_image_stack(), cutouts.to_stack())
    npt.assert_array_equal(column.shapes, cutouts.shapes)
    # The tensor-backed series shares the same base class and API.
    tensor_column = column.to_tensor()
    assert isinstance(tensor_column, ImageSeries)
    assert isinstance(tensor_column, BaseImageSeries)
    npt.assert_array_equal(tensor_column.to_image_stack(), column.to_image_stack())


def test_series_repr_uses_descriptors_only(cutouts, store):
    series = CutoutSeries(pd.Series(cutouts, name="cut"))
    baseline = store.reads
    text = repr(series)
    assert store.reads == baseline
    assert "image img1 [20:28, 10:18]" in text
    assert "store: CountingStore" in text


def test_to_tensor(cutouts):
    tensor = cutouts.to_tensor()
    assert isinstance(tensor, TensorArray)
    assert tensor.dtype == ImageDtype("float32", shape=(8, 8))
    npt.assert_array_equal(tensor.to_stack(), cutouts.to_stack())


def test_to_tensor_mixed_sizes_gives_ragged(store):
    mixed = CutoutArray.from_descriptors(
        image_id=["img1", "img2"], x0=[0, 0], y0=[0, 0], width=[8, 4], height=[8, 4], store=store
    )
    tensor = mixed.to_tensor()
    assert tensor.dtype == ImageDtype("float32", ndim=2)
    assert tensor[1].shape == (4, 4)


def test_parquet_roundtrip_converts_to_tensor(tmp_path, cutouts):
    nf = npd.NestedFrame({"cut": pd.Series(cutouts)})
    path = tmp_path / "cutouts.parquet"
    nf.to_parquet(path)

    # The file holds an ordinary image column, not descriptors.
    assert isinstance(pq.read_schema(path).field("cut").type, ImageType)

    back = npd.read_parquet(path)
    assert isinstance(back["cut"].dtype, ImageDtype)
    assert isinstance(back["cut"], ImageSeries)
    npt.assert_array_equal(back["cut"].to_image_stack(), cutouts.to_stack())


def test_serialize_without_store_raises(cutouts, tmp_path):
    nf = npd.NestedFrame({"cut": pd.Series(cutouts.with_store(None))})
    with pytest.raises(ValueError, match="no image store attached"):
        nf.to_parquet(tmp_path / "detached.parquet")
