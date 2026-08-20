import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import nested_pandas as npd
from nested_pandas import ImageDtype, ImageSeries, TensorArray, TensorDtype, TensorSeries
from nested_pandas.tensor.arrow_ext import TensorType
from nested_pandas.tensor.display import image_cell_html, image_series_html


@pytest.fixture
def image_array():
    return TensorArray._from_sequence(
        [np.ones((2, 2), np.float32), np.zeros((2, 2), np.float32)],
        dtype=ImageDtype("float32", shape=(2, 2)),
    )


def test_image_dtype_string_roundtrip():
    dtype = ImageDtype("float32", shape=(25, 25))
    assert dtype.name == "image[float, (25, 25)]"
    assert ImageDtype.construct_from_string(dtype.name) == dtype
    assert pd.api.types.pandas_dtype(dtype.name) == dtype
    with pytest.raises(TypeError):
        ImageDtype.construct_from_string("tensor[float, (25, 25)]")


def test_image_dtype_is_tensor_dtype():
    dtype = ImageDtype("float32", ndim=2)
    assert isinstance(dtype, TensorDtype)
    assert not dtype.is_fixed_shape


def test_image_column_returns_image_series(image_array):
    nf = npd.NestedFrame({"a": [1, 2]})
    nf["img"] = image_array
    column = nf["img"]
    assert isinstance(column, ImageSeries)
    assert column.tensor_shape == (2, 2)
    np.testing.assert_array_equal(column.to_image_stack()[0], np.ones((2, 2)))


def test_plain_tensor_column_stays_tensor_series():
    nf = npd.NestedFrame({"a": [1, 2]})
    nf["tensor"] = TensorArray.from_stack(np.zeros((2, 3, 3), np.float32))
    column = nf["tensor"]
    assert isinstance(column, TensorSeries)
    assert not isinstance(column, ImageSeries)


def test_image_dtype_survives_operations(image_array):
    taken = image_array.take([1, 0])
    assert isinstance(taken.dtype, ImageDtype)
    concatenated = TensorArray._concat_same_type([image_array, image_array])
    assert isinstance(concatenated.dtype, ImageDtype)


def test_image_pyarrow_type_carries_kind(image_array):
    ext = image_array.__arrow_array__()
    assert isinstance(ext.type, TensorType)
    assert ext.type.kind == "image"
    assert ext.type.shape == (2, 2)
    assert TensorDtype.from_pyarrow(ext.type) == image_array.dtype
    assert isinstance(TensorDtype.from_pyarrow(ext.type), ImageDtype)


@pytest.mark.parametrize("with_na", [False, True])
def test_parquet_roundtrip_fixed_image(tmp_path, with_na):
    tensors = [np.ones((2, 2), np.float32), np.zeros((2, 2), np.float32)]
    if with_na:
        tensors.append(None)
    array = TensorArray._from_sequence(tensors, dtype=ImageDtype("float32", shape=(2, 2)))
    nf = npd.NestedFrame({"img": pd.Series(array)})
    path = tmp_path / "img.parquet"
    nf.to_parquet(path)

    file_type = pq.read_schema(path).field("img").type
    assert isinstance(file_type, TensorType)
    assert file_type.kind == "image"

    back = npd.read_parquet(path)
    assert isinstance(back["img"].dtype, ImageDtype)
    assert back["img"].dtype == array.dtype
    assert isinstance(back["img"], ImageSeries)
    np.testing.assert_array_equal(back["img"].array.isna(), array.isna())
    np.testing.assert_array_equal(back["img"].to_image_stack(), array.to_stack())


def test_parquet_roundtrip_ragged_image(tmp_path):
    array = TensorArray._from_sequence(
        [np.ones((1, 2), np.float32), np.zeros((3, 2), np.float32), None],
        dtype=ImageDtype("float32", ndim=2),
    )
    nf = npd.NestedFrame({"img": pd.Series(array)})
    path = tmp_path / "img.parquet"
    nf.to_parquet(path)
    back = npd.read_parquet(path)
    assert isinstance(back["img"].dtype, ImageDtype)
    assert back["img"].dtype == array.dtype
    np.testing.assert_array_equal(back["img"].array[1], array[1])
    assert back["img"].array[2] is pd.NA


def test_plain_fixed_tensor_still_canonical(tmp_path):
    """Image serialization must not change the plain tensor rule."""
    array = TensorArray.from_stack(np.zeros((2, 2, 2), np.float32))
    nf = npd.NestedFrame({"tensor": pd.Series(array)})
    path = tmp_path / "tensor.parquet"
    nf.to_parquet(path)
    assert isinstance(pq.read_schema(path).field("tensor").type, pa.FixedShapeTensorType)


def test_image_cell_html_renders_thumbnail(image_array):
    html = image_cell_html(image_array[0])
    assert html.startswith("<img src=") or "[2×2] float32" in html
    assert image_cell_html(pd.NA) == "&lt;NA&gt;"
    # Non-2d cells fall back to descriptor text.
    cube = np.zeros((2, 3, 3), np.float32)
    assert "[2×3×3] float32" in image_cell_html(cube)


def test_image_series_repr_html(image_array):
    series = ImageSeries(pd.Series(image_array, name="img"))
    html = series._repr_html_()
    assert "<table>" in html
    assert "img" in html
    assert f"dtype: {image_array.dtype}" in html


def test_nestedframe_repr_html_uses_image_formatter(image_array):
    nf = npd.NestedFrame({"img": pd.Series(image_array)})
    html = nf._repr_html_()
    assert "<img src=" in html or "[2×2] float32" in html
