import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import nested_pandas as npd
from nested_pandas import ImageDtype, ImageSeries, NestedFrame, TensorArray, TensorDtype, TensorSeries
from nested_pandas.tensor.arrow_ext import TensorType, is_tensor_pyarrow_type


@pytest.fixture
def fixed_array():
    return TensorArray.from_stack(np.arange(24, dtype=np.float32).reshape(4, 2, 3))


@pytest.fixture
def ragged_array():
    return TensorArray._from_sequence([np.ones((1, 2), np.float64), np.zeros((2, 3), np.float64), None])


def test_dtype_fixed_shape():
    dtype = TensorDtype("float32", shape=(25, 25))
    assert dtype.is_fixed_shape
    assert dtype.shape == (25, 25)
    assert dtype.ndim == 2
    assert dtype.value_type == pa.float32()
    assert dtype.name == "tensor[float, (25, 25)]"


def test_dtype_variable_shape():
    dtype = TensorDtype(pa.float64(), ndim=3)
    assert not dtype.is_fixed_shape
    assert dtype.shape is None
    assert dtype.ndim == 3
    assert dtype.name == "tensor[double, ndim=3]"


def test_dtype_string_roundtrip():
    for dtype in (
        TensorDtype("float32", shape=(25, 25)),
        TensorDtype("float64", ndim=2),
        TensorDtype("int16", shape=(7,)),
    ):
        assert TensorDtype.construct_from_string(dtype.name) == dtype
        assert pd.api.types.pandas_dtype(dtype.name) == dtype


def test_dtype_construct_from_string_rejects():
    with pytest.raises(TypeError):
        TensorDtype.construct_from_string("int64")
    with pytest.raises(TypeError):
        TensorDtype.construct_from_string("tensor[float]")
    with pytest.raises(TypeError):
        TensorDtype.construct_from_string(1)


def test_dtype_requires_shape_or_ndim():
    with pytest.raises(ValueError):
        TensorDtype("float32")
    with pytest.raises(ValueError):
        TensorDtype("float32", shape=())
    with pytest.raises(ValueError):
        TensorDtype("float32", ndim=0)


def test_dtype_from_pyarrow():
    fixed = TensorDtype.from_pyarrow(pa.fixed_shape_tensor(pa.float32(), (2, 3)))
    assert fixed == TensorDtype("float32", shape=(2, 3))
    variable = TensorDtype.from_pyarrow(TensorType(pa.float64(), 2))
    assert variable == TensorDtype("float64", ndim=2)
    declared = TensorDtype.from_pyarrow(TensorType(pa.float32(), 2, shape=(2, 3)))
    assert declared == TensorDtype("float32", shape=(2, 3))
    with pytest.raises(TypeError):
        TensorDtype.from_pyarrow(pa.float32())


def test_from_stack_and_scalars(fixed_array):
    assert len(fixed_array) == 4
    assert fixed_array.dtype == TensorDtype("float32", shape=(2, 3))
    np.testing.assert_array_equal(fixed_array[1], np.arange(6, 12, dtype=np.float32).reshape(2, 3))
    np.testing.assert_array_equal(fixed_array[-1], fixed_array[3])
    with pytest.raises(IndexError):
        fixed_array[4]


def test_scalar_access_is_zero_copy(fixed_array):
    buffer = fixed_array._storage.chunk(0).buffers()[-1]
    scalar = fixed_array[2]
    itemsize = np.dtype(np.float32).itemsize
    assert scalar.__array_interface__["data"][0] == buffer.address + 2 * 6 * itemsize


def test_to_stack_zero_copy(fixed_array):
    buffer = fixed_array._storage.chunk(0).buffers()[-1]
    stack = fixed_array.to_stack()
    assert stack.__array_interface__["data"][0] == buffer.address
    np.testing.assert_array_equal(stack, np.arange(24, dtype=np.float32).reshape(4, 2, 3))


def test_from_sequence_infers_fixed_shape():
    array = TensorArray._from_sequence([np.ones((2, 2), np.float32), np.zeros((2, 2), np.float32)])
    assert array.dtype == TensorDtype("float32", shape=(2, 2))


def test_from_sequence_infers_variable_shape():
    array = TensorArray._from_sequence([np.ones((2, 2)), np.zeros((3, 1))])
    assert array.dtype == TensorDtype("float64", ndim=2)


def test_from_sequence_mixed_ndim_raises():
    with pytest.raises(ValueError, match="mixed ndim"):
        TensorArray._from_sequence([np.ones((2, 2)), np.zeros(3)])


def test_from_sequence_all_na_requires_dtype():
    with pytest.raises(ValueError, match="pass dtype explicitly"):
        TensorArray._from_sequence([None, pd.NA])
    array = TensorArray._from_sequence([None, pd.NA], dtype=TensorDtype("float32", shape=(2, 2)))
    assert len(array) == 2
    assert array.isna().all()


def test_from_sequence_shape_mismatch_raises():
    with pytest.raises(ValueError, match="does not match dtype shape"):
        TensorArray._from_sequence([np.ones((3, 3))], dtype=TensorDtype("float32", shape=(2, 2)))


def test_missing_values(ragged_array):
    np.testing.assert_array_equal(ragged_array.isna(), [False, False, True])
    assert ragged_array[2] is pd.NA
    dropped = ragged_array.dropna()
    assert len(dropped) == 2
    assert not dropped.isna().any()


def test_getitem_slice_mask_fancy(fixed_array):
    assert len(fixed_array[1:3]) == 2
    np.testing.assert_array_equal(fixed_array[1:3][0], fixed_array[1])
    stepped = fixed_array[::2]
    assert len(stepped) == 2
    np.testing.assert_array_equal(stepped[1], fixed_array[2])
    masked = fixed_array[np.array([True, False, False, True])]
    np.testing.assert_array_equal(masked[1], fixed_array[3])
    fancy = fixed_array[np.array([3, 0])]
    np.testing.assert_array_equal(fancy[0], fixed_array[3])


def test_take_allow_fill(fixed_array):
    taken = fixed_array.take([1, -1, 0], allow_fill=True)
    np.testing.assert_array_equal(taken.isna(), [False, True, False])
    np.testing.assert_array_equal(taken[0], fixed_array[1])
    filled = fixed_array.take([1, -1], allow_fill=True, fill_value=np.zeros((2, 3), np.float32))
    np.testing.assert_array_equal(filled[1], np.zeros((2, 3)))
    with pytest.raises(ValueError):
        fixed_array.take([1, -2], allow_fill=True)
    with pytest.raises(IndexError):
        fixed_array.take([10])


def test_take_negative_wraps(fixed_array):
    taken = fixed_array.take([-1, 0])
    np.testing.assert_array_equal(taken[0], fixed_array[3])


def test_setitem(fixed_array):
    fixed_array[1] = np.full((2, 3), 7, dtype=np.float32)
    np.testing.assert_array_equal(fixed_array[1], np.full((2, 3), 7))
    fixed_array[0] = pd.NA
    assert fixed_array[0] is pd.NA


def test_concat_and_equals(fixed_array):
    doubled = TensorArray._concat_same_type([fixed_array, fixed_array])
    assert len(doubled) == 8
    np.testing.assert_array_equal(doubled[5], fixed_array[1])
    assert fixed_array.equals(fixed_array.copy())
    other = TensorArray._from_sequence([np.ones((3, 3))])
    with pytest.raises(TypeError):
        TensorArray._concat_same_type([fixed_array, other])


def test_to_stack_with_missing_fills_na():
    array = TensorArray._from_sequence(
        [np.ones((2, 2), np.float32), None], dtype=TensorDtype("float32", shape=(2, 2))
    )
    stack = array.to_stack()
    np.testing.assert_array_equal(stack[0], np.ones((2, 2)))
    assert np.isnan(stack[1]).all()


def test_to_stack_variable_shapes(ragged_array):
    with pytest.raises(ValueError, match="Cannot stack"):
        ragged_array.to_stack()
    uniform = TensorArray._from_sequence(
        [np.ones((2, 2)), np.zeros((2, 2))], dtype=TensorDtype("float64", ndim=2)
    )
    np.testing.assert_array_equal(uniform.to_stack()[1], np.zeros((2, 2)))


def test_shapes(fixed_array, ragged_array):
    np.testing.assert_array_equal(fixed_array.shapes, np.tile([2, 3], (4, 1)))
    np.testing.assert_array_equal(ragged_array.shapes, [[1, 2], [2, 3], [0, 0]])


def test_to_numpy_is_1d_object(fixed_array):
    result = fixed_array.to_numpy()
    assert result.shape == (4,)
    assert result.dtype == object
    np.testing.assert_array_equal(result[1], fixed_array[1])


def test_astype_between_tensor_dtypes(fixed_array):
    variable = fixed_array.astype(TensorDtype("float64", ndim=2))
    assert variable.dtype == TensorDtype("float64", ndim=2)
    np.testing.assert_array_equal(variable[1], fixed_array[1])


def test_pandas_series_construction(fixed_array):
    series = pd.Series(fixed_array)
    assert series.dtype == fixed_array.dtype
    assert isinstance(series.array, TensorArray)


def test_nestedframe_returns_tensor_series(fixed_array):
    nf = npd.NestedFrame({"a": [1, 2, 3, 4]})
    nf["tensor"] = fixed_array
    column = nf["tensor"]
    assert isinstance(column, TensorSeries)
    assert column.tensor_shape == (2, 3)
    assert column.tensor_ndim == 2
    assert column.value_dtype == np.float32
    np.testing.assert_array_equal(column.to_stack(), fixed_array.to_stack())
    np.testing.assert_array_equal(column.shapes, fixed_array.shapes)


def test_tensor_series_guards_wrong_dtype():
    series = TensorSeries([1, 2, 3])
    with pytest.raises(TypeError, match="TensorDtype"):
        _ = series.tensor_shape


def test_repr_cells(fixed_array):
    nf = npd.NestedFrame({"tensor": pd.Series(fixed_array)})
    text = repr(nf)
    assert "[2×3] float32" in text


@pytest.mark.parametrize("with_na", [False, True])
def test_parquet_roundtrip_fixed(tmp_path, with_na):
    tensors = [np.arange(4, dtype=np.float32).reshape(2, 2), np.ones((2, 2), np.float32)]
    if with_na:
        tensors.append(None)
    array = TensorArray._from_sequence(tensors, dtype=TensorDtype("float32", shape=(2, 2)))
    nf = npd.NestedFrame({"tensor": pd.Series(array)})
    path = tmp_path / "fixed.parquet"
    nf.to_parquet(path)

    file_type = pq.read_schema(path).field("tensor").type
    assert is_tensor_pyarrow_type(file_type)
    if with_na:
        # Nullable fixed-shape columns cannot use the canonical type in parquet.
        assert isinstance(file_type, TensorType)
    else:
        assert isinstance(file_type, pa.FixedShapeTensorType)

    back = npd.read_parquet(path)
    assert back["tensor"].dtype == array.dtype
    assert isinstance(back["tensor"], TensorSeries)
    np.testing.assert_array_equal(back["tensor"].array.isna(), array.isna())
    np.testing.assert_array_equal(back["tensor"].to_stack(), array.to_stack())


def test_parquet_roundtrip_variable(tmp_path, ragged_array):
    nf = npd.NestedFrame({"tensor": pd.Series(ragged_array)})
    path = tmp_path / "ragged.parquet"
    nf.to_parquet(path)
    back = npd.read_parquet(path)
    assert back["tensor"].dtype == ragged_array.dtype
    np.testing.assert_array_equal(back["tensor"].array[1], ragged_array[1])
    assert back["tensor"].array[2] is pd.NA
    np.testing.assert_array_equal(back["tensor"].shapes, ragged_array.shapes)


def test_from_pyarrow_table(fixed_array):
    table = pa.table({"a": [1, 2, 3, 4], "tensor": fixed_array.__arrow_array__()})
    nf = npd.from_pyarrow(table)
    assert nf["tensor"].dtype == fixed_array.dtype
    assert isinstance(nf["tensor"], TensorSeries)


def test_unregistered_readers_get_storage(tmp_path, fixed_array):
    """Plain parquet readers see the tensor column as its storage type."""
    nf = npd.NestedFrame({"tensor": pd.Series(fixed_array)})
    path = tmp_path / "fixed.parquet"
    nf.to_parquet(path)
    # The canonical fixed_shape_tensor storage is a fixed-size list; a reader
    # without the extension type registered would degrade to exactly this.
    schema = pq.read_schema(path)
    assert schema.field("tensor").type.storage_type == pa.list_(pa.float32(), 6)


def test_tensor_cell_html_renders_viridis_thumbnail_with_colorbar():
    """2-d tensor cells render as a colormapped thumbnail plus a labelled colorbar."""
    pytest.importorskip("matplotlib")
    from nested_pandas.tensor.display import tensor_cell_html

    cell = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    html = tensor_cell_html(cell)
    assert html.count("<img src=") == 2  # thumbnail + colorbar strip
    assert 'title="colorbar"' in html
    assert "[2×2] float32" in html
    # colorbar labels are the displayed (1st-99th percentile) range, top then bottom
    assert html.index("29.7") < html.index("0.3")

    assert tensor_cell_html(pd.NA) == "&lt;NA&gt;"
    assert "[2×3×3] float32" in tensor_cell_html(np.zeros((2, 3, 3), dtype=np.float32))
    assert tensor_cell_html(np.zeros(3, dtype=np.float32)) == "[3] float32"


def test_tensor_cell_html_bool_tensor():
    """Boolean tensors render too (values are cast to float for the colormap)."""
    pytest.importorskip("matplotlib")
    from nested_pandas.tensor.display import tensor_cell_html

    html = tensor_cell_html(np.array([[True, False], [False, True]]))
    assert 'title="colorbar"' in html
    assert "[2×2] bool" in html


def test_tensor_series_repr_html():
    """TensorSeries HTML repr renders the first MAX_RENDERED rows with colorbars; images stay grayscale."""
    pytest.importorskip("matplotlib")
    from nested_pandas.tensor.display import MAX_RENDERED

    array = TensorArray.from_stack(np.arange(4 * (MAX_RENDERED + 2), dtype=np.float32).reshape(-1, 2, 2))
    series = TensorSeries(pd.Series(array, name="tensor"))
    html = series._repr_html_()
    assert "<table>" in html
    assert html.count('title="colorbar"') == MAX_RENDERED
    assert "not rendered in preview" in html
    assert f"dtype: {array.dtype}" in html
    # image columns keep their grayscale, colorbar-free rendering
    image_dtype = ImageDtype("float32", shape=(2, 2))
    image = ImageSeries(pd.Series(TensorArray._from_sequence(list(array.to_numpy()), dtype=image_dtype)))
    assert "colorbar" not in image._repr_html_()


def test_nestedframe_repr_html_uses_tensor_formatter():
    """NestedFrame HTML repr renders every displayed tensor cell with a colorbar."""
    pytest.importorskip("matplotlib")
    array = TensorArray.from_stack(np.zeros((3, 2, 2), dtype=np.float32))
    nf = NestedFrame({"t": array, "x": [1, 2, 3]})
    html = nf._repr_html_()
    assert html.count('title="colorbar"') == 3
