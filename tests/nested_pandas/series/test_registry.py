"""Tests of the registry of column kinds, and of how nested columns inside nested columns use it."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from nested_pandas import NestedDtype, NestedFrame, NestedSeries, TensorDtype, TensorSeries
from nested_pandas.datasets import generate_data
from nested_pandas.series.registry import (
    arrow_type_from_pandas_dtype,
    arrow_type_from_string,
    has_special_dtype,
    html_cell_formatter,
    pandas_dtype_from_arrow,
    text_cell_formatter,
    wrap_series,
)
from nested_pandas.series.utils import (
    contains_extension_type,
    if_else,
    list_scalar,
    normalize_tensor_types,
    null_scalar,
    storage_type,
)

TENSOR_TYPE = pa.fixed_shape_tensor(pa.float64(), [2, 2])
LIST_STRUCT_TYPE = pa.large_list(pa.struct({"a": pa.int64()}))


@pytest.fixture
def double_nf():
    """A frame with a nested column "outer" of sub-columns "a", "b" and the nested "inner"."""
    nf = generate_data(5, 3, seed=1).rename(columns={"nested": "inner"})
    nf["id"] = [0, 0, 0, 1, 1]
    return NestedFrame.from_flat(nf, base_columns=[], on="id", name="outer")


def test_pandas_dtype_from_arrow():
    """Registered kinds get their extension dtype, everything else a pd.ArrowDtype."""
    assert pandas_dtype_from_arrow(LIST_STRUCT_TYPE) == NestedDtype(LIST_STRUCT_TYPE)
    assert pandas_dtype_from_arrow(TENSOR_TYPE) == TensorDtype(TENSOR_TYPE)
    assert pandas_dtype_from_arrow(pa.float64()) == pd.ArrowDtype(pa.float64())
    assert pandas_dtype_from_arrow(pa.large_list(pa.float64())) == pd.ArrowDtype(pa.large_list(pa.float64()))


def test_pandas_dtype_from_arrow_unsupported_type():
    """A type its extension dtype rejects stays a pd.ArrowDtype."""
    permuted = pa.fixed_shape_tensor(pa.float64(), [2, 3], permutation=[1, 0])
    assert pandas_dtype_from_arrow(permuted) == pd.ArrowDtype(permuted)


def test_wrap_series():
    """Series are returned as the subclass registered for their dtype."""
    tensors = pd.Series(TensorSeries.from_stack(np.zeros((2, 2, 2))).array)
    assert type(tensors) is pd.Series
    assert isinstance(wrap_series(tensors), TensorSeries)

    nested = pd.Series(generate_data(2, 2, seed=1)["nested"].array)
    assert isinstance(wrap_series(nested), NestedSeries)

    plain = pd.Series([1, 2])
    assert wrap_series(plain) is plain
    wrapped = wrap_series(tensors)
    assert wrap_series(wrapped) is wrapped


def test_formatters_and_special_dtypes():
    """Nested and tensor dtypes have HTML and text formatters, other dtypes do not."""
    nested_dtype, tensor_dtype = NestedDtype(LIST_STRUCT_TYPE), TensorDtype(TENSOR_TYPE)
    for dtype in (nested_dtype, tensor_dtype):
        assert callable(html_cell_formatter(dtype))
        assert callable(text_cell_formatter(dtype))
    assert html_cell_formatter(np.dtype("float64")) is None
    assert text_cell_formatter(pd.ArrowDtype(pa.float64())) is None
    # A fresh HTML formatter per column, since they keep state
    assert html_cell_formatter(tensor_dtype) is not html_cell_formatter(tensor_dtype)

    assert has_special_dtype([np.dtype("float64"), tensor_dtype])
    assert has_special_dtype([nested_dtype])
    assert not has_special_dtype([np.dtype("float64"), pd.ArrowDtype(pa.float64())])


def test_double_nesting_types(double_nf):
    """A nested column inside a nested column is a NestedSeries on every access path."""
    for series in (
        double_nf["outer.inner"],
        double_nf["outer"]["inner"],
        double_nf["outer"].nest["inner"],
        double_nf["outer"].nest.to_flat()["inner"],
        double_nf.explode("outer")["inner"],
        double_nf["outer"][0]["inner"],
    ):
        assert isinstance(series, NestedSeries)
        assert series.columns == ["t", "flux", "flux_error", "band"]
    assert isinstance(double_nf["outer"][0], NestedFrame)
    assert isinstance(double_nf["outer"].nest.to_flat(), NestedFrame)


def test_double_nesting_dtype_string(double_nf):
    """The name of a doubly nested dtype parses back to it."""
    dtype = double_nf.dtypes["outer"]
    assert dtype.name == (
        "nested<a: [double], b: [double], "
        "inner: [nested<t: [double], flux: [double], flux_error: [double], band: [large_string]>]>"
    )
    assert pd.api.types.pandas_dtype(dtype.name) == dtype


def test_dtype_string_with_parametric_types():
    """Parametric pyarrow types pandas can parse are supported in nested dtype strings."""
    dtype = NestedDtype(
        {"t": pa.timestamp("ns"), "img": pa.fixed_shape_tensor(pa.int32(), [4], dim_names=["x"])}
    )
    assert dtype.name == "nested<t: [timestamp[ns]], img: [tensor[int32, (4,), dim_names=[x]]]>"
    assert NestedDtype.construct_from_string(dtype.name) == dtype
    with pytest.raises(TypeError, match="Cannot parse the type of 'a'"):
        NestedDtype.construct_from_string("nested<a: [decimal128(5, 2)]>")
    with pytest.raises(TypeError, match="expects a string"):
        NestedDtype.construct_from_string(1)


def test_double_nesting_text_repr(double_nf):
    """The inner nested column is shown in the same short format as the outer one."""
    text = double_nf["outer"][0].to_string()
    assert "[{t: 8.38389, flux: " in text
    assert "; …] (3 rows)" in text
    assert "inner: [{t: 8.3838" in double_nf["outer"].to_string()
    assert "<nested_pandas" not in repr(double_nf)


def test_double_nesting_html_repr(double_nf):
    """The inner nested column is rendered as a table inside the table of the outer one."""
    html = double_nf._repr_html_()
    # The styled table of the frame, one table per outer cell and one per inner cell shown
    assert html.count("<table") == 1 + 2 + 3
    assert "<th>flux_error</th>" in html
    assert "<i>+2 rows</i>" in html
    assert "@@" not in html
    assert "\\n" not in html


def test_storage_type():
    """Extension types are replaced by their storage at any depth of structs and lists."""
    nested = pa.struct({"t": pa.large_list(pa.float64()), "img": pa.large_list(TENSOR_TYPE)})
    assert contains_extension_type(nested)
    assert contains_extension_type(pa.list_(pa.struct({"img": TENSOR_TYPE}), 2))
    assert not contains_extension_type(pa.struct({"t": pa.large_list(pa.float64())}))

    expected = pa.struct({"t": pa.large_list(pa.float64()), "img": pa.large_list(pa.list_(pa.float64(), 4))})
    assert storage_type(nested) == expected
    assert storage_type(pa.list_(TENSOR_TYPE)) == pa.list_(pa.list_(pa.float64(), 4))
    assert storage_type(pa.list_(TENSOR_TYPE, 3)) == pa.list_(pa.list_(pa.float64(), 4), 3)
    assert storage_type(pa.int64()) == pa.int64()


def test_normalize_tensor_types():
    """Identity permutations are dropped, other tensor types are untouched."""
    with_identity = pa.fixed_shape_tensor(pa.float64(), [2, 2], permutation=[0, 1])
    normalized = normalize_tensor_types(pa.large_list(pa.struct({"img": with_identity})))
    assert normalized.value_type.field("img").type.permutation is None
    permuted = pa.fixed_shape_tensor(pa.float64(), [2, 3], permutation=[1, 0])
    assert normalize_tensor_types(permuted).permutation == [1, 0]
    assert normalize_tensor_types(pa.int64()) == pa.int64()


def test_scalars_and_if_else_with_extension_types():
    """The helpers do what pyarrow does not implement for extension types inside structs and lists."""
    tensors = pa.ExtensionArray.from_storage(
        TENSOR_TYPE, pa.FixedSizeListArray.from_arrays(pa.array(np.arange(12.0)), 4)
    )
    list_type = pa.large_list(TENSOR_TYPE)
    assert not null_scalar(pa.struct({"img": list_type})).is_valid

    scalar = list_scalar(pa.chunked_array([tensors]), list_type)
    assert scalar.type == list_type
    assert scalar.values.equals(tensors)
    assert list_scalar(pa.array([1, 2]), pa.large_list(pa.int64())).as_py() == [1, 2]

    lists = pa.LargeListArray.from_arrays(pa.array([0, 1, 3], type=pa.int64()), tensors)
    result = if_else(pa.array([True, False]), lists[1], lists)
    assert result.type == lists.type
    assert result.to_pylist() == [lists[1].as_py(), lists[1].as_py()]
    assert if_else(pa.array([True, False]), pa.scalar(0), pa.array([1, 2])).to_pylist() == [0, 2]


def test_compact_view_text(double_nf):
    """The compact view shows inner nested columns by their sub-column names and number of rows."""
    with pd.option_context("display.max_colwidth", 200):
        compact = repr(double_nf.compact_view())
        full = repr(double_nf)
    assert "inner: [t, flux, flux_error, band] (3 rows)}; …] (3 rows)" in compact
    assert "inner: [{t: 8.38389" in full
    assert "inner: [{t:" not in compact
    # The view does not change how the frame is displayed afterwards
    with pd.option_context("display.max_colwidth", 200):
        assert repr(double_nf) == full


def test_compact_view_html(double_nf):
    """In HTML, only the tables of the outer nested column are rendered."""
    compact = double_nf.compact_view()._repr_html_()
    # The styled table of the frame and one table per outer cell
    assert compact.count("<table") == 1 + 2
    assert "[t, flux, flux_error, band] <i>3 rows</i>" in compact
    assert double_nf._repr_html_().count("<table") == 1 + 2 + 3


def test_compact_view_without_inner_nesting():
    """Frames without nested columns inside nested columns display the same in the compact view."""
    nf = generate_data(3, 2, seed=1)
    assert repr(nf.compact_view()) == repr(nf)
    assert nf.compact_view()._repr_html_().count("<table") == nf._repr_html_().count("<table")
    plain = NestedFrame({"a": [1, 2]})
    assert repr(plain.compact_view()) == repr(plain)


def test_compact_format_of_nested_values():
    """Long lists of sub-column names are cut, and missing values and single rows are handled."""
    from nested_pandas.series.ext_array import format_nested_value_compact

    wide = pd.DataFrame({f"c{i}": [1] for i in range(7)})
    assert format_nested_value_compact(wide) == "[c0, c1, c2, c3, c4, …] (1 row)"
    assert format_nested_value_compact(None) == "<NA>"
    assert format_nested_value_compact(pd.NA) == "<NA>"


@pytest.mark.parametrize(
    "string,expected",
    [
        ("double", pa.float64()),
        ("string", pa.string()),
        ("timestamp[ns]", pa.timestamp("ns")),
        ("timestamp[ns, tz=UTC]", pa.timestamp("ns", tz="UTC")),
        ("tensor[double, (2, 2)]", TENSOR_TYPE),
        ("nested<a: [int64]>", LIST_STRUCT_TYPE),
        (
            "nested<a: [nested<b: [tensor[double, (2, 2)]]>]>",
            NestedDtype({"a": NestedDtype({"b": TENSOR_TYPE}).list_struct_pa_dtype}).list_struct_pa_dtype,
        ),
        # an extension dtype which is not registered with nested-pandas, only with pandas
        ("Int64", pa.int64()),
    ],
)
def test_arrow_type_from_string(string, expected):
    """Names of pyarrow types and of pandas extension dtypes are parsed to pyarrow types."""
    assert arrow_type_from_string(string) == expected


def test_arrow_type_from_string_not_understood():
    """Strings which name neither kind of type are rejected."""
    for string in ("garbage", "decimal128(5, 2)", "float64x"):
        with pytest.raises(TypeError, match="neither the name"):
            arrow_type_from_string(string)


def test_arrow_type_from_pandas_dtype():
    """The pyarrow type of a dtype inverts the dtype of a pyarrow type."""
    for pa_type in (TENSOR_TYPE, LIST_STRUCT_TYPE, pa.float64(), pa.large_list(pa.float64())):
        assert arrow_type_from_pandas_dtype(pandas_dtype_from_arrow(pa_type)) == pa_type
    assert arrow_type_from_pandas_dtype(pd.BooleanDtype()) == pa.bool_()
