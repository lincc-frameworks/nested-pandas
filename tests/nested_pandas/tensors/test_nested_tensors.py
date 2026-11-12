"""Tests of tensor columns inside nested columns."""

import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from nested_pandas import NestedDtype, NestedFrame, NestedSeries, TensorDtype, TensorSeries, read_parquet
from nested_pandas.series.ext_array import NestedExtensionArray

STACK = np.arange(24.0).reshape(6, 2, 2)
"""Six 2x2 tensors, packed as two, one and three per row by the fixtures."""


@pytest.fixture
def flat():
    """A flat frame with a tensor column "img", to be packed into three rows."""
    return pd.DataFrame(
        {"t": np.arange(6.0), "img": TensorSeries.from_stack(STACK, index=[0, 0, 1, 2, 2, 2])},
    )


@pytest.fixture
def nf(flat):
    """A frame with a nested column "nested" holding the tensor sub-column "img"."""
    return NestedFrame({"a": [1, 2, 3]}).join_nested(flat, "nested")


def _assert_tensor_series(series, stack):
    assert isinstance(series, TensorSeries)
    assert series.dtype == TensorDtype(pa.fixed_shape_tensor(pa.float64(), [2, 2]))
    np.testing.assert_array_equal(series.to_stack(), stack)


def test_dtype(nf):
    """The dtype of the nested column names the tensor sub-column by its own dtype."""
    dtype = nf.dtypes["nested"]
    assert dtype.name == "nested<t: [double], img: [tensor[double, (2, 2)]]>"
    assert dtype.column_dtype("img") == TensorDtype(pa.fixed_shape_tensor(pa.float64(), [2, 2]))
    assert pd.api.types.pandas_dtype(dtype.name) == dtype


def test_identity_permutation_is_normalized():
    """An identity permutation inside a nested type is dropped, keeping equality and hashing consistent."""
    with_permutation = pa.FixedShapeTensorArray.from_numpy_ndarray(STACK)
    assert with_permutation.type.permutation == [0, 1]
    offsets = pa.array([0, 2, 3, 6], type=pa.int64())
    struct = pa.StructArray.from_arrays([pa.LargeListArray.from_arrays(offsets, with_permutation)], ["img"])
    dtype = NestedExtensionArray(struct).dtype
    expected = NestedDtype({"img": pa.fixed_shape_tensor(pa.float64(), [2, 2])})
    assert dtype == expected
    assert hash(dtype) == hash(expected)


@pytest.mark.parametrize(
    "get",
    [
        lambda nf: nf["nested.img"],
        lambda nf: nf["nested"]["img"],
        lambda nf: nf["nested"].nest["img"],
        lambda nf: nf["nested"].nest.to_flat()["img"],
        lambda nf: nf["nested"].explode()["img"],
        lambda nf: nf.explode("nested")["img"],
    ],
)
def test_flat_access_returns_tensor_series(nf, get):
    """Every way to the flat sub-column gives a TensorSeries of all its tensors."""
    series = get(nf)
    _assert_tensor_series(series, STACK)
    assert series.index.tolist() == [0, 0, 1, 2, 2, 2]


def test_row_access_returns_tensor_series(nf):
    """The dataframe of one row keeps the tensor sub-column typed."""
    row = nf["nested"][2]
    assert isinstance(row, NestedFrame)
    _assert_tensor_series(row["img"], STACK[3:])
    _assert_tensor_series(nf["nested"].iloc[0]["img"], STACK[:2])
    assert row["img"][0].shape == (2, 2)


def test_rows_without_special_columns_stay_plain_dataframes():
    """Rows and flat frames of nested columns without nested or tensor sub-columns are unchanged."""
    nf = NestedFrame({"a": [1]}).join_nested(pd.DataFrame({"t": [1.0, 2.0]}, index=[0, 0]), "nested")
    assert type(nf["nested"][0]) is pd.DataFrame
    assert type(nf["nested"].nest.to_flat()) is pd.DataFrame


def test_row_operations(nf):
    """Operations that select, reorder or introduce missing rows work with a tensor sub-column."""
    _assert_tensor_series(nf[nf["a"] > 1]["nested.img"], STACK[2:])
    _assert_tensor_series(nf.iloc[[2, 0]]["nested.img"], np.concatenate([STACK[3:], STACK[:2]]))
    _assert_tensor_series(nf.query("nested.t > 1")["nested.img"], STACK[2:])
    _assert_tensor_series(nf["nested"][nf["nested.t"] > 1].nest["img"], STACK[2:])
    _assert_tensor_series(
        nf.sort_values("nested.t", ascending=False)["nested.img"], STACK[[1, 0, 2, 5, 4, 3]]
    )
    _assert_tensor_series(nf.dropna(subset="nested.t")["nested.img"], STACK)
    _assert_tensor_series(pd.concat([nf, nf])["nested.img"], np.concatenate([STACK, STACK]))

    reindexed = nf.reindex([0, 1, 5])
    assert reindexed["nested"].isna().tolist() == [False, False, True]
    _assert_tensor_series(reindexed["nested.img"], STACK[:3])


def test_join_nested_with_unmatched_rows(flat):
    """Rows of the frame without nested rows get a missing value."""
    nf = NestedFrame({"a": [1, 2, 3, 4]}).join_nested(flat, "nested")
    assert nf["nested"].isna().tolist() == [False, False, False, True]
    _assert_tensor_series(nf["nested.img"], STACK)


def test_set_rows_from_dataframes(nf):
    """Row dataframes, which hold the tensors as a TensorSeries, can be assigned and packed back."""
    nested = nf["nested"].copy()
    nested[0] = nested[1]
    _assert_tensor_series(nested.nest["img"], STACK[[2, 2, 3, 4, 5]])

    rows = [nf["nested"][0], None, nf["nested"][2]]
    packed = NestedSeries(rows, dtype=nf["nested"].dtype)
    assert packed.isna().tolist() == [False, True, False]
    _assert_tensor_series(packed.nest["img"], STACK[[0, 1, 3, 4, 5]])

    joined = NestedFrame({"a": [1, 2, 3]}).join_nested([nf["nested"][i] for i in range(3)], "other")
    assert joined.dtypes["other"] == nf.dtypes["nested"]


def test_set_tensor_sub_column(nf):
    """A tensor sub-column is set from a TensorSeries or from a numpy stack of one tensor per value."""
    nf["nested.double"] = nf["nested.img"].to_stack() * 2
    _assert_tensor_series(nf["nested.double"], STACK * 2)

    nf["nested.img"] = STACK + 1
    _assert_tensor_series(nf["nested.img"], STACK + 1)

    nf["nested.vector"] = np.zeros((6, 3))
    assert nf["nested.vector"].dtype == TensorDtype(pa.fixed_shape_tensor(pa.float64(), [3]))

    nested = nf["nested"]
    nested.nest["img"] = STACK
    _assert_tensor_series(nested.nest["img"], STACK)

    with pytest.raises(ValueError, match="same length as the flat arrays"):
        nf["nested.img"] = STACK[:2]


def test_map_rows_gets_stacks(nf):
    """Row functions get the tensors of each row as one array of shape (n, *tensor_shape)."""
    result = nf.map_rows(
        lambda img: {"n": img.shape[0], "sum": img.sum()}, columns=["nested.img"], row_container="args"
    )
    assert result["n"].tolist() == [2, 1, 3]
    assert result["sum"].tolist() == [STACK[:2].sum(), STACK[2:3].sum(), STACK[3:].sum()]


def test_round_trips(nf, tmp_path):
    """Parquet and pickle round trips keep the tensor sub-column."""
    path = tmp_path / "nested_tensor.parquet"
    nf.to_parquet(path)
    loaded = read_parquet(path)
    assert loaded.dtypes["nested"] == nf.dtypes["nested"]
    _assert_tensor_series(loaded["nested.img"], STACK)

    _assert_tensor_series(pickle.loads(pickle.dumps(nf))["nested.img"], STACK)


def test_text_repr(nf):
    """Tensors inside a nested column are shown by their values if small, else by a descriptor."""
    assert "img: [[0.0, 1.0], [2.0, 3.0]]" in repr(nf["nested"])

    big = pd.DataFrame({"img": TensorSeries.from_stack(np.zeros((2, 32, 32)), index=[0, 0])})
    big_nf = NestedFrame({"a": [1]}).join_nested(big, "nested")
    assert "[{img: [32×32] float64}; …] (2 rows)" in repr(big_nf)


def test_html_repr(nf):
    """Tensors inside a nested column are rendered as tensor columns of the frame are."""
    html = nf._repr_html_()
    assert "<pre" in html
    assert "[[0. 1.]\n [2. 3.]]" in html
    assert "@@" not in html


def test_html_repr_caps_thumbnails():
    """The limit on rendered thumbnails holds across all the cells of the nested column."""
    pytest.importorskip("matplotlib")
    n_rows = 8
    big = pd.DataFrame(
        {"img": TensorSeries.from_stack(np.random.default_rng(0).random((2 * n_rows, 32, 32)))},
        index=np.repeat(np.arange(n_rows), 2),
    )
    nf = NestedFrame({"a": range(n_rows)}).join_nested(big, "nested")
    html = nf._repr_html_()
    assert html.count('title="[32×32] float64"') == 10
    assert html.count("not rendered in preview") == 2 * n_rows - 10


def test_tensor_in_double_nesting(nf):
    """A tensor sub-column two levels down is reached by its path and keeps its type."""
    nf["group"] = [0, 0, 1]
    double_nf = NestedFrame.from_flat(nf, base_columns=[], on="group", name="outer")
    assert double_nf.dtypes["outer"].name == (
        "nested<a: [int64], nested: [nested<t: [double], img: [tensor[double, (2, 2)]]>]>"
    )
    series = double_nf["outer.nested.img"]
    _assert_tensor_series(series, STACK)
    _assert_tensor_series(double_nf.query("outer.nested.t > 2")["outer.nested.img"], STACK[3:])
    assert "img: [[0.0, 1.0], [2.0, 3.0]]" in repr(double_nf["outer"][0]["nested"])
    assert "<pre" in double_nf._repr_html_()
