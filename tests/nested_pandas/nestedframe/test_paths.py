"""Tests of column path resolution and of the frame methods that use it for doubly nested columns."""

import numpy as np
import pandas as pd
import pytest

from nested_pandas import NestedFrame, NestedSeries
from nested_pandas.datasets import generate_data
from nested_pandas.nestedframe.paths import ColumnPath, resolve_column_path
from nested_pandas.series.dtype import NestedDtype


@pytest.fixture
def flat_nf():
    """The frame that ``double_nf`` is packed from: five rows with a nested column "inner"."""
    nf = generate_data(5, 3, seed=1).rename(columns={"nested": "inner"})
    nf["id"] = [0, 0, 0, 1, 1]
    return nf


@pytest.fixture
def double_nf(flat_nf):
    """A frame with a nested column "outer" of sub-columns "a", "b" and the nested "inner"."""
    return NestedFrame.from_flat(flat_nf, base_columns=[], on="id", name="outer")


def test_resolve_base_column(double_nf):
    """A column of the frame resolves to a path without fields."""
    path = resolve_column_path(double_nf, ["outer"])
    assert path == ColumnPath("outer")
    assert path.is_base and path.is_nested
    with pytest.raises(ValueError, match="not a sub-column"):
        _ = path.parent
    assert path.leaf == "outer"
    assert path.name == "outer"


def test_resolve_sub_columns(double_nf):
    """Sub-columns resolve at any depth, with the dtype of every step."""
    path = resolve_column_path(double_nf, ["outer", "a"])
    assert path == ColumnPath("outer", ("a",))
    assert not path.is_base and not path.is_nested
    assert path.parent == ColumnPath("outer")

    path = resolve_column_path(double_nf, ["outer", "inner", "t"])
    assert path == ColumnPath("outer", ("inner", "t"))
    assert path.leaf == "t"
    assert path.name == "outer.inner.t"
    assert path.parent == ColumnPath("outer", ("inner",))
    assert path.parent.is_nested
    assert [isinstance(dtype, NestedDtype) for dtype in path.dtypes] == [True, True, False]


def test_resolve_unknown(double_nf):
    """Names that do not lead to a column resolve to None."""
    assert resolve_column_path(double_nf, ["missing"]) is None
    assert resolve_column_path(double_nf, ["outer", "missing"]) is None
    assert resolve_column_path(double_nf, ["outer", "inner", "missing"]) is None
    # "a" is not a nested column, so nothing can be below it
    assert resolve_column_path(double_nf, ["outer", "a", "t"]) is None


def test_resolve_new_field(double_nf):
    """With allow_new_field, only the last component may be unknown, and only below a nested column."""
    path = resolve_column_path(double_nf, ["outer", "inner", "new"], allow_new_field=True)
    assert path == ColumnPath("outer", ("inner", "new"))
    assert path.dtype is None
    assert resolve_column_path(double_nf, ["outer", "a", "new"], allow_new_field=True) is None
    assert resolve_column_path(double_nf, ["outer", "new", "t"], allow_new_field=True) is None
    assert resolve_column_path(double_nf, ["new"], allow_new_field=True) is None


def test_resolve_names_with_dots():
    """Dots in names: the longest existing name wins at every level, the whole-name column first."""
    nf = NestedFrame({"a": [1, 2], "n.x": [3, 4]}, index=[0, 1])
    nf = nf.join_nested(pd.DataFrame({"x": [1.0, 2.0], "y.z": [3.0, 4.0]}, index=[0, 1]), "n")
    assert resolve_column_path(nf, ["n", "x"]) == ColumnPath("n.x")
    assert resolve_column_path(nf, ["n", "y", "z"]) == ColumnPath("n", ("y.z",))
    assert nf._resolve_column_path("n.y.z") == ColumnPath("n", ("y.z",))
    assert nf._resolve_column_path("n.x") == ColumnPath("n.x")
    assert nf._resolve_column_path(1) is None


def test_getitem_deep(double_nf, flat_nf):
    """Sub-columns of a nested column inside a nested column are returned flat."""
    inner = double_nf["outer.inner"]
    assert isinstance(inner, NestedSeries)
    assert len(inner) == len(flat_nf)

    t = double_nf["outer.inner.t"]
    np.testing.assert_array_equal(t.to_numpy(), flat_nf["inner.t"].to_numpy())
    assert t.index.tolist() == [0] * 9 + [1] * 6

    with pytest.raises(KeyError, match=r"outer\.inner\.missing"):
        double_nf["outer.inner.missing"]


def test_getitem_list_deep(double_nf):
    """A list of names selects sub-columns at every depth, keeping the requested order."""
    selected = double_nf[["outer.a", "outer.inner.t", "outer.inner.band"]]
    assert selected["outer"].columns == ["a", "inner"]
    assert selected["outer.inner"].columns == ["t", "band"]
    np.testing.assert_array_equal(selected["outer.inner.t"].to_numpy(), double_nf["outer.inner.t"].to_numpy())

    # All of a nested sub-column wins over a part of it
    selected = double_nf[["outer.inner", "outer.inner.t"]]
    assert selected["outer.inner"].columns == double_nf["outer.inner"].columns

    with pytest.raises(KeyError, match=r"outer\.inner\.missing"):
        double_nf[["outer.inner.missing"]]


def test_setitem_deep(double_nf):
    """Sub-columns can be replaced and added at any depth, leaving everything else unchanged."""
    t = double_nf["outer.inner.t"].to_numpy()
    nf = double_nf.copy()
    nf["outer.inner.t"] = nf["outer.inner.t"] + 1.0
    nf["outer.inner.t2"] = t * 2.0
    np.testing.assert_array_equal(nf["outer.inner.t"].to_numpy(), t + 1.0)
    np.testing.assert_array_equal(nf["outer.inner.t2"].to_numpy(), t * 2.0)
    assert nf["outer.inner"].columns == [*double_nf["outer.inner"].columns, "t2"]
    np.testing.assert_array_equal(nf["outer.a"].to_numpy(), double_nf["outer.a"].to_numpy())
    # The original frame is untouched
    assert "t2" not in double_nf["outer.inner"].columns

    nf["outer.inner.t"] = 0.0
    assert (nf["outer.inner.t"] == 0.0).all()


def test_setitem_deep_filled(double_nf):
    """A series aligned with the containing nested column is repeated into each of its lists."""
    nf = double_nf.copy()
    nf["outer.inner.a"] = nf["outer.a"]
    np.testing.assert_array_equal(nf["outer.inner.a"].to_numpy(), np.repeat(nf["outer.a"].to_numpy(), 3))


def test_setitem_invalid_paths(double_nf):
    """Assignment below something that is not a nested column fails clearly."""
    with pytest.raises(ValueError, match="'outer.a' is not a nested column"):
        double_nf["outer.a.x"] = 1
    with pytest.raises(ValueError, match="Only one level of nesting"):
        double_nf["new.nested.x"] = 1


def test_drop_deep(double_nf):
    """Sub-columns can be dropped at several depths at once."""
    dropped = double_nf.drop(columns=["outer.inner.t", "outer.b"])
    assert dropped["outer"].columns == ["a", "inner"]
    assert dropped["outer.inner"].columns == ["flux", "flux_error", "band"]
    assert double_nf["outer"].columns == ["a", "b", "inner"]

    nf = double_nf.copy()
    nf.drop(columns=["outer.inner.flux"], inplace=True)
    assert "flux" not in nf["outer.inner"].columns


def test_query_deep(double_nf):
    """A query on a doubly nested sub-column filters the rows of the inner nested column."""
    t = double_nf["outer.inner.t"]
    result = double_nf.query("outer.inner.t > 10")
    np.testing.assert_array_equal(result["outer.inner.t"].to_numpy(), t[t > 10].to_numpy())
    np.testing.assert_array_equal(result["outer.a"].to_numpy(), double_nf["outer.a"].to_numpy())
    assert result.index.equals(double_nf.index)

    # Inner nested values that lose all of their rows become missing
    result = double_nf.query("outer.inner.t > 17")
    np.testing.assert_array_equal(result["outer.inner.t"].to_numpy(), t[t > 17].to_numpy())
    assert result["outer.inner"].isna().tolist() == [True, False, True, False, False]

    with pytest.raises(ValueError, match="multiple structs/layers"):
        double_nf.query("outer.inner.t > 10 and outer.a > 0")


def test_eval_deep(double_nf):
    """eval reads and assigns doubly nested sub-columns."""
    t = double_nf["outer.inner.t"].to_numpy()
    np.testing.assert_array_equal(double_nf.eval("outer.inner.t * 2").to_numpy(), t * 2)
    result = double_nf.eval("outer.inner.t3 = outer.inner.t + 1")
    np.testing.assert_array_equal(result["outer.inner.t3"].to_numpy(), t + 1)


def test_dropna_deep(double_nf):
    """dropna targets the inner nested column, by subset or by on_nested."""
    t = double_nf["outer.inner.t"].to_numpy().copy()
    t[[1, 5]] = np.nan
    nf = double_nf.copy()
    nf["outer.inner.t"] = t

    for result in (nf.dropna(subset="outer.inner.t"), nf.dropna(on_nested="outer.inner")):
        np.testing.assert_array_equal(result["outer.inner.t"].to_numpy(), t[~np.isnan(t)])
        assert len(result["outer.a"]) == len(double_nf["outer.a"])

    with pytest.raises(ValueError, match="not found in nested dataframes"):
        nf.dropna(on_nested="outer.a")
    with pytest.raises(ValueError, match="not found in the nested column 'outer'"):
        nf.dropna(subset="outer.a.t")


def test_sort_values_deep(double_nf):
    """sort_values sorts within each value of the inner nested column."""
    result = double_nf.sort_values("outer.inner.t", ascending=False)
    expected = np.sort(double_nf["outer.inner.t"].to_numpy().reshape(5, 3), axis=1)[:, ::-1]
    np.testing.assert_array_equal(result["outer.inner.t"].to_numpy().reshape(5, 3), expected)
    np.testing.assert_array_equal(result["outer.a"].to_numpy(), double_nf["outer.a"].to_numpy())


def test_deep_paths_not_supported(double_nf):
    """Methods without a meaning for deeper sub-columns say so."""
    with pytest.raises(NotImplementedError, match="deeper than one level"):
        double_nf.map_rows(lambda t: 1, columns=["outer.inner.t"])
    with pytest.raises(NotImplementedError, match="deeper than one level"):
        double_nf.fillna({"outer.inner.t": 0.0})


def test_triple_nesting(double_nf):
    """Paths work the same one level deeper."""
    t = double_nf["outer.inner.t"].to_numpy()
    nf = double_nf.reset_index()
    nf["group"] = 0
    triple_nf = NestedFrame.from_flat(nf, base_columns=[], on="group", name="top")

    np.testing.assert_array_equal(triple_nf["top.outer.inner.t"].to_numpy(), t)
    triple_nf["top.outer.inner.t"] = triple_nf["top.outer.inner.t"] + 1.0
    np.testing.assert_array_equal(triple_nf["top.outer.inner.t"].to_numpy(), t + 1.0)
    result = triple_nf.query("top.outer.inner.t > 10")
    np.testing.assert_array_equal(result["top.outer.inner.t"].to_numpy(), (t + 1.0)[t + 1.0 > 10])
