import numpy as np
import pandas as pd
import pytest

from nested_pandas import NestedFrame
from nested_pandas.display import MAX_RENDERED
from nested_pandas.tensors import TensorExtensionArray
from nested_pandas.tensors.display import tensor_cell_html, tensor_column_formatter

pytest.importorskip("matplotlib")


def test_tensor_cell_html_small_tensor_shows_values():
    """Test that tensors up to TENSOR_FORMATTING_MAX_ELEMENTS elements show their values, one line per
    row, in a monospace block."""
    html = tensor_cell_html(np.array([[0.0, 1.5], [2.0, 3.0]]))
    assert html.startswith("<pre") and html.endswith("</pre>")
    assert "[[0.  1.5]\n [2.  3. ]]" in html
    assert "<img" not in html
    assert "[0 1 2]" in tensor_cell_html(np.arange(3))
    assert tensor_cell_html(np.zeros((2, 2, 3))).count("\n") > 1  # 12 elements, still small


def test_tensor_cell_html_renders_thumbnail_with_colorbar():
    """Test that larger 2-d tensor cells render as a colormapped thumbnail plus a labelled colorbar."""
    cell = np.linspace(0.0, 30.0, 16, dtype=np.float32).reshape(4, 4)
    html = tensor_cell_html(cell)
    assert html.count("<img src=") == 2  # thumbnail + colorbar strip
    assert 'title="colorbar"' in html
    assert "[4×4] float32" in html
    # colorbar labels are the displayed (1st-99th percentile) range, top then bottom
    assert html.index("29.7") < html.index("0.3")


def test_tensor_cell_html_non_2d_and_missing():
    """Test that missing cells and larger tensors that are not 2-d fall back to a descriptor."""
    assert tensor_cell_html(pd.NA) == "&lt;NA&gt;"
    assert tensor_cell_html(None) == "&lt;NA&gt;"
    assert tensor_cell_html(np.zeros((2, 3, 3), dtype=np.float32)) == "[2×3×3] float32"
    assert tensor_cell_html(np.zeros(20, dtype=np.float32)) == "[20] float32"


def test_tensor_cell_html_bool_tensor():
    """Test that boolean tensors render too, cast to float for the colormap."""
    html = tensor_cell_html(np.eye(4, dtype=bool))
    assert 'title="colorbar"' in html
    assert "[4×4] bool" in html


def test_nestedframe_repr_html_uses_tensor_formatter():
    """Test that the NestedFrame HTML repr renders every displayed tensor cell with a colorbar."""
    array = TensorExtensionArray.from_sequence([np.zeros((4, 4), dtype=np.float32)] * 3 + [None])
    nf = NestedFrame({"t": array, "x": [1, 2, 3, 4]})
    html = nf._repr_html_()
    assert html.count('title="colorbar"') == 3
    assert "&lt;NA&gt;" in html
    assert "4 rows x 2 columns" in html


def test_nestedframe_repr_html_truncates_tensor_rows():
    """Test that pandas' row truncation bounds how many tensor cells are rendered."""
    array = TensorExtensionArray.from_stack(np.zeros((30, 4, 4), dtype=np.float32))
    nf = NestedFrame({"t": array})
    with pd.option_context("display.max_rows", 10, "display.min_rows", 4):
        html = nf._repr_html_()
    assert html.count('title="colorbar"') < 30
    assert "30 rows x 1 columns" in html


def test_tensor_column_formatter_caps_thumbnails():
    """Test that a column formatter renders MAX_RENDERED thumbnails and placeholders after, without
    counting cells that show values or a descriptor."""
    formatter = tensor_column_formatter()
    big = np.zeros((4, 4), dtype=np.float32)
    outputs = [formatter(cell) for cell in [np.zeros((2, 2))] + [big] * (MAX_RENDERED + 2) + [None]]
    assert outputs[0].startswith("<pre")  # values, not counted
    assert all('title="colorbar"' in html for html in outputs[1 : MAX_RENDERED + 1])
    assert all("not rendered in preview" in html for html in outputs[MAX_RENDERED + 1 : -1])
    assert outputs[-1] == "&lt;NA&gt;"
    assert "[2×3×3] float32" in formatter(np.zeros((2, 3, 3), dtype=np.float32))
    assert "not rendered in preview" in tensor_column_formatter(max_rendered=0)(big)


def test_nestedframe_repr_html_caps_thumbnails_per_column():
    """Test that each tensor column of a NestedFrame repr renders MAX_RENDERED thumbnails, then
    placeholders, independently of the other columns."""
    n = MAX_RENDERED + 3
    array = TensorExtensionArray.from_stack(np.zeros((n, 4, 4), dtype=np.float32))
    nf = NestedFrame({"t": array, "u": array.copy()})
    html = nf._repr_html_()
    assert html.count('title="colorbar"') == 2 * MAX_RENDERED
    assert html.count("not rendered in preview") == 2 * 3
    assert f"{n} rows x 2 columns" in html
