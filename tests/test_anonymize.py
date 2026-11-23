"""Public item-level examples carry no item name, date or raw units."""

import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from forecasting.config import PROJECT_ROOT
from forecasting.viz.anonymize import index_base, item_label, relative_days, to_index

ITEM_NAME = re.compile(r"(FOODS|HOBBIES|HOUSEHOLD)_\d_\d{3}")
NOTEBOOKS = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))


def test_item_label_is_generic() -> None:
    assert item_label(0) == "Item A"
    assert item_label(1, "lumpy") == "Item B (lumpy)"
    with pytest.raises(ValueError):
        item_label(26)


def test_index_is_relative_to_base() -> None:
    s = pd.Series([1.0, 2.0, 3.0])
    out = to_index(s, index_base(s))
    np.testing.assert_allclose(out, [50.0, 100.0, 150.0])
    with pytest.raises(ValueError):
        index_base(np.zeros(3))


def test_relative_days_hide_the_calendar() -> None:
    dates = pd.date_range("2016-01-04", periods=3)
    np.testing.assert_array_equal(relative_days(dates, "2016-01-05"), [-1, 0, 1])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebooks_show_no_item_names(path: Path) -> None:
    nb = nbformat.read(path, 4)
    for cell in nb.cells:
        texts = [cell.source]
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            texts += [out.get("text", ""), data.get("text/plain", ""), data.get("text/html", "")]
        assert not any(ITEM_NAME.search(t) for t in texts), f"item name in {path.name}"
