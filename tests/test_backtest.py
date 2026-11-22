"""Tests for fold generation, level data loading and the backtest loop."""

from pathlib import Path

import pandas as pd
import pytest

from forecasting.config import load_config
from forecasting.models.base import ForecastModel
from forecasting.validation.backtest import (
    Fold,
    generate_folds,
    get_test_fold,
    load_level_data,
    run_backtest,
)

# ------------------------------------------------------------------ folds


def test_folds_follow_config_and_never_include_test() -> None:
    cfg = load_config()
    folds = generate_folds(cfg)
    test = get_test_fold(cfg)
    horizon = cfg["forecast"]["horizon"]

    assert [f.name for f in folds] == ["F1", "F2", "F3", "F4"]
    assert test.name == "TEST"
    assert all(f.name != "TEST" for f in folds)
    for f in folds:
        assert f.train_end < f.valid_start
        assert f.valid_end - f.valid_start + 1 == horizon
        assert f.valid_end < test.valid_start
    for prev, nxt in zip(folds, folds[1:], strict=False):
        assert prev.valid_end < nxt.valid_start  # validation windows never overlap


def test_fold_rejects_overlap() -> None:
    with pytest.raises(ValueError):
        Fold("bad", train_end=10, valid_start=10, valid_end=20)


# ------------------------------------------------------------------ level data


def _cfg(tmp_path: Path) -> dict:
    cfg = load_config()
    cfg["paths"] = {"interim": str(tmp_path), "processed": str(tmp_path)}
    cfg["backtest"] = {"folds": [[5, 6, 7], [7, 8, 9]], "test": [9, 10, 11]}
    cfg["forecast"] = {"horizon": 2, "min_lag": 2}
    return cfg


def test_load_level_data_excludes_test_period(tmp_path: Path) -> None:
    days = range(1, 12)
    pd.DataFrame(
        {
            "item_id": pd.Categorical(["A"] * 11),
            "d": list(days),
            "date": pd.date_range("2016-01-01", periods=11),
            "sales": list(days),
            "sell_price": 1.0,
            "is_closed": 0,
        }
    ).to_parquet(tmp_path / "sales_long.parquet")
    data = load_level_data(_cfg(tmp_path), level="item")
    assert data["d"].max() == 9  # last validation day, test days 10-11 are not loaded
    assert "series_id" in data.columns


# ------------------------------------------------------------------ backtest loop


class SpyModel(ForecastModel):
    """Records what it receives and predicts a constant (negative to test clipping)."""

    name = "spy"
    calls: list[dict] = []

    def fit(self, train: pd.DataFrame) -> "SpyModel":
        self._train_max_d = int(train["d"].max())
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        SpyModel.calls.append(
            {"train_max_d": self._train_max_d, "future_cols": set(future.columns)}
        )
        return future[["series_id", "d"]].assign(y_pred=-1.0)


def _toy_data() -> pd.DataFrame:
    rows = [
        {"series_id": s, "d": d, "sales": d * (1 if s == "A" else 10), "is_closed": 0}
        for s in ["A", "B"]
        for d in range(1, 10)
    ]
    return pd.DataFrame(rows)


def test_backtest_hides_target_and_respects_cutoff() -> None:
    SpyModel.calls = []
    folds = [Fold("F1", 5, 6, 7), Fold("F2", 7, 8, 9)]
    result = run_backtest(SpyModel, _toy_data(), folds)

    assert [c["train_max_d"] for c in SpyModel.calls] == [5, 7]
    assert all("sales" not in c["future_cols"] for c in SpyModel.calls)

    preds = result.preds
    assert set(preds["fold"]) == {"F1", "F2"}
    assert len(preds) == 2 * 2 * 2  # folds x series x horizon
    assert (preds["y_pred"] == 0).all()  # negative predictions clipped
    f1_a = preds[(preds["fold"] == "F1") & (preds["series_id"] == "A")]
    assert f1_a["y_true"].tolist() == [6, 7]
    assert set(result.runtime_sec) == {"F1", "F2"}


class IncompleteModel(SpyModel):
    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        return future[["series_id", "d"]].iloc[:1].assign(y_pred=1.0)


def test_backtest_rejects_missing_predictions() -> None:
    with pytest.raises(ValueError, match="missing"):
        run_backtest(IncompleteModel, _toy_data(), [Fold("F1", 5, 6, 7)])
