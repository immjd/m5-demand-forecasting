"""Tests for the global LightGBM model and bottom-up aggregation."""

import numpy as np
import pandas as pd
from test_features_leakage import _config, _features, _sales

from forecasting.data.aggregate import bottom_up
from forecasting.features.build import feature_columns
from forecasting.models.lgbm import LGBMModel

CUTOFF, H = 130, 28


def _model(**kw) -> LGBMModel:
    cfg = _config()
    params = {**cfg["lgbm"]["params"], "num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 1}
    defaults = dict(
        features=feature_columns(cfg),
        params=params,
        objective="tweedie",
        num_boost_round=60,
        early_stopping_rounds=10,
        inner_valid_days=14,
        seed=42,
    )
    return LGBMModel(**{**defaults, **kw})


def _split():
    feats = _features(_sales(seed=3))
    train = feats[feats["d"] <= CUTOFF]
    future = feats[feats["d"].between(CUTOFF + 1, CUTOFF + H)].drop(columns="sales")
    return train, future


def test_inner_validation_is_last_days_of_train_fold() -> None:
    train, future = _split()
    m = _model().fit(train)
    assert m.inner_valid_range_ == (CUTOFF - 14 + 1, CUTOFF)
    assert m.max_train_d_ == CUTOFF  # never trained past the cutoff
    assert 1 <= m.best_iteration_ <= 60


def test_predict_shape_and_positive() -> None:
    train, future = _split()
    p = _model().fit(train).predict(future)
    assert len(p) == len(future)
    assert np.isfinite(p["y_pred"]).all()
    assert (p["y_pred"] > 0).all()  # tweedie predicts positive means


def test_deterministic_with_seed() -> None:
    train, future = _split()
    a = _model().fit(train).predict(future)["y_pred"].to_numpy()
    b = _model().fit(train).predict(future)["y_pred"].to_numpy()
    np.testing.assert_allclose(a, b)


def test_objectives_and_feature_drop() -> None:
    train, future = _split()
    cfg = _config()
    for obj in ("poisson", "regression"):
        p = _model(objective=obj).fit(train).predict(future)
        assert np.isfinite(p["y_pred"]).all()
    no_price = feature_columns(cfg, drop_groups=("price",))
    m = _model(features=no_price).fit(train)
    assert "sell_price" not in m.importance_.index


def test_bottom_up_sums_items() -> None:
    preds = pd.DataFrame(
        {
            "fold": "F1",
            "series_id": ["A", "B", "C"] * 2,
            "d": [1, 1, 1, 2, 2, 2],
            "y_pred": [1.0, 2.0, 4.0, 0.5, 0.5, 1.0],
        }
    )
    meta = pd.DataFrame(
        {
            "series_id": ["A", "B", "C"],
            "store_id": "CA_1",
            "cat_id": ["FOODS", "FOODS", "HOBBIES"],
            "dept_id": ["FOODS_1", "FOODS_2", "HOBBIES_1"],
        }
    )
    out = bottom_up(preds, meta).set_index(["series_id", "d"])["y_pred"]
    assert out[("CA_1", 1)] == 7.0
    assert out[("FOODS", 1)] == 3.0
    assert out[("HOBBIES_1", 2)] == 1.0
    assert len(out) == (1 + 2 + 3) * 2


def test_shap_values_add_up_to_raw_prediction() -> None:
    from forecasting.evaluation.explain import mean_abs_shap, shap_sample

    train, future = _split()
    m = _model().fit(train)
    vals, X, base = shap_sample(m, future.assign(is_closed=0), n_rows=50, seed=42)
    raw = m.booster_.predict(X, raw_score=True)
    np.testing.assert_allclose(base + vals.sum(axis=1).to_numpy(), raw, rtol=1e-4, atol=1e-4)
    assert mean_abs_shap(vals).index[0] in m.features
