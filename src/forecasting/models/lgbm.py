"""Global LightGBM model: one model for all items, trained on lag/calendar/price features.

Early stopping uses the LAST ``inner_valid_days`` of the training fold as an
inner validation set (never the fold's own validation window). The model is
then optionally refitted on the whole training fold with the best number of
boosting rounds.
"""

from __future__ import annotations

import logging
from typing import Any, Self

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecasting.features.price import item_mean_price, relative_price
from forecasting.models.base import ForecastModel

log = logging.getLogger(__name__)
TARGET = "sales"


class LGBMModel(ForecastModel):
    """Global gradient-boosted trees on the feature table built by ``scripts.build_features``."""

    name = "lgbm"

    def __init__(
        self,
        features: list[str],
        params: dict[str, Any],
        objective: str = "tweedie",
        num_boost_round: int = 3000,
        early_stopping_rounds: int = 100,
        inner_valid_days: int = 28,
        refit_full: bool = True,
        train_window_days: int | None = None,
        seed: int = 42,
        name: str | None = None,
    ) -> None:
        self.features = features
        self.params = params
        self.objective = objective
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.inner_valid_days = inner_valid_days
        self.refit_full = refit_full
        self.train_window_days = train_window_days
        self.seed = seed
        if name:
            self.name = name

    # ------------------------------------------------------------------ helpers
    def _lgb_params(self) -> dict[str, Any]:
        params = {
            **self.params,
            "objective": self.objective,
            "metric": "rmse",  # same early-stopping metric for every objective (fair comparison)
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
        }
        if self.objective != "tweedie":
            params.pop("tweedie_variance_power", None)
        return params

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.reindex(columns=[f for f in self.features if f != "price_rel_item_mean"]).copy()
        if "price_rel_item_mean" in self.features:
            X["price_rel_item_mean"] = relative_price(df, self.mean_price_).to_numpy()
        return X[self.features]

    def feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Model input for ``df`` exactly as used in ``predict`` (needs a fitted model)."""
        return self._matrix(df)

    # ------------------------------------------------------------------ API
    def fit(self, train: pd.DataFrame) -> Self:
        cutoff = int(train["d"].max())
        self.mean_price_ = item_mean_price(train, cutoff)  # fold-dependent, train data only

        rows = train.loc[(train["is_closed"] == 0) & train["lag_28"].notna()]
        if self.train_window_days:
            rows = rows.loc[rows["d"] > cutoff - self.train_window_days]
        inner_start = cutoff - self.inner_valid_days + 1
        self.inner_valid_range_ = (inner_start, cutoff)
        tr, va = rows.loc[rows["d"] < inner_start], rows.loc[rows["d"] >= inner_start]

        params = self._lgb_params()
        d_tr = lgb.Dataset(self._matrix(tr), tr[TARGET].astype("float32"), free_raw_data=True)
        d_va = lgb.Dataset(self._matrix(va), va[TARGET].astype("float32"), reference=d_tr)
        booster = lgb.train(
            params,
            d_tr,
            num_boost_round=self.num_boost_round,
            valid_sets=[d_va],
            valid_names=["inner_valid"],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )
        self.best_iteration_ = int(booster.best_iteration or self.num_boost_round)
        self.inner_valid_rmse_ = float(booster.best_score["inner_valid"]["rmse"])

        if self.refit_full:
            d_all = lgb.Dataset(self._matrix(rows), rows[TARGET].astype("float32"))
            booster = lgb.train(params, d_all, num_boost_round=self.best_iteration_)
        self.booster_ = booster
        self.max_train_d_ = int(rows["d"].max())
        self.n_train_rows_ = len(rows)
        self.importance_ = pd.Series(
            booster.feature_importance("gain"), index=self.features, name="gain"
        )
        log.info(
            "%s: cutoff d_%d | %s rows | best_iter %d | inner-valid RMSE %.4f",
            self.name,
            cutoff,
            f"{len(rows):,}",
            self.best_iteration_,
            self.inner_valid_rmse_,
        )
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        out = future[["series_id", "d"]].copy()
        out["y_pred"] = np.asarray(self.booster_.predict(self._matrix(future)), dtype="float64")
        return out
