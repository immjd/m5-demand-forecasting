"""Prophet per series: weekly + yearly seasonality, M5 events as holidays, SNAP regressor."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forecasting.models.statistical import PerSeriesModel, _quiet

EVENT_COLS = ["event_name_1", "event_name_2"]


def holidays_frame(days: pd.DataFrame) -> pd.DataFrame | None:
    """Prophet holidays table (``holiday, ds``) from the calendar event columns.

    Events are scheduled in advance, so using future event dates is not leakage.
    """
    parts = []
    for col in EVENT_COLS:
        if col in days.columns:
            ev = days.loc[days[col].notna(), ["date", col]]
            parts.append(pd.DataFrame({"holiday": ev[col].astype(str), "ds": ev["date"]}))
    if not parts:
        return None
    hol = pd.concat(parts, ignore_index=True).drop_duplicates()
    return hol if len(hol) else None


def _prophet_worker(p: dict) -> np.ndarray:
    import logging

    import cmdstanpy  # noqa: F401 - imported first so its logger exists before we silence it
    from prophet import Prophet

    _quiet()  # prophet/cmdstanpy reset their loggers on import, so silence them afterwards
    logging.getLogger("cmdstanpy").disabled = True
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        holidays=p["holidays"],
        uncertainty_samples=0,  # point forecasts only -> much faster
    )
    if p["use_snap"]:
        model.add_regressor("snap")
    model.fit(p["train"])
    return model.predict(p["future"])["yhat"].to_numpy()


class ProphetModel(PerSeriesModel):
    """Prophet fitted per series on the last ``window`` days."""

    name = "prophet"
    worker = staticmethod(_prophet_worker)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _payload(self, sid: str, y: pd.Series, future_days: pd.DataFrame) -> dict:
        cal = self.train_days_.set_index("d")
        use_snap = "snap" in cal.columns and "snap" in future_days.columns
        train = pd.DataFrame({"ds": cal["date"].reindex(y.index).to_numpy(), "y": y.to_numpy()})
        if train["ds"].isna().any():  # day not in the shared calendar -> rebuild by offset
            last_date = cal["date"].iloc[-1]
            train["ds"] = last_date - pd.to_timedelta(y.index.max() - y.index, unit="D")
        future = pd.DataFrame({"ds": future_days["date"].to_numpy()})
        if use_snap:
            train["snap"] = cal["snap"].reindex(y.index).fillna(0).to_numpy()
            future["snap"] = future_days["snap"].to_numpy()
        days = pd.concat([self.train_days_, future_days], ignore_index=True)
        return {
            "train": train,
            "future": future,
            "holidays": holidays_frame(days),
            "use_snap": use_snap,
        }
