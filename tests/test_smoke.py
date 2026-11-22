"""Smoke tests: the package imports and the config is internally consistent."""

import forecasting
from forecasting.config import PROJECT_ROOT, load_config


def test_package_imports() -> None:
    assert forecasting.__version__


def test_config_has_required_sections() -> None:
    cfg = load_config()
    for key in ("seed", "paths", "scope", "forecast", "backtest", "inventory"):
        assert key in cfg


def test_min_lag_covers_horizon() -> None:
    cfg = load_config()
    assert cfg["forecast"]["min_lag"] >= cfg["forecast"]["horizon"]


def test_folds_are_ordered_and_precede_test() -> None:
    cfg = load_config()
    horizon = cfg["forecast"]["horizon"]
    folds = cfg["backtest"]["folds"]
    test_train_end, test_start, _ = cfg["backtest"]["test"]

    for train_end, valid_start, valid_end in folds:
        assert train_end < valid_start
        assert valid_end - valid_start + 1 == horizon
        assert valid_end < test_start

    for (_, _, prev_end), (_, next_start, _) in zip(folds, folds[1:], strict=False):
        assert prev_end < next_start
    assert test_train_end < test_start


def test_project_root_points_to_repo() -> None:
    assert (PROJECT_ROOT / "configs" / "config.yaml").exists()
