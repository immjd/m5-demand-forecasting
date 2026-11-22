"""The final-test script must refuse to run unless confirmed, and only once."""

import pytest

from scripts.final_test import CANDIDATES, check_guard


def test_guard_requires_confirmation(tmp_path) -> None:
    with pytest.raises(SystemExit, match="confirm-final-test"):
        check_guard(False, tmp_path / "test_results.csv", None)


def test_guard_blocks_second_run(tmp_path) -> None:
    out = tmp_path / "test_results.csv"
    check_guard(True, out, None)  # first run is allowed
    out.write_text("done")
    with pytest.raises(SystemExit, match="already exists"):
        check_guard(True, out, None)
    check_guard(True, out, "bug in scoring")  # logged re-run is allowed


def test_candidates_locked() -> None:
    models = {m for m, _ in CANDIDATES}
    assert models == {"seasonal_naive", "moving_average", "sarimax", "lgbm_tuned"}
