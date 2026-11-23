"""Every dashboard page runs with only the packages of ``app/requirements.txt``."""

import subprocess
import sys

from forecasting.config import PROJECT_ROOT

# Runs in a fresh interpreter: every third-party package outside the dependency closure of
# app/requirements.txt is made un-importable, as on Streamlit Community Cloud.
SCRIPT = r"""
import importlib.abc, re, sys
from importlib import metadata
from pathlib import Path
from packaging.requirements import Requirement

norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
lines = Path("app/requirements.txt").read_text().split()
todo = [norm(Requirement(line).name) for line in lines]
allowed = set()
while todo:
    name = todo.pop()
    if name in allowed:
        continue
    allowed.add(name)
    try:
        reqs = metadata.requires(name) or []
    except metadata.PackageNotFoundError:
        continue
    for r in map(Requirement, reqs):
        if r.marker is None or r.marker.evaluate({"extra": ""}):
            todo.append(norm(r.name))
owners = metadata.packages_distributions()
blocked = {m for m, d in owners.items() if not {norm(x) for x in d} & allowed}
blocked -= {"forecasting", "packaging", "pkg_resources", "setuptools"}
assert {"yaml", "lightgbm", "statsmodels"} <= blocked

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError(f"blocked: {fullname}")

sys.meta_path.insert(0, Block())
from streamlit.testing.v1 import AppTest

pages = ["overview", "model_comparison", "forecast_explorer", "inventory_planner", "methodology"]
for page in pages:
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60).run()
    if page != "overview":
        at.switch_page(f"pages/{page}.py").run()
    assert not at.exception, (page, [e.value for e in at.exception])
    assert at.title, page
print("pages ok")
"""


def test_all_pages_run_with_deploy_dependencies_only() -> None:
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert "pages ok" in out.stdout, out.stderr[-2000:]


def test_app_requirements_match_the_project_pins() -> None:
    def pins(path):
        return set((PROJECT_ROOT / path).read_text(encoding="utf-8").split())

    assert pins("app/requirements.txt") <= pins("requirements.txt")
