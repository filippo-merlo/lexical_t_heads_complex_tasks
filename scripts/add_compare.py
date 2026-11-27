"""Add vs replace at 40 heads (``scripts/rebuild_add.sh``): check-1 accuracy (alpha cross-fitted) of LRP-T and
Selected, per task and shared, with alpha * mean added to each head's own output (``add`` stage) next to the same
sets with it replacing the output (``lrp`` and ``s2`` stages), over the 14 tasks and both folds; then per task.
No GPU.

Writes ``results/rebuild/additions/add_compare.csv`` and prints it.

    .venv/bin/python scripts/add_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import RESULTS  # noqa: E402

DIR = RESULTS / "rebuild" / "additions"
SETS = {"lrpT40": "LRP-T", "selected40": "Selected", "shared_lrpT40": "LRP-T", "shared_selected40": "Selected"}
REPLACE = {"lrpT40": "lrp", "shared_lrpT40": "lrp", "selected40": "s2", "shared_selected40": "s2"}


def numeric(d: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:  # unfillable sets elsewhere are written as "--"
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def main() -> None:
    add = numeric(pd.read_csv(DIR / "add_summary.csv"), ("task", "control", "random")).set_index("set")
    rep = {s: numeric(pd.read_csv(DIR / f"{s}_summary.csv"), ("task", "control", "random")).set_index("set")
           for s in ("lrp", "s2")}
    rows = []
    for name, label in SETS.items():
        for form, r in (("replace", rep[REPLACE[name]].loc[name]), ("add", add.loc[name])):
            assert r.tasks_folds == 28, (name, form)
            rows.append(dict(scope=r.scope, rule=label, form=form, task=r.task, control=r.control, random=r.random))
    df = pd.DataFrame(rows)
    df.to_csv(DIR / "add_compare.csv", index=False)
    print("== 40 heads: accuracy (control mean / random sets), alpha cross-fitted, 14 tasks x 2 folds")
    print(f"{'scope':10s}{'rule':10s}{'replace':>22s}{'add':>22s}")
    for (scope, rule), d in df.groupby(["scope", "rule"], sort=False):
        cell = lambda f: "{:.2f} ({:.2f}/{:.2f})".format(*d[d.form == f][["task", "control", "random"]].iloc[0])  # noqa: E731
        print(f"{scope:10s}{rule:10s}{cell('replace'):>22s}{cell('add'):>22s}")

    per = {s: numeric(pd.read_csv(DIR / f"{s}_per_task.csv"), ("accuracy",)) for s in ("add", "lrp", "s2")}
    out = {}
    for name in SETS:
        for form, d in (("replace", per[REPLACE[name]]), ("add", per["add"])):
            d = d[(d.set == name) & (d.source == "task")]
            out[(name, form)] = d.groupby("task").accuracy.mean()
            out[(name, form, "alpha")] = d.groupby("task").alpha.agg(lambda a: "/".join(f"{x:g}" for x in a))
    print("\n== per task, 40 heads, mean of the two folds (cross-fitted alpha per fold: fold 1/fold 2)")
    tab = pd.DataFrame({f"{n} {f}": out[(n, f)] for n in SETS for f in ("replace", "add")})
    alphas = pd.DataFrame({f"{n} add alpha": out[(n, "add", "alpha")] for n in SETS})
    print(pd.concat([tab.round(2), alphas], axis=1).to_string())


if __name__ == "__main__":
    main()
