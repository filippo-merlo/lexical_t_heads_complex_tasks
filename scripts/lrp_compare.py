"""T and LRP-T next to the rules they are compared with: check-1 accuracy (alpha cross-fitted) over the 14 tasks and
both folds, with the control mean and the random sets, per scope and size. No GPU.

Reads ``results/rebuild/additions/{lrp,s1,s2,zs}_summary.csv`` (``scripts/rebuild_report.py``) and, for the shared
Selected at the sizes with their own screen, ``results/rebuild/shared_n<N>/sel_summary.csv`` (as
``rebuild_figures``). Writes ``results/rebuild/additions/lrp_compare.csv`` and prints it.

    .venv/bin/python scripts/lrp_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import RESULTS  # noqa: E402

DIR = RESULTS / "rebuild" / "additions"
SIZES = (10, 20, 40, 60, 80, 100, 120)
RULES = {"T": "T", "lrpT": "LRP-T", "selected": "Selected", "causal": "Causal", "causalzs": "Causal (zero-shot)"}


def main() -> None:
    summ = pd.concat([pd.read_csv(DIR / f"{s}_summary.csv") for s in ("s1", "s2", "zs", "lrp")], ignore_index=True)
    for n in SIZES:
        sized = RESULTS / "rebuild" / f"shared_n{n}" / "sel_summary.csv"
        if sized.exists():
            new = pd.read_csv(sized)
            summ = pd.concat([summ[summ.set != f"shared_selected{n}"], new[new.set == f"shared_selected{n}"]])
    summ = summ.set_index("set")
    for c in ("task", "control", "random"):  # unfillable sets elsewhere are written as "--"
        summ[c] = pd.to_numeric(summ[c], errors="coerce")
    rows = []
    for scope, prefix in (("per task", ""), ("shared", "shared_")):
        for n in SIZES:
            for rule, label in RULES.items():
                name = f"{prefix}{rule}{n}"
                if name in summ.index:
                    r = summ.loc[name]
                    assert r.tasks_folds == 28, name
                    rows.append(dict(scope=scope, n=n, rule=label, task=r.task, control=r.control, random=r.random))
    df = pd.DataFrame(rows)
    df.to_csv(DIR / "lrp_compare.csv", index=False)
    for scope in ("per task", "shared"):
        d = df[df.scope == scope]
        print(f"\n== {scope}: accuracy (control mean / random sets)")
        print(f"{'N':>4} " + "".join(f"{label:>22}" for label in RULES.values()))
        for n in SIZES:
            cells = []
            for label in RULES.values():
                x = d[(d.n == n) & (d.rule == label)]
                cells.append(f"{x.task.iloc[0]:.2f} ({x.control.iloc[0]:.2f}/{x.random.iloc[0]:.2f})" if len(x) else "--")
            print(f"{n:>4} " + "".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()
