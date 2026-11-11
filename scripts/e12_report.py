"""E12 (scripts/e12_generic_halves.py, e12_patch.sh): accuracy files in the layout of E11, and the
no-generic sets with per-half generic heads next to E11's (generic heads from all items).

- ``results/e12_generic_halves/accuracy_alpha.csv``: ``fvuniv20_nogen_half`` patched head by head
  (accuracy = mean over the five rule means), as ``scripts/e11_report.py``;
- ``results/e12_generic_halves/fv_residual_alpha.csv``: ``fv30_nogen_half``, Davidson et al.'s
  intervention (one task mean);
- ``results/e12_generic_halves/summary.csv``: per set, mean over the 14 tasks at the cross-fitted
  alpha over {1, 2} (fold 1 at the better alpha of fold 2 and the reverse, ties to alpha 1; the
  control at the task's alpha), check 1 per fold and both checks over folds.

    .venv/bin/python scripts/e12_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import heldout_report as hr  # noqa: E402
from lth import RESULTS  # noqa: E402

OUT = RESULTS / "e12_generic_halves"
E11 = RESULTS / "e11_complete"
KEYS = ["fold", "family", "task", "set", "source", "alpha", "accuracy", "n_heads"]


def fv_rows(folder: Path) -> pd.DataFrame:
    rows = [pd.read_csv(p).assign(fold=f) for f in (1, 2) for p in sorted((folder / f"fold{f}").glob("fv_residual*.csv"))]
    fv = pd.concat(rows, ignore_index=True)
    fv = fv[fv.source != "none"][KEYS]
    assert not fv.duplicated(["fold", "family", "task", "set", "source", "alpha"]).any()
    return fv


def crossfit(d: pd.DataFrame) -> pd.DataFrame:
    """Each (fold, family, task, set, source) at the alpha best for the task means on the other fold."""
    t = d[d.source == "task"].sort_values(["accuracy", "alpha"], ascending=[False, True])
    best = t.drop_duplicates(["fold", "family", "task", "set"])[["fold", "family", "task", "set", "alpha"]]
    best = best.assign(fold=3 - best.fold)
    return d.merge(best, on=["fold", "family", "task", "set", "alpha"])


def main() -> None:
    hr.HELD = OUT
    curves, _ = hr.load()
    curves.to_csv(OUT / "accuracy_alpha.csv", index=False)
    fv = fv_rows(OUT)
    fv.to_csv(OUT / "fv_residual_alpha.csv", index=False)

    e11 = pd.concat([pd.read_csv(E11 / "fv_residual_alpha.csv"), pd.read_csv(E11 / "accuracy_alpha.csv")])
    e11 = e11[e11.set.isin(["fv30_nogen", "fvuniv20_nogen"])]
    d = pd.concat([fv, curves[KEYS], e11[KEYS]], ignore_index=True)
    assert d.groupby(["set", "family", "source", "fold"]).task.nunique().eq(14).all()
    cf = crossfit(d)
    rows = []
    for s, g in cf.groupby("set"):
        per = g.groupby(["family", "source", "task"]).accuracy.mean()  # over folds
        c1 = g[(g.family == "simple") & (g.source == "task")]
        rows.append(dict(
            set=s, heads_fold1=int(c1[c1.fold == 1].n_heads.iloc[0]), heads_fold2=int(c1[c1.fold == 2].n_heads.iloc[0]),
            check1_fold1=c1[c1.fold == 1].accuracy.mean(), check1_fold2=c1[c1.fold == 2].accuracy.mean(),
            check1_task=per["simple", "task"].mean(), check1_control=per["simple", "control"].mean(),
            check2_task=per["simple-wrong", "task"].mean(), check2_control=per["simple-wrong", "control"].mean(),
            alpha2_share=(c1.alpha == 2).mean()))
    s = pd.DataFrame(rows).set_index("set").loc[["fv30_nogen", "fv30_nogen_half", "fvuniv20_nogen", "fvuniv20_nogen_half"]]
    s.to_csv(OUT / "summary.csv")
    print(s.round(3).to_string())
    per_task = (cf[(cf.family == "simple") & (cf.source == "task")].groupby(["task", "set"]).accuracy.mean()
                .unstack("set")[s.index])
    per_task.round(3).to_csv(OUT / "check1_per_task.csv")
    print(per_task.round(2).to_string())


if __name__ == "__main__":
    main()
