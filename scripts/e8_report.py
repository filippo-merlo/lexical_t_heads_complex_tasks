"""E8 (scripts/e8_sets.py, e8_patch.sh): accuracies of the 31 filter-combination sets, E4 protocol,
alphas 1 and 2, no controls, no statistical tests.

As in ``scripts/e7_report.py``: task accuracy = mean over the five rule means of the choosing
half; a set's accuracy at its cross-fitted alpha (fold 1 at the best alpha of fold 2 and the
reverse; ties: smaller |alpha|, then the positive one), mean of the two folds.

Filter choice without the reported items: the set with the highest check-1 task-mean accuracy on
fold 1 (scores from half A, scored on half B, each task at its own best alpha) is picked, and its
fold-2 accuracy (scores from B, scored on A, at fold 1's best alpha) is the reported number.

References on the same items and folds, cross-fitted over alpha 1 and 2 only (their own runs had
ten alphas): E7 ``raw_att`` (= the first run's ``DB``: 10-word list, Yang's expansion only) and E4's
CC-20, FV-20, CC and FV-N_t.

Writes ``results/e8_filters/accuracy_alpha.csv``, ``accuracy_crossfit.csv``, ``report.md``.

    .venv/bin/python scripts/e8_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import heldout_report as hr  # noqa: E402
from e7_report import md  # noqa: E402
from e8_sets import COMBOS  # noqa: E402
from lth import RESULTS, simple  # noqa: E402

OUT = RESULTS / "e8_filters"
ALPHAS = (1.0, 2.0)
CHECKS = {"simple": "check 1", "simple-wrong": "check 2"}
REFS = {"E7 raw_att": (RESULTS / "e7_attfilter", "raw_att"), "E4 cc20": (RESULTS / "heldout", "cc20"),
        "E4 fv20": (RESULTS / "heldout", "fv20"), "E4 cc": (RESULTS / "heldout", "cc"),
        "E4 fvNt": (RESULTS / "heldout", "fvNt")}


def crossfit(c: pd.DataFrame) -> pd.DataFrame:
    best = hr.best_alpha(c)
    other = best.assign(fold=3 - best.fold)
    return other.merge(c, on=["fold", "family", "task", "set", "alpha"], how="left")


def refs() -> pd.DataFrame:
    out = []
    for name, (root, key) in REFS.items():
        c = pd.read_csv(root / "accuracy_alpha.csv")
        c = c[(c.set == key) & c.alpha.isin(ALPHAS)]
        out.append(crossfit(c).assign(set=name))
    return pd.concat(out)[["fold", "family", "task", "set", "source", "accuracy"]]


def main() -> None:
    hr.HELD = OUT
    curves, base = hr.load()
    curves.to_csv(OUT / "accuracy_alpha.csv", index=False)
    cf = crossfit(curves)
    cf.to_csv(OUT / "accuracy_crossfit.csv", index=False)
    perfold = pd.concat([cf[["fold", "family", "task", "set", "source", "accuracy"]], refs()])
    s = perfold.groupby(["family", "task", "set", "source"], as_index=False).accuracy.mean()
    cols = ["unpatched", *COMBOS, *REFS]
    sizes = curves.groupby(["task", "set", "fold"]).n_heads.first().groupby(["task", "set"]).mean().unstack()

    lines = ["# E8: every combination of five head filters (D decoding, B BOS, T task words, I input, R rest)",
             "", "Accuracy at the cross-fitted alpha (1 or 2), mean of the two folds. A set is named by the "
             "filters a head must pass. D uses each task's 30 most active decoded words, matched with and "
             "without a leading space in lower, capitalised and upper case. References: earlier sets on the "
             "same items, cross-fitted over alpha 1 and 2 only (E7 raw_att = DB with E7's 10-word list and "
             "no leading-space capitalised forms). Empty sets are left out for that task (blank cell).", ""]
    for family, check in CHECKS.items():
        for source in ("task", "control"):
            g = s[(s.family == family) & (s.source == source)].pivot_table(index="task", columns="set", values="accuracy")
            g = g.reindex(list(simple.TASKS))
            if source == "task":
                g["unpatched"] = base[base.family == family].groupby("task").accuracy.mean()
            g.loc["mean"] = g.mean()
            g = g[[c for c in cols if c in g.columns]]
            lines += [f"## {check}, {source} mean", "", md(g.T), ""]

    over = pd.DataFrame({f"{CHECKS[f]} {src}": s[(s.family == f) & (s.source == src)].groupby("set").accuracy.mean()
                         for f in CHECKS for src in ("task", "control")})
    over["heads (median over tasks)"] = sizes.median()
    over["tasks with a set"] = sizes.notna().sum()
    over = over.reindex([c for c in [*COMBOS, *REFS] if c in over.index]).sort_values("check 1 task", ascending=False)
    lines += ["## Overview: mean over tasks, sorted by check 1", "", md(over), ""]

    own = hr.best_alpha(curves).merge(curves, on=["fold", "family", "task", "set", "alpha"])
    own = own[(own.fold == 1) & (own.family == "simple") & (own.source == "task")]
    pick = own.groupby("set").accuracy.mean().sort_values(ascending=False)
    f2 = perfold[(perfold.fold == 2) & (perfold.family == "simple") & (perfold.source == "task")].groupby("set").accuracy.mean()
    sel = pd.DataFrame({"fold 1, own best alpha": pick, "fold 2, cross-fitted": f2.reindex(pick.index)}).head(10)
    lines += ["## Filter choice on fold 1, read on fold 2 (check 1, task mean, mean over tasks; top 10)", "", md(sel), "",
              f"Picked: **{pick.index[0]}**, fold 2 = {f2[pick.index[0]]:.2f}. References on fold 2: "
              + ", ".join(f"{r} {f2[r]:.2f}" for r in REFS if r in f2), ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
