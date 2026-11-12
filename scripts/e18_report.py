"""E18 report: check-1 accuracy of E13's list-based filter sets with 20-, 30- and 40-word lists.

30 words = E13's own results. 20 and 40: a set equal to E13's set of the same name (same fold and
task, ``sets.csv``) takes E13's curves; the others come from ``results/e18_list_length/w{n}``. Every
set is then scored as in ``e13_report.py``: cross-fitted alpha over {1, 2}, task accuracy = mean of
the five rule means, mean of the two folds, mean of the 14 tasks.

Check: the 30-word rows equal E13's ``summary.csv``.

Writes ``results/e18_list_length/{per_task,summary}.csv`` and ``report.md``.

    .venv/bin/python scripts/e18_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e13_report as er  # noqa: E402
import heldout_report as hr  # noqa: E402
from lth import RESULTS, simple  # noqa: E402

OUT = RESULTS / "e18_list_length"
E13 = RESULTS / "e13_capped"
LENGTHS = (20, 30, 40)
SIZES = (20, 40, 60, 80)
CONDITIONS = ("D", "T", "DB", "DT", "BT", "DBT", "DBT_nogen")
LABEL = {**{c: c for c in CONDITIONS}, "DBT_nogen": "DBT − generic"}
KEYS = ["fold", "task", "set"]


def curves(folder: Path) -> pd.DataFrame:
    hr.HELD = folder
    c, _ = hr.load()
    c = c[c.family == "simple"]
    meta = pd.DataFrame([er.parse(s) for s in c.set], columns=["scope", "condition", "n"], index=c.index)
    c = pd.concat([c, meta], axis=1)
    return c[c.condition.isin(CONDITIONS) & c.n.isin(SIZES)]


def main() -> None:
    e13 = curves(E13)
    parts = [e13.assign(length=30)]
    for n in (20, 40):
        sets = pd.concat([pd.read_csv(OUT / f"w{n}" / f"fold{f}" / "sets.csv") for f in (1, 2)], ignore_index=True)
        equal = sets[sets.equal_e13 == 1][KEYS]
        reused = e13.merge(equal, on=KEYS)
        new = curves(OUT / f"w{n}")
        assert new.merge(equal, on=KEYS).empty, f"w{n}: a set equal to E13's was patched"
        c = pd.concat([reused, new], ignore_index=True)
        have = c[c.source == "task"][KEYS].drop_duplicates()
        assert len(have) == len(sets) and len(have.merge(sets[KEYS], on=KEYS)) == len(sets), f"w{n}: sets missing"
        parts.append(c.assign(length=n))
    cf = []
    for p in parts:
        cf.append(er.crossfit(p.drop(columns="length")).assign(length=int(p.length.iloc[0])))
    cf = pd.concat(cf, ignore_index=True)
    per = (cf.groupby(["length", "task", "set", "scope", "condition", "n", "source"], as_index=False)
           .agg(accuracy=("accuracy", "mean"), n_heads=("n_heads", "mean"), folds=("fold", "nunique")))
    task = per[per.source == "task"]
    assert (task.folds == 2).all()
    summary = (task.groupby(["length", "scope", "condition", "n", "set"], as_index=False)
               .agg(check1_task=("accuracy", "mean"), tasks=("task", "nunique")))
    ctrl = (per[per.source == "control"].groupby(["length", "set"], as_index=False)
            .agg(check1_control=("accuracy", "mean")))
    summary = summary.merge(ctrl, on=["length", "set"], how="left")
    assert (summary.tasks == 14).all()

    ref = pd.read_csv(E13 / "summary.csv").set_index("set").check1_task
    mine = summary[summary.length == 30].set_index("set").check1_task
    assert np.allclose(mine.values, ref.loc[mine.index].values, atol=1e-12), "30 words do not reproduce E13's summary"

    per.to_csv(OUT / "per_task.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    write_report(summary, per)
    print(f"30-word rows reproduce E13 ({len(mine)} sets); wrote summary.csv, per_task.csv, report.md")


def write_report(summary: pd.DataFrame, per: pd.DataFrame) -> None:
    lines = ["# E18: length of the task-word list (check 1)", "",
             "E13's rules; only the list behind D and T has 20, 30 (E13) or 40 words. Cross-fitted alpha over "
             "{1, 2}, mean of 14 tasks and both folds.", ""]
    for scope, title in (("task", "Per task"), ("shared", "Shared")):
        s = summary[summary.scope == scope]
        head = "| Set | " + " | ".join(f"N={n} 20w | 30w | 40w" for n in SIZES) + " |"
        lines += [f"## {title}", "", head, "|" + "---|" * (1 + 3 * len(SIZES))]
        for c in CONDITIONS:
            cells = []
            for n in SIZES:
                for L in LENGTHS:
                    v = s[(s.condition == c) & (s.n == n) & (s.length == L)].check1_task
                    cells.append(f"{v.iloc[0]:.2f}" if len(v) else "--")
            lines.append(f"| {LABEL[c]} | " + " | ".join(cells) + " |")
        lines.append("")
    # Per task, per-task DBT and T at N = 40.
    lines += ["## Per task, N = 40 (per-task sets)", "", "| Task | " +
              " | ".join(f"{c} {L}w" for c in ("D", "T", "DBT") for L in LENGTHS) + " |",
              "|" + "---|" * (1 + 3 * len(LENGTHS))]
    p = per[(per.source == "task") & (per.scope == "task") & (per.n == 40)]
    for t in er.TASK_ORDER:
        cells = []
        for c in ("D", "T", "DBT"):
            for L in LENGTHS:
                v = p[(p.task == t) & (p.condition == c) & (p.length == L)].accuracy
                cells.append(f"{v.iloc[0]:.2f}" if len(v) else "--")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines.append("")
    # Words that enter or leave the list, and where T changed.
    lines += ["## List words 21-30 and 31-40 (fold 1), and the words T finds in the prompts", "",
              "| Task | words 21-30 | words 31-40 | T words (20 / 30 / 40) |", "|---|---|---|---|"]
    w40 = json.loads((OUT / "w40" / "fold1" / "word_lists.json").read_text())["tasks"]
    ts = {n: json.loads((OUT / f"w{n}" / "fold1" / "t_source.json").read_text()) for n in (20, 40)}
    for t in er.TASK_ORDER:
        w = w40[t]["words"]
        tw = " / ".join([", ".join(ts[20][t]["found"]), ", ".join(ts[20][t]["found_e13"]), ", ".join(ts[40][t]["found"])])
        lines.append(f"| {t} | {', '.join(w[20:30])} | {', '.join(w[30:40])} | {tw} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
