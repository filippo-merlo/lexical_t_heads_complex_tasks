"""E25 report: S split into P (punctuation) and W (stopwords), at 40 heads, check 1.

Accuracies as ``scripts/e13_report.py`` (cross-fitted alpha over {1, 2}, mean over the five rule
means, mean of the two folds). Each new set is shown next to the S version it replaces, from
``results/e22_all_combos`` (same protocol), with the share of heads the two sets have in common.

Writes ``results/e25_punct_split/{per_task,summary}.csv`` and ``report.md``.

    .venv/bin/python scripts/e25_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e13_report as e13  # noqa: E402
from e21_report import md  # noqa: E402
from lth import RESULTS  # noqa: E402

OUT = RESULTS / "e25_punct_split"
ROWS = [("task", "DBTMS"), ("shared", "DBTMS"), ("shared", "DBTLS")]
SCOPES = {"task": "per task", "shared": "shared"}


def main() -> None:
    e13.OUT = OUT
    curves, _ = e13.load()
    per, summ = e13.summarise(e13.crossfit(curves))
    per.to_csv(OUT / "per_task.csv", index=False)
    summ.to_csv(OUT / "summary.csv", index=False)
    old_s = pd.read_csv(RESULTS / "e22_all_combos" / "all_combos.csv")
    old_p = pd.read_csv(RESULTS / "e22_all_combos" / "per_task.csv")
    sets = pd.concat([pd.read_csv(OUT / f"fold{f}" / "sets.csv") for f in (1, 2)])
    overlap = sets.groupby("set").shared_with_S_version.mean()

    s = pd.concat([summ, old_s[(old_s.n == 40)][list(summ.columns)]], ignore_index=True)
    p = pd.concat([per, old_p[old_p.n == 40]], ignore_index=True)
    groups = {t: g for g, ts in e13.GROUPS.items() for t in ts}
    p = p.assign(group=p.task.map(groups))

    lines = ["# E25: punctuation (P) and stopwords (W) as separate filters, 40 heads, check 1", "",
             "S = attention to punctuation + NLTK stopwords (Bai et al.'s stopword type); P = punctuation only; "
             "W = stopwords only. In the three leading S combinations S is replaced by P and by W. "
             "Held-out items, cross-fitted alpha in {1, 2}, mean of the two folds and of the 14 tasks. "
             "S rows: E22 (same protocol). Shared heads: share of the 40 heads also in the S version, mean over tasks and folds.", ""]
    rows = []
    for scope, c in ROWS:
        for v in (c, c.replace("S", "P"), c.replace("S", "W")):
            r = s[(s.scope == scope) & (s.condition == v)].iloc[0]
            name = r.set
            rows.append({"scope": SCOPES[scope], "set": name, "task mean": r.check1_task, "control": r.check1_control,
                         "fold 1": r.fold1, "fold 2": r.fold2,
                         "shared heads with S version": 1.0 if v == c else overlap[name]})
    t = pd.DataFrame(rows).set_index(["scope", "set"])
    lines += ["## Mean over tasks", "", md(t.round(3)), ""]

    for scope, c in ROWS:
        names = [("shared_" if scope == "shared" else "") + v + "40" for v in (c, c.replace("S", "P"), c.replace("S", "W"))]
        g = (p[(p.source == "task") & p.set.isin(names)].groupby(["group", "set"]).accuracy.mean()
             .unstack("set")[names].reindex(list(e13.GROUPS)))
        lines += [f"## By task type: {SCOPES[scope]} {c}", "", md(g.round(2)), ""]
    pt = (p[(p.source == "task") & p.set.isin([r["set"] for r in rows])]
          .pivot(index="task", columns="set", values="accuracy")[[r["set"] for r in rows]])
    lines += ["## Per task", "", md(pt.round(2)), ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
