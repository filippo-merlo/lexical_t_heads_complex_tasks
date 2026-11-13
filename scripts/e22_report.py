"""E22 report: the 79 combinations of D, B, T, L, M, S, N at 40 heads (none with N and T or S), per
task and shared, check 1.

Accuracies as ``scripts/e13_report.py`` (cross-fitted alpha over {1, 2}, mean over the five rule
means, mean of the two folds), from three runs of the same protocol: E13 (D, B, T, DB, DT, BT, DBT),
E21 (X, BX, DBX for X in L, M, S, N) and E22 (the other 60).

Reports: every combination ranked (both scopes); what each letter adds (paired: accuracy of
combination + X minus that of the same combination without X, over every pair where both are
among the 79 combinations); the
best combination per number of letters; task types for the ten best.

Writes ``results/e22_all_combos/{per_task,summary,all_combos}.csv`` and ``report.md``.

    .venv/bin/python scripts/e22_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e13_report as e13  # noqa: E402
from e21_report import md  # noqa: E402
from e22_sets import COMBOS, LETTERS, N  # noqa: E402
from lth import RESULTS  # noqa: E402

OUT = RESULTS / "e22_all_combos"
NAMES = {"D": "decodes task words", "B": "low BOS", "T": "task words", "L": "last token",
         "M": "template markers", "S": "punctuation + stopwords", "N": "whole instruction"}
SCOPES = {"task": "per task", "shared": "shared"}


def main() -> None:
    e13.OUT = OUT
    curves, _ = e13.load()
    per, summ = e13.summarise(e13.crossfit(curves))
    per.to_csv(OUT / "per_task.csv", index=False)
    summ.to_csv(OUT / "summary.csv", index=False)

    s = [summ.assign(run="E22")]
    p = [per]
    for exp, run in (("e13_capped", "E13"), ("e21_bai_rules", "E21")):
        a = pd.read_csv(RESULTS / exp / "summary.csv")
        b = pd.read_csv(RESULTS / exp / "per_task.csv")
        s.append(a[(a.n == N) & a.condition.isin(COMBOS)].assign(run=run))
        p.append(b[(b.n == N) & b.condition.isin(COMBOS)])
    s, p = pd.concat(s, ignore_index=True), pd.concat(p, ignore_index=True)
    assert s.groupby("scope").condition.nunique().eq(len(COMBOS)).all(), "not all combinations have results"
    assert not s.duplicated(["scope", "condition"]).any()
    s["letters"] = s.condition.str.len()
    s.to_csv(OUT / "all_combos.csv", index=False)

    lines = [f"# E22: every combination of {', '.join(LETTERS)} at {N} heads (none with N and T or S), check 1", "",
             "Held-out items, cross-fitted alpha in {1, 2}, mean of the two folds, mean of 14 tasks. "
             "Runs: E13 (D, B, T, DB, DT, BT, DBT), E21 (X, BX, DBX for X in L, M, S, N), E22 (the rest), same protocol. "
             "Letters: " + "; ".join(f"{k} = {v}" for k, v in NAMES.items()) + ".", ""]

    # what each letter adds, paired
    acc = s.set_index(["scope", "condition"]).check1_task
    rows = []
    for x in LETTERS:
        for scope in SCOPES:
            d = []
            for c in COMBOS:
                w = "".join(l for l in LETTERS if l in c + x)
                if x in c or w not in COMBOS:
                    continue
                d.append(acc[(scope, w)] - acc[(scope, c)])
            d = np.array(d)
            rows.append(dict(letter=x, scope=SCOPES[scope], mean_gain=d.mean(), helps=int((d > 0.005).sum()),
                             hurts=int((d < -0.005).sum()), pairs=len(d)))
    g = pd.DataFrame(rows).pivot(index="letter", columns="scope", values=["mean_gain", "helps", "hurts"]).reindex(list(LETTERS))
    g.columns = [f"{v} / {sc}" for v, sc in g.columns]
    g = g.round(3).astype({c: int for c in g.columns if not c.startswith("mean_gain")})
    lines += ["## What each letter adds",
              "", "Accuracy of a combination with the letter minus the same combination without it (the "
              "letter alone vs nothing is left out); helps/hurts: pairs moving by more than 0.005.", "",
              md(g), ""]

    # best per number of letters
    b = (s.sort_values("check1_task", ascending=False).groupby(["scope", "letters"]).head(1)
         .sort_values(["scope", "letters"])[["scope", "letters", "condition", "check1_task", "check1_control"]])
    b = b.assign(scope=b.scope.map(SCOPES)).set_index("scope")
    lines += ["## Best combination per number of letters", "", md(b.round(3)), ""]

    # ten best: task types
    groups = {t: g for g, ts in e13.GROUPS.items() for t in ts}
    pt = p[p.source == "task"].assign(group=lambda x: x.task.map(groups))
    for scope in SCOPES:
        top = s[s.scope == scope].nlargest(10, "check1_task").condition.tolist()
        t = pt[(pt.scope == scope)].groupby(["condition", "group"]).accuracy.mean().unstack("group").reindex(top)[list(e13.GROUPS)]
        t.insert(0, "mean", s[s.scope == scope].set_index("condition").check1_task.reindex(top))
        lines += [f"## Ten best, by task type ({SCOPES[scope]})", "", md(t.round(2)), ""]

    # all combinations ranked
    r = s.pivot(index="condition", columns="scope", values=["check1_task", "check1_control", "fold1", "fold2"])
    r.columns = [f"{v.replace('check1_', '')} / {SCOPES[sc]}" for v, sc in r.columns]
    r = r.sort_values("task / per task", ascending=False)
    r.insert(0, "run", s.drop_duplicates("condition").set_index("condition").run.reindex(r.index))
    cols = ["run"] + [f"{v} / {SCOPES[sc]}" for sc in SCOPES for v in ("task", "control", "fold1", "fold2")]
    lines += [f"## All {len(COMBOS)} combinations (sorted by per-task accuracy)", "", md(r[cols].round(3)), ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
