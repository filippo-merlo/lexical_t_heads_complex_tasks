"""E21 report: the token-type rankings (scripts/e21_sets.py, e21_patch.sh) next to E13's sets at 40 heads.

Same computation as ``scripts/e13_report.py`` (its ``load``, ``crossfit`` and ``summarise`` on
``results/e21_bai_rules``): check 1, accuracy at the alpha over {1, 2} chosen on the other fold
(ties to alpha 1), task accuracy = mean over the five rule means, mean of the two folds. E13's rows
at N = 40 are read from ``results/e13_capped/{summary,per_task}.csv``.

Writes ``results/e21_bai_rules/{per_task,summary}.csv`` and ``report.md``.

    .venv/bin/python scripts/e21_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e13_report as e13  # noqa: E402
from lth import RESULTS  # noqa: E402

OUT = RESULTS / "e21_bai_rules"
N = 40
NEW = [p + x for x in "LMSN" for p in ("", "B", "DB")]
REF = ["T", "BT", "DBT", "D", "B", "causal", "causal_nogen", "random"]
LABEL = {"L": "last token", "M": "template markers", "S": "punctuation + stopwords", "N": "whole instruction"}


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    rows += ["| " + " | ".join([str(i)] + ["--" if pd.isna(v) else str(v) for v in r]) + " |"
             for i, r in zip(df.index, df.values)]
    return "\n".join(rows)


def main() -> None:
    e13.OUT = OUT
    curves, _ = e13.load()
    cf = e13.crossfit(curves)
    per, summ = e13.summarise(cf)
    per.to_csv(OUT / "per_task.csv", index=False)
    summ.to_csv(OUT / "summary.csv", index=False)

    ref_s = pd.read_csv(RESULTS / "e13_capped" / "summary.csv")
    ref_p = pd.read_csv(RESULTS / "e13_capped" / "per_task.csv")
    ref_s, ref_p = ref_s[(ref_s.n == N) & ref_s.condition.isin(REF)], ref_p[(ref_p.n == N) & ref_p.condition.isin(REF)]
    s = pd.concat([summ.assign(source_exp="E21"), ref_s.assign(source_exp="E13")], ignore_index=True)
    p = pd.concat([per, ref_p], ignore_index=True)

    order = NEW + REF
    lines = [f"# E21: token-type rankings at {N} heads, check 1", "",
             "Held-out items, cross-fitted alpha in {1, 2}, mean of the two folds; task accuracy = mean over "
             "the five rule means. E13 rows: results/e13_capped (same protocol). Letters: "
             + "; ".join(f"{k} = {v}" for k, v in LABEL.items()) + "; B = low BOS; D = decodes the task's words.", ""]
    tab = (s.pivot_table(index="condition", columns="scope", values=["check1_task", "check1_control"])
           .reindex(order))
    tab.columns = [f"{'task mean' if v == 'check1_task' else 'control'} / {sc}" for v, sc in tab.columns]
    tab = tab[[c for c in ("task mean / task", "control / task", "task mean / shared", "control / shared") if c in tab]]
    lines += ["## Mean over the 14 tasks", "", md(tab.round(3)), ""]

    groups = {t: g for g, ts in e13.GROUPS.items() for t in ts}
    pt = p[p.source == "task"].assign(group=lambda x: x.task.map(groups))
    for scope in ("task", "shared"):
        g = (pt[pt.scope == scope].groupby(["condition", "group"]).accuracy.mean().unstack("group")
             .reindex(order)[list(e13.GROUPS)])
        lines += [f"## By task type ({'per task' if scope == 'task' else 'shared'})", "", md(g.round(2)), ""]
        t = pt[pt.scope == scope].groupby(["condition", "task"]).accuracy.mean().unstack("task").reindex(order)[e13.TASK_ORDER]
        lines += [f"Per task ({'per task' if scope == 'task' else 'shared'}):", "", md(t.round(2)), ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
