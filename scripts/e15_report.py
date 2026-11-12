"""E15 report: shared Causal / DBT minus corpus-defined vs across-task generic heads (check 1, all 35 means).

Same cross-fit as ``e14_report.py`` (alpha chosen on the other fold, ties to the smaller alpha; accuracy
averaged over the 35 templates' means, then folds, then tasks). E14's shared a35 rows are read from
``results/e14_wordings/per_task.csv``; ``a35_shared_causal40_nogen`` is patched again here as a check.
Writes ``results/e15_corpus_generic/summary.csv``, ``per_task.csv`` and prints the table.

    .venv/bin/python scripts/e15_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e14_report import GROUPS, crossfit, curves, load_rows  # noqa: E402
from lth import RESULTS  # noqa: E402

OUT = RESULTS / "e15_corpus_generic"
E14 = RESULTS / "e14_wordings"
CHECK = "a35_shared_causal40_nogen"


def main() -> None:
    d = load_rows(OUT)
    cf = crossfit(curves(d), lambda s: s)
    per = (cf.groupby(["task", "set", "source"], as_index=False)
           .agg(accuracy=("accuracy", "mean"), n_heads=("n_heads", "mean"),
                alpha2=("alpha", lambda a: float((a == 2).mean()))))
    per.to_csv(OUT / "per_task.csv", index=False)

    e14 = pd.read_csv(E14 / "per_task.csv")
    e14 = e14[e14.set.str.startswith("a35_shared_")][["task", "set", "source", "accuracy", "n_heads", "alpha2"]]
    mine = per[per.set == CHECK].set_index(["task", "source"]).accuracy
    ref = e14[e14.set == CHECK].set_index(["task", "source"]).accuracy
    diff = float((mine - ref.loc[mine.index]).abs().max())
    print(f"check: {CHECK} re-patched vs E14, largest per-task difference {diff:.2e}")

    allp = pd.concat([e14, per[per.set != CHECK]], ignore_index=True)
    allp.to_csv(OUT / "per_task_with_e14.csv", index=False)
    task = allp[allp.source == "task"]
    group = {t: g for g, ts in GROUPS.items() for t in ts}
    s = task.groupby("set").agg(check1=("accuracy", "mean"), alpha2=("alpha2", "mean"), tasks=("task", "nunique"))
    s["control"] = allp[allp.source == "control"].groupby("set").accuracy.mean()
    g = task.assign(group=task.task.map(group)).pivot_table(index="set", columns="group", values="accuracy")
    s = s.join(g[list(GROUPS)])
    s.to_csv(OUT / "summary.csv")
    order = [f"a35_shared_{m}{n}{x}" for m in ("causal", "DBT") for n in (40, 80) for x in ("", "_nogen", "_nogen_corpus")]
    pd.set_option("display.width", 200)
    print(s.reindex([o for o in order if o in s.index]).round(3).to_string())


if __name__ == "__main__":
    main()
