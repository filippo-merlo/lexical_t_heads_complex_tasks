"""E10 (scripts/e10_sets.py, e10_patch.sh): accuracies of the 93 shared 30-head sets, E8's protocol,
alphas 1 and 2, no statistical tests.

As in ``scripts/e8_report.py``: task accuracy = mean over the five rule means of the choosing half; a
set's accuracy at its cross-fitted alpha (fold 1 at the best alpha of fold 2 and the reverse; ties:
smaller |alpha|, then the positive one), mean of the two folds.

Set choice without the reported items: the set with the highest check-1 task-mean accuracy on fold 1
(each task at its own best alpha) is picked, and its fold-2 cross-fitted accuracy is reported.

References on the same items and folds, cross-fitted over alpha 1 and 2: E8's per-task set of the
same filter combination (E8's thresholds, each task its own heads and size), and E8's other references
(E7 raw_att, E4 CC-20, FV-20, CC, FV-N_t). Davidson et al.'s shared FV heads (all30, task24; scripts/
davidson_fv.py, e3/fv_split_check.py) were run on all items at fixed alphas, with the FV added to the
residual stream: listed apart, not on the same items.

Answers: for the five best sets in check 1, the share of each task's patched items given the task's
most common answer (mean over tasks) and the most common answers overall, at alpha 1 and 2.

Writes ``results/e10_shared/accuracy_alpha.csv``, ``accuracy_crossfit.csv``, ``report.md``.

    .venv/bin/python scripts/e10_report.py
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import heldout_report as hr  # noqa: E402
from e10_sets import WAYS  # noqa: E402
from e7_report import md  # noqa: E402
from e8_report import ALPHAS, CHECKS, REFS, crossfit, refs  # noqa: E402
from e8_sets import COMBOS  # noqa: E402
from lth import RESULTS, simple  # noqa: E402

OUT = RESULTS / "e10_shared"
SPLIT = RESULTS / "davidson_fv_split"


def e8_same() -> pd.DataFrame:
    """E8's per-task sets, cross-fitted over alpha 1 and 2, named ``e8_<combination>``."""
    c = pd.read_csv(RESULTS / "e8_filters" / "accuracy_alpha.csv")
    c = c[c.alpha.isin(ALPHAS) & c.set.isin(COMBOS)]
    return crossfit(c).assign(set=lambda x: "e8_" + x.set)[["fold", "family", "task", "set", "source", "accuracy"]]


def davidson() -> pd.DataFrame:
    """Davidson FV check 1, all items: all30 and task24, task and control means over tasks, alpha 1, 2."""
    import glob

    x = pd.concat(pd.read_csv(p) for p in sorted(glob.glob(str(SPLIT / "patching_simple.part-*.csv"))))
    x = x[x.alpha.isin(ALPHAS) & x.set.isin(["all30", "task24"])]
    return x.pivot_table(index="set", columns=["source", "alpha"], values="accuracy", aggfunc="mean")


def answers(sets: list[str]) -> list[str]:
    """Most-common-answer share and top answers of ``sets`` in check 1, alpha 1 and 2, both folds."""
    tally: dict[tuple[str, float], dict[str, Counter]] = {}
    for fold in (1, 2):
        for path in (OUT / f"fold{fold}" / "answers").glob("patching_simple.part-*.jsonl.gz"):
            with gzip.open(path, "rt") as f:
                for line in f:
                    r = json.loads(line)
                    if r["heads"] in sets and r["source"] == "task":
                        tally.setdefault((r["heads"], r["alpha"]), {}).setdefault(r["task"], Counter())[r["predicted"]] += 1
    lines = ["| set | alpha | top-answer share (task mean) | top answers overall |", "|---|---|---|---|"]
    for (name, alpha), by_task in sorted(tally.items()):
        share = sum(c.most_common(1)[0][1] / sum(c.values()) for c in by_task.values()) / len(by_task)
        overall = sum(by_task.values(), Counter())
        common = "; ".join(f"`{str(a)[:20]}` {n}" for a, n in overall.most_common(5)).replace("|", "\\|")
        lines.append(f"| {name} | {alpha:g} | {share:.2f} | {common} |")
    return ["\n".join(lines)] if tally else ["(no answers found)"]


def main() -> None:
    hr.HELD = OUT
    curves, base = hr.load()
    curves.to_csv(OUT / "accuracy_alpha.csv", index=False)
    cf = crossfit(curves)
    cf.to_csv(OUT / "accuracy_crossfit.csv", index=False)
    perfold = pd.concat([cf[["fold", "family", "task", "set", "source", "accuracy"]], e8_same(), refs()])
    s = perfold.groupby(["family", "task", "set", "source"], as_index=False).accuracy.mean()
    task_mean = s.groupby(["family", "source", "set"]).accuracy.mean()
    sets = pd.read_csv(OUT / "fold1" / "sets.csv").set_index("set")

    lines = ["# E10: one shared 30-head set for all tasks, from E8's five scores", "",
             "Each of E8's per-head scores (D decoding, B BOS, T task words, I input, R rest; choosing half) is "
             "combined over the 14 tasks: `mean` (average), `worst` (weakest task), `rank` (average within-task "
             "rank); higher is better for D, T, I, lower for B, R. For each of the 31 combinations the heads are "
             "ranked on each of its filters' combined scores, the ranks averaged, and the best 30 kept: one set "
             "for every task (each task still patched with its own means). Accuracy at the cross-fitted alpha "
             "(1 or 2), mean of the two folds, mean over the 14 tasks. `e8` = E8's per-task set of the same "
             "combination (its thresholds, each task its own heads and size).", ""]
    for family, check in CHECKS.items():
        for source in ("task", "control"):
            g = pd.DataFrame({way: [task_mean.get((family, source, f"{way}_{c}")) for c in COMBOS] for way in WAYS},
                             index=COMBOS)
            g["e8 per task"] = [task_mean.get((family, source, f"e8_{c}")) for c in COMBOS]
            lines += [f"## {check}, {source} mean, mean over tasks", "", md(g), ""]
    unpatched = base.groupby("family").accuracy.mean()
    lines += ["Unpatched: " + ", ".join(f"{CHECKS[f]} {unpatched[f]:.2f}" for f in CHECKS), ""]

    shared = [f"{w}_{c}" for w in WAYS for c in COMBOS]
    over = pd.DataFrame({f"{CHECKS[f]} {src}": task_mean.loc[f, src] for f in CHECKS for src in ("task", "control")})
    top = over.reindex(shared).sort_values("check 1 task", ascending=False).head(15)
    top["Davidson 30"] = sets.davidson30.reindex(top.index)
    top["generic 6"] = sets.generic6.reindex(top.index)
    top["share in E8 set"] = sets.e8_share.reindex(top.index)
    lines += ["## Best 15 shared sets by check 1 (fold-1 heads: Davidson 30 / generic 6 overlap, mean share of "
              "the set in E8's per-task set of the same combination)", "", md(top), ""]
    lines += ["## References on the same items (cross-fitted over alpha 1 and 2)", "",
              md(over.reindex([r for r in REFS if r in over.index])), ""]
    lines += ["## Davidson et al.'s shared FV (all items, fixed alpha, FV added after layer 11; not the same "
              "items or intervention)", "", md(davidson()), ""]

    best = top.index[:5].tolist()
    for check_family in CHECKS:
        g = s[(s.family == check_family) & (s.source == "task") & s.set.isin(best + [f"e8_{n.split('_', 1)[1]}" for n in best])]
        g = g.pivot_table(index="task", columns="set", values="accuracy").reindex(list(simple.TASKS))
        g.loc["mean"] = g.mean()
        lines += [f"## {CHECKS[check_family]}, task mean per task: best 5 shared sets and their E8 per-task counterparts",
                  "", md(g.T), ""]

    own = hr.best_alpha(curves).merge(curves, on=["fold", "family", "task", "set", "alpha"])
    own = own[(own.fold == 1) & (own.family == "simple") & (own.source == "task")]
    pick = own.groupby("set").accuracy.mean().sort_values(ascending=False)
    f2 = perfold[(perfold.fold == 2) & (perfold.family == "simple") & (perfold.source == "task")].groupby("set").accuracy.mean()
    sel = pd.DataFrame({"fold 1, own best alpha": pick, "fold 2, cross-fitted": f2.reindex(pick.index)}).head(10)
    lines += ["## Set choice on fold 1, read on fold 2 (check 1, task mean, mean over tasks; top 10)", "", md(sel), "",
              f"Picked: **{pick.index[0]}**, fold 2 = {f2[pick.index[0]]:.2f}; its E8 per-task counterpart on fold 2: "
              f"{f2.get('e8_' + pick.index[0].split('_', 1)[1], float('nan')):.2f}.", ""]
    lines += ["## Answers of the best 5 shared sets (check 1, task means, both folds)", "", *answers(best), ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
