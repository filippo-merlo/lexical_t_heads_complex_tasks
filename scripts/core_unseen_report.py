"""Report of the core-heads test on the unseen tasks (results/rebuild/newtasks_core). No GPU.

As scripts/rebuild_newtask_report.py: accuracy of a (fold, task, set, source, alpha) cell = mean over the
template means; alpha fixed (1, 2) and also cross-fitted between folds (rebuild_report.crossfit); control =
the control task's means; random = mean of the three draws of the core (task means). Unpatched accuracy and
majority-label rate of the patched items; conditions with one reply >= 50% of replies.
Reference on the 14 tasks: none at 7 heads exists, so none is shown.
Writes per_task.csv, summary.csv, repeated.csv into the run folder.

    .venv/bin/python scripts/core_unseen_report.py
"""
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from lth import simple  # noqa: E402
import rebuild_report  # noqa: E402

DIR = ROOT / "results" / "rebuild" / "newtasks_core"
RULES = {"core (7)": "core7", "shared DBTL top 7": "shared_DBTL7", "shared LRP-T top 7": "shared_lrpT7",
         "shared Causal top 7": "shared_causal7", "shared Causal - generic top 7": "shared_causalnogen7"}
TASKS = simple.UNSEEN_TASKS
SHORT = {"person-occupation": "occupation", "person-sport": "sport", "person-instrument": "instrument", "capitalize": "capitalize"}


def main():
    rebuild_report.DIR = DIR
    acc = rebuild_report.load("core")
    cf = rebuild_report.crossfit(acc)
    rows = []
    for label, nm in RULES.items():
        for alpha in (1.0, 2.0, "cf"):
            for fold in (1, 2):
                for t in TASKS:
                    def get(name, source):
                        if alpha == "cf":
                            v = cf.get((fold, t, name, source))
                            return v[1] if v else np.nan
                        return acc[(fold, t, name, source)][alpha]
                    rnd = float(np.mean([get(f"random_{nm}_{i}", "task") for i in range(3)])) if nm == "core7" else np.nan
                    rows.append(dict(rule=label, alpha=alpha, fold=fold, task=t, accuracy=get(nm, "task"),
                                     control=get(nm, "control"), random=rnd))
    with (DIR / "per_task.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for label in RULES:
        for alpha in (1.0, 2.0, "cf"):
            sel = [r for r in rows if r["rule"] == label and r["alpha"] == alpha]
            line = dict(rule=label, alpha=alpha, **{SHORT[t]: np.mean([r["accuracy"] for r in sel if r["task"] == t]) for t in TASKS})
            line["unseen mean"] = np.mean([r["accuracy"] for r in sel])
            line["control"] = np.mean([r["control"] for r in sel])
            line["random"] = np.mean([r["random"] for r in sel])
            summary.append(line)
    with (DIR / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    # baseline and repeated replies
    repeated = []
    print("baseline of the patched items:")
    for fold in (1, 2):
        for t in TASKS:
            lines = [json.loads(x) for x in gzip.open(DIR / f"fold{fold}" / "answers" / f"patching_simple.part-core-{t}.jsonl.gz", "rt")]
            none = [x for x in lines if x["source"] == "none"]
            lab, cnt = Counter(x["expected"] for x in none).most_common(1)[0]
            print(f"  fold {fold} {t:18s} items {len(none):3d} unpatched {np.mean([x['correct'] for x in none]):.3f} "
                  f"majority '{lab}' {cnt / len(none):.3f}")
            groups = defaultdict(list)
            for x in lines:
                if x["source"] != "none":
                    groups[(x["heads"], x["source"], x["mean_wording"], x["alpha"])].append(x)
            for (nm, source, wd, a), xs in groups.items():
                reply, n = Counter(x["predicted"].strip().lower() for x in xs).most_common(1)[0]
                if n / len(xs) >= 0.5:
                    repeated.append(dict(fold=fold, task=t, set=nm, source=source, wording=wd, alpha=a, reply=reply,
                                         share=round(n / len(xs), 3),
                                         correct=round(float(np.mean([x["correct"] for x in xs if x["predicted"].strip().lower() == reply])), 3)))
    with (DIR / "repeated.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "task", "set", "source", "wording", "alpha", "reply", "share", "correct"])
        w.writeheader()
        w.writerows(repeated)
    print("\naccuracy at 7 heads, mean of the two folds (cf = alpha cross-fitted between folds):")
    for s in summary:
        a = s["alpha"] if s["alpha"] == "cf" else f"{s['alpha']:.0f}"
        print(f"  {s['rule']:30s} a={a:2s} " + "  ".join(f"{SHORT[t]}={s[SHORT[t]]:.3f}" for t in TASKS)
              + f"  | mean {s['unseen mean']:.3f}  ctrl {s['control']:.3f}  rand {s['random']:.3f}")
    own = [r for r in repeated if r["source"] == "task" and not r["set"].startswith("random_")]
    print(f"\nconditions with one reply >= 50% of replies: {len(repeated)} ({len(own)} with task means on a rule's own set)")
    for r in own[:20]:
        print("  ", r)


if __name__ == "__main__":
    main()
