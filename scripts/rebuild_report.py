"""Rebuilt design (paper/head-sets-simple-design.tex): check-1 accuracies of a stage, cross-fitted.

Reads ``results/rebuild/additions/fold{1,2}/patching_simple.part-<prefix>*.csv`` (``scripts/patch.py``,
``scripts/e26_summed.py``). For every fold, task, set and mean source, accuracy is averaged over the
means of the templates (one row per template and alpha); alpha is cross-fitted (Sec. 3.4): each fold
takes the alpha that was better on the other fold for the same task and set, ties to 1. A random set
is scored at its rule's alpha, the control mean at the task mean's alpha. A set that does not exist for
a task (it cannot fill its size) leaves the task out of that set; the mean over tasks is reported only
where every task and both folds have the set (``--`` otherwise; Sec. 3.6).

Writes ``results/rebuild/additions/<stage>_per_task.csv`` (fold, task, set, source, alpha, accuracy)
and ``<stage>_summary.csv`` (set, scope, size, task / control / random accuracy over tasks and folds,
tasks), and prints the summary.

    .venv/bin/python scripts/rebuild_report.py --stage s1
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import RESULTS, simple  # noqa: E402

DIR = RESULTS / "rebuild" / "additions"
RANDOM = re.compile(r"^random_(.+)_(\d+)$")


def load(stage: str):
    """acc[(fold, task, set, source)][alpha] = mean accuracy over the template means."""
    cells = defaultdict(lambda: defaultdict(list))
    for fold in (1, 2):
        for path in sorted((DIR / f"fold{fold}").glob(f"patching_simple.part-{stage}*.csv")):
            for r in csv.DictReader(path.open()):
                if r["source"] == "none":
                    continue
                cells[(fold, r["task"], r["heads"], r["source"])][float(r["alpha"])].append(float(r["accuracy"]))
    return {k: {a: sum(v) / len(v) for a, v in d.items()} for k, d in cells.items()}


def crossfit(acc):
    out = {}
    for (fold, task, name, source), by_alpha in acc.items():
        m = RANDOM.match(name)
        rule = m.group(1) if m else name
        other = acc.get((3 - fold, task, rule, "task"), {})
        if not other:
            continue
        best = max(sorted(other), key=lambda a: (other[a], -a))  # ties to the smaller alpha
        if best in by_alpha:
            out[(fold, task, name, source)] = (best, by_alpha[best])
    return out


def size_of(name: str):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m and not name.endswith("_own") else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", required=True, help="part prefix, e.g. s1")
    args = parser.parse_args()
    cf = crossfit(load(args.stage))
    with (DIR / f"{args.stage}_per_task.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "task", "set", "source", "alpha", "accuracy"])
        for (fold, task, name, source), (alpha, a) in sorted(cf.items()):
            w.writerow([fold, task, name, source, alpha, round(a, 4)])
    rules = sorted({name for (_, _, name, _) in cf if not RANDOM.match(name)})
    summary = []
    for rule in rules:
        def over(source, names):
            vals, tasks = [], set()
            for fold in (1, 2):
                for t in simple.TASKS:
                    got = [cf[(fold, t, n, source)][1] for n in names if (fold, t, n, source) in cf]
                    if got:
                        vals.append(sum(got) / len(got))
                        tasks.add((fold, t))
            return vals, tasks
        task_vals, tasks = over("task", [rule])
        complete = len(tasks) == 2 * len(simple.TASKS)
        ctrl_vals, _ = over("control", [rule])
        rnd_names = sorted({n for (_, _, n, _) in cf if RANDOM.match(n) and RANDOM.match(n).group(1) == rule})
        rnd_vals, _ = over("task", rnd_names)
        mean = lambda v: round(sum(v) / len(v), 3) if v and complete else "--"  # noqa: E731
        summary.append(dict(set=rule, scope="shared" if rule.startswith("shared") else "per task", n=size_of(rule),
                            task=mean(task_vals), control=mean(ctrl_vals), random=mean(rnd_vals),
                            tasks_folds=len(tasks)))
    with (DIR / f"{args.stage}_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for r in sorted(summary, key=lambda r: (r["scope"], re.sub(r"\d+$", "", r["set"]), r["n"] or 0)):
        print(f"{r['set']:28s} {r['scope']:9s} N={str(r['n']):4s} task={r['task']!s:6s} control={r['control']!s:6s} "
              f"random={r['random']!s:6s} ({r['tasks_folds']}/28)")


if __name__ == "__main__":
    main()
