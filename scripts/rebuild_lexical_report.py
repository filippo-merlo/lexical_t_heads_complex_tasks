"""Step A report (``scripts/rebuild_lexical.py``, ``scripts/rebuild_lexical.sh``): check-1 accuracy of the
40-head sets of every combination of the decoding features T, Y, A, C, alone and with Selected's attention part
(per task ntspl, shared ntm), scored as ``scripts/rebuild_report.py`` (template means averaged, alpha
cross-fitted, mean over the 14 tasks and both folds; task means only), and the share of patched replies that
name the task: a word of Yang et al.'s term list for the task (whole word, any case) in the reply, at the same
alpha. Also each set's median layer and the heads it shares with the T set of its scope.

Writes ``results/rebuild/lexical/summary.csv`` and ``per_task.csv``; prints the summary.

    .venv/bin/python scripts/rebuild_lexical_report.py
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rebuild_report  # noqa: E402
from lth import RESULTS, simple  # noqa: E402
from lth.terms import simple_term_lists  # noqa: E402

DIR = RESULTS / "rebuild" / "lexical"


def display(name: str) -> str:
    body = name.removeprefix("shared_")
    if body.startswith("mix_"):  # mix_T30C10 -> T·attn 30 + C·attn 10
        t, c = body[5:].split("C")
        return f"T+attn {t} / C+attn {c}"
    return body.removeprefix("lex_").replace("_", "·")


def main() -> None:
    rebuild_report.DIR = DIR
    cf = {k: v for k, v in rebuild_report.crossfit(rebuild_report.load("lex")).items() if k[3] == "task"}
    pats = {t: re.compile(r"\b(" + "|".join(re.escape(w) for w in simple_term_lists()[t]) + r")\b", re.I)
            for t in simple.TASKS}
    named = defaultdict(list)  # (fold, task, set, alpha) -> reply names the task
    for f in (1, 2):
        for t in simple.TASKS:
            for part in ("lex", "lex-mix"):
                path = DIR / f"fold{f}" / "answers" / f"patching_simple.part-{part}-{t}.jsonl.gz"
                if not path.exists():
                    continue
                with gzip.open(path, "rt") as fh:
                    for line in fh:
                        r = json.loads(line)
                        if r["source"] == "task":
                            named[(f, t, r["heads"], float(r["alpha"]))].append(bool(pats[t].search(r["reply"])))
    sets = {}
    for f in (1, 2):
        for r in csv.DictReader((DIR / f"fold{f}" / "sets.csv").open()):
            sets.setdefault(r["set"], defaultdict(list))
            sets[r["set"]]["layer"].append(float(r["layer_median"]))
            sets[r["set"]]["overlap"].append(int(r["overlap_T"]))
    names = sorted({k[2] for k in cf}, key=lambda n: (n.startswith("shared"), "_" in n.removeprefix("shared_").removeprefix("lex_"), len(n), n))
    per_task, summary = [], []
    for n in names:
        acc, nm = [], []
        for f in (1, 2):
            for t in simple.TASKS:
                a, x = cf[(f, t, n, "task")]
                acc.append(x)
                v = named[(f, t, n, a)]
                nm.append(sum(v) / len(v))
                per_task.append(dict(set=n, fold=f, task=t, alpha=a, accuracy=round(x, 4), names_task=round(nm[-1], 4)))
        assert len(acc) == 2 * len(simple.TASKS), n
        summary.append(dict(scope="shared" if n.startswith("shared") else "per task", set=display(n), name=n,
                            accuracy=round(float(np.mean(acc)), 4), names_task=round(float(np.mean(nm)), 4),
                            layer_median=float(np.mean(sets[n]["layer"])), overlap_T=float(np.mean(sets[n]["overlap"]))))
    for path, rows in ((DIR / "summary.csv", summary), (DIR / "per_task.csv", per_task)):
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for scope in ("per task", "shared"):
        print(f"== {scope} (40 heads): accuracy / share of replies naming the task / median layer / heads shared with T")
        for r in summary:
            if r["scope"] == scope:
                print(f"   {r['set']:12s} {r['accuracy']:.3f} / {r['names_task']:.3f} / L{r['layer_median']:4.1f} / {r['overlap_T']:4.1f}")


if __name__ == "__main__":
    main()
