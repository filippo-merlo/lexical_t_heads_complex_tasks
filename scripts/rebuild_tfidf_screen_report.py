"""Stage-2 screen with the TF-IDF word lists (``rebuild_tfidf_sets.py --screen``,
``rebuild_quick_screen.sh tfidf_screen all``): check-1 accuracy at 40 heads of the shortlisted combinations and
of Selected, next to the same combinations with the stage-2 lists (stage 2's full screen,
results/rebuild/additions). Both scored as ``scripts/rebuild_report.py``: accuracy averaged over the template
means, alpha cross-fitted (the alpha better on the other fold for the same task and set, ties to 1), mean
over the 14 tasks and both folds; task means only. Selected: each fold's chosen combination on its own
test half (``selected.json`` of each pool), mean over the folds.

Repeated replies: for the TF-IDF Selected sets, per fold and task at the cross-fitted alpha, the share of
patched replies equal to the most common one; listed where it is at least 0.5.

Writes ``results/rebuild/tfidf_screen/``: ``summary.csv`` (scope, combination, shortlisted by which fold,
selected by which fold, accuracy with each list over tasks and folds and per fold), ``per_task.csv`` (scope,
task, Selected with each list, mean over the folds) and ``repeated.csv``; prints them.

    .venv/bin/python scripts/rebuild_tfidf_screen_report.py
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rebuild_report  # noqa: E402
from lth import RESULTS, simple  # noqa: E402

ADD = RESULTS / "rebuild" / "additions"
TS = RESULTS / "rebuild" / "tfidf_screen"
LISTS = {"stage2": (ADD, "s2"), "tfidf": (TS, "s2-screen")}
SCOPES = {"per task": "screen_", "shared": "shared_screen_"}


def scores():
    out = {}
    for lst, (d, stage) in LISTS.items():
        rebuild_report.DIR = d
        cf = rebuild_report.crossfit(rebuild_report.load(stage))
        out[lst] = {(f, t, n): v for (f, t, n, s), v in cf.items() if s == "task" and "screen_" in n}
    return out


def main() -> None:
    short = json.loads((TS / "shortlist.json").read_text())
    sel = {lst: json.loads((d / "selected.json").read_text()) for lst, (d, _) in LISTS.items()}
    sc = scores()

    def acc(lst, fold, name):
        vals = [sc[lst][(fold, t, name)][1] for t in simple.TASKS if (fold, t, name) in sc[lst]]
        assert len(vals) == len(simple.TASKS), (lst, fold, name, len(vals))
        return float(np.mean(vals))

    summary, per_task = [], []
    for scope, prefix in SCOPES.items():
        for lst in LISTS:
            picks = {f: sel[lst][str(f)][scope]["combo"] for f in (1, 2)}
            row = dict(scope=scope, combo=f"Selected ({lst}: {picks[1]} / {picks[2]})", shortlisted="", selected="")
            for l2 in LISTS:
                if l2 == lst:
                    per_fold = [acc(lst, f, prefix + picks[f]) for f in (1, 2)]
                    row[l2], row[f"{l2}_fold1"], row[f"{l2}_fold2"] = (round(float(np.mean(per_fold)), 4),
                                                                       round(per_fold[0], 4), round(per_fold[1], 4))
                else:
                    row[l2] = row[f"{l2}_fold1"] = row[f"{l2}_fold2"] = ""
            summary.append(row)
        for c in short["union"][scope]:
            row = dict(scope=scope, combo=c,
                       shortlisted=",".join(str(f) for f in (1, 2) if c in [x for x, _ in short[str(f)][scope]]),
                       selected=",".join(str(f) for f in (1, 2) if sel["tfidf"][str(f)][scope]["combo"] == c))
            for lst in LISTS:
                per_fold = [acc(lst, f, prefix + c) for f in (1, 2)]
                row[lst], row[f"{lst}_fold1"], row[f"{lst}_fold2"] = (round(float(np.mean(per_fold)), 4),
                                                                      round(per_fold[0], 4), round(per_fold[1], 4))
            summary.append(row)
        for t in simple.TASKS:
            r = dict(scope=scope, task=t)
            for lst in LISTS:
                r[lst] = round(float(np.mean([sc[lst][(f, t, prefix + sel[lst][str(f)][scope]["combo"])][1] for f in (1, 2)])), 4)
            per_task.append(r)
    with (TS / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with (TS / "per_task.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_task[0]))
        w.writeheader()
        w.writerows(per_task)

    # Repeated replies of the TF-IDF Selected sets.
    wanted = {(f, prefix + sel["tfidf"][str(f)][scope]["combo"]) for f in (1, 2) for scope, prefix in SCOPES.items()}
    replies = defaultdict(list)
    for f in (1, 2):
        for path in sorted((TS / f"fold{f}" / "answers").glob("patching_simple.part-s2-screen-*.jsonl.gz")):
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get("source") == "task" and (f, r.get("heads")) in wanted:
                        replies[(f, r["task"], r["heads"], float(r["alpha"]))].append(r["predicted"])
    rep, top_share = [], 0.0
    for (f, t, name, a), preds in replies.items():
        if sc["tfidf"][(f, t, name)][0] != a:
            continue
        reply, n = Counter(preds).most_common(1)[0]
        top_share = max(top_share, n / len(preds))
        if n / len(preds) >= 0.5:
            rep.append(dict(fold=f, task=t, set=name, alpha=a, reply=reply, share=round(n / len(preds), 3),
                            accuracy=round(sc["tfidf"][(f, t, name)][1], 3)))
    with (TS / "repeated.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "task", "set", "alpha", "reply", "share", "accuracy"])
        w.writeheader()
        w.writerows(rep)

    for r in summary:
        print(f"{r['scope']:9s} {r['combo']:34s} short={r['shortlisted']:4s} sel={r['selected']:4s} "
              + "  ".join(f"{lst} {r[lst]!s:6s} ({r[f'{lst}_fold1']!s:6s} {r[f'{lst}_fold2']!s:6s})" for lst in LISTS))
    for r in per_task:
        print(f"{r['scope']:9s} {r['task']:17s} stage2 {r['stage2']:.3f}  tfidf {r['tfidf']:.3f}  {r['tfidf'] - r['stage2']:+.3f}")
    print(f"repeated replies of the TF-IDF Selected sets (share >= 0.5): {len(rep)}; largest share {top_share:.3f}")


if __name__ == "__main__":
    main()
