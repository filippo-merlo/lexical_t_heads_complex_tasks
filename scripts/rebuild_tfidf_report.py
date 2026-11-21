"""Word-list variant (``scripts/rebuild_tfidf_sets.py``, ``scripts/rebuild_tfidf.sh``): check-1 accuracy of
the shared 40-head sets re-identified with the smooth TF-IDF list, next to the stage-2 screen sets of the
same combinations (``shared_screen_<combo>``, results/rebuild/additions). Both scored as in
``scripts/rebuild_report.py``: accuracy averaged over the template means, alpha cross-fitted (each fold
takes the alpha better on the other fold for the same task and set, ties to 1), then the mean over the
14 tasks and both folds. Task means only (as in the screen).

Repeated replies: for every set, fold and task at its cross-fitted alpha, the share of patched replies
equal to the most common one, and that reply; listed where the share is at least 0.5.

Each fold's choice among its own five (Selected's rule, scored on the other fold) is printed with its
accuracy on the fold's own test half.

Writes ``results/rebuild/tfidf/per_task.csv`` (fold, task, combo, list, alpha, accuracy),
``summary.csv`` (combo, top 5 of which fold, accuracy of each list over tasks and folds, per fold, change
from stage 2, tasks up / down, heads in common with stage 2) and ``repeated.csv``; prints the summary.

    .venv/bin/python scripts/rebuild_tfidf_report.py
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
LISTS = {"stage2": (ADD, "s2", "shared_screen_"),
         "tfidf": (RESULTS / "rebuild" / "tfidf", "tfidf", "shared_tfidf_")}


def scores(lists):
    out = {}
    for lst in lists:
        d, stage, prefix = LISTS[lst]
        rebuild_report.DIR = d
        cf = rebuild_report.crossfit(rebuild_report.load(stage))
        out[lst] = {(f, t, n[len(prefix):]): v for (f, t, n, s), v in cf.items() if s == "task" and n.startswith(prefix)}
    return out


def repeated(combos, sc, lists):
    rows = []
    for lst in lists:
        d, stage, prefix = LISTS[lst]
        replies = defaultdict(list)  # (fold, task, combo, alpha) -> predicted
        for fold in (1, 2):
            for path in sorted((d / f"fold{fold}" / "answers").glob(f"patching_simple.part-{stage}-*.jsonl.gz")):
                if stage == "s2" and "s2sel" in path.name:
                    continue
                with gzip.open(path, "rt") as f:
                    for line in f:
                        r = json.loads(line)
                        name = r.get("heads", "")
                        if r.get("source") == "task" and name.startswith(prefix) and name[len(prefix):] in combos:
                            replies[(fold, r["task"], name[len(prefix):], float(r["alpha"]))].append(r["predicted"])
        for (fold, t, c, a), preds in replies.items():
            if sc[lst].get((fold, t, c), (None,))[0] != a:
                continue
            reply, n = Counter(preds).most_common(1)[0]
            if n / len(preds) >= 0.5:
                rows.append(dict(list=lst, fold=fold, task=t, combo=c, alpha=a, reply=reply,
                                 share=round(n / len(preds), 3), accuracy=round(sc[lst][(fold, t, c)][1], 3)))
    return rows


def main() -> None:
    new, shown, scanned = "tfidf", list(LISTS), list(LISTS)
    out_dir = LISTS[new][0]
    stage2 = json.loads((ADD / "selected.json").read_text())
    top5 = {f: [c for c, _ in stage2[str(f)]["shared"]["top10_other_fold"][:5]] for f in (1, 2)}
    combos = list(dict.fromkeys(top5[1] + top5[2]))
    sc = scores(shown)
    with (out_dir / "per_task.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "task", "combo", "list", "alpha", "accuracy"])
        for lst in shown:
            for (fold, t, c), (a, acc) in sorted(sc[lst].items()):
                if c in combos:
                    w.writerow([fold, t, c, lst, a, round(acc, 4)])
    sets = {}
    for fold in (1, 2):
        sets[fold] = {r["combo"]: int(r["overlap_stage2"]) for r in csv.DictReader((out_dir / f"fold{fold}" / "sets.csv").open())}
    summary = []
    for c in combos:
        row = dict(combo=c, top5_of_fold=",".join(str(f) for f in (1, 2) if c in top5[f]))
        for lst in shown:
            vals = {f: [sc[lst][(f, t, c)][1] for t in simple.TASKS if (f, t, c) in sc[lst]] for f in (1, 2)}
            assert all(len(v) == len(simple.TASKS) for v in vals.values()), (lst, c, {f: len(v) for f, v in vals.items()})
            row[lst] = round(float(np.mean(vals[1] + vals[2])), 4)
            for f in (1, 2):
                row[f"{lst}_fold{f}"] = round(float(np.mean(vals[f])), 4)
        row["diff"] = round(row[new] - row["stage2"], 4)
        per_task = [np.mean([sc[new][(f, t, c)][1] - sc["stage2"][(f, t, c)][1] for f in (1, 2)]) for t in simple.TASKS]
        row["tasks_up"] = int(sum(d > 0 for d in per_task))
        row["tasks_down"] = int(sum(d < 0 for d in per_task))
        row["heads_kept_fold1"], row["heads_kept_fold2"] = sets[1][c], sets[2][c]
        summary.append(row)
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    rep = repeated(set(combos), sc, scanned)
    with (out_dir / "repeated.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["list", "fold", "task", "combo", "alpha", "reply", "share", "accuracy"])
        w.writeheader()
        w.writerows(rep)
    print(f"{'combo':8s} {'top5':5s} " + " ".join(f"{lst:>7s}" for lst in shown)
          + f" {'diff':>7s}  up/down  kept(f1,f2)  per fold s2 | {new}")
    for r in summary:
        print(f"{r['combo']:8s} {r['top5_of_fold']:5s} " + " ".join(f"{r[lst]:7.3f}" for lst in shown)
              + f" {r['diff']:+7.3f}  {r['tasks_up']:2d}/{r['tasks_down']:2d}    {r['heads_kept_fold1']},{r['heads_kept_fold2']}      "
              f"{r['stage2_fold1']:.3f} {r['stage2_fold2']:.3f} | {r[f'{new}_fold1']:.3f} {r[f'{new}_fold2']:.3f}")
    # Each fold's choice among its own five, scored on the other fold (Selected's rule: per task the better
    # alpha, mean over the 14 tasks), and the chosen combination's accuracy on the fold's own test half.
    for lst in shown:
        if lst == "stage2":
            continue
        d, stage, prefix = LISTS[lst]
        best_alpha = {}
        for fold in (1, 2):
            cells = defaultdict(lambda: defaultdict(list))
            for path in (d / f"fold{fold}").glob(f"patching_simple.part-{stage}-*.csv"):
                for r in csv.DictReader(path.open()):
                    if r["source"] == "task" and r["heads"].startswith(prefix):
                        cells[(r["task"], r["heads"][len(prefix):])][r["alpha"]].append(float(r["accuracy"]))
            per = defaultdict(list)
            for (t, c), by_alpha in cells.items():
                per[c].append(max(np.mean(v) for v in by_alpha.values()))
            best_alpha[fold] = {c: float(np.mean(v)) for c, v in per.items() if len(v) == len(simple.TASKS)}
        picks = {f: max(top5[f], key=lambda c: (best_alpha[3 - f][c], -len(c))) for f in (1, 2)}
        test = [float(np.mean([sc[lst][(f, t, picks[f])][1] for t in simple.TASKS])) for f in (1, 2)]
        print(f"{lst}: fold 1 picks {picks[1]} ({test[0]:.3f}), fold 2 picks {picks[2]} ({test[1]:.3f}); "
              f"mean {np.mean(test):.3f} (stage-2 Selected DBTML {next(r['stage2'] for r in summary if r['combo'] == 'DBTML'):.3f})")
    print(f"repeated replies (share >= 0.5): " + ", ".join(f"{lst} {sum(r['list'] == lst for r in rep)}" for lst in scanned)
          + f" ({out_dir.relative_to(RESULTS.parent)}/repeated.csv)")


if __name__ == "__main__":
    main()
