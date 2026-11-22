"""Shared combination screen at another size (user 2026-11-22: 80 heads): the stage-2 screen of all 511
combinations of the nine features, shared scope only, with sets of ``--n`` heads instead of 40, in two stages as
``scripts/rebuild_quick_screen.sh`` (runner: ``scripts/rebuild_screen_n.sh``). No GPU.

- ``sets``: every combination's shared set of N heads, ranked as ``rebuild_sets.build`` ranks the shared screen
  (each feature's within-task ranks averaged over the 14 tasks and ranked again, a combination the mean of its
  features' ranks) from the stage-2 features (``results/rebuild/additions/fold<f>/features.npz``); check: at 40
  heads the same code gives the stored ``shared_screen_<combo>`` sets. Listed under every task, as stage 2 does.
- ``shortlist``: quick pass (alpha 2, the seeded 10 test items per task of ``patch.py --n-items``, one-token
  replies): a combination's score on a fold is its first-token accuracy averaged over the template means, then
  over the 14 tasks; fold f's shortlist is the 10 best on fold 3 - f (ties to fewer features, then combination
  order), as ``rebuild_shortlist.py``.
- ``select``: full pass of the union of both shortlists (alphas 1 and 2, all test items, replies of at most 16
  tokens): fold f chooses among its shortlist the combination best on fold 3 - f (per task the better alpha of
  the template-mean accuracy, mean over tasks; ties to fewer features), as ``rebuild_select.py``. Reports each
  fold's choice on its own test half with the alpha cross-fitted (``rebuild_report.py``), and every shortlisted
  combination's accuracy (after the fact). Then adds each fold's pick to its ``heads.json`` as
  ``shared_selected<N>`` with ``N_RANDOM`` random sets of the same layer counts (``rebuild_sets.random_sets``, key
  ``shared_n<N>/<fold>/shared/shared_selected<N>``), which the runner patches with control means (random sets with
  task means only), as stage 2's phase B; ``scripts/rebuild_report.py --dir shared_n<N> --stage sel`` reports them.

Writes ``results/rebuild/shared_n<N>/``: ``fold<f>/heads.json``, ``shortlist.json``, ``selected.json``,
``summary.csv``.

    .venv/bin/python scripts/rebuild_screen_n.py sets --n 80
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rebuild_report  # noqa: E402
from lth import RESULTS, simple  # noqa: E402
from rebuild_sets import COMBOS, FEATURES, FOLDS, N_RANDOM, order_of, rankdata, random_sets, ranks, top  # noqa: E402

K = 10
ADD = RESULTS / "rebuild" / "additions"


def out_dir(n: int) -> Path:
    return RESULTS / "rebuild" / f"shared_n{n}"


def shared_sets(fold: int, n: int) -> dict[str, list[list[int]]]:
    z = np.load(ADD / f"fold{fold}" / "features.npz")
    single = {t: {c: ranks(z[f"{t}/{c}"]) for c in FEATURES} for t in simple.TASKS}
    shared_single = {c: rankdata(np.mean([single[t][c] for t in simple.TASKS], axis=0)) for c in FEATURES}
    return {c: top(order_of(np.mean([shared_single[x] for x in c], axis=0)), n) for c in COMBOS}


def cmd_sets(n: int) -> None:
    for fold in FOLDS:
        stored = {nm: hs for _, nm, hs in json.loads((ADD / f"fold{fold}" / "heads.json").read_text())["simple"][simple.TASKS[0]]["sets"]}
        at40 = shared_sets(fold, 40)
        assert all(at40[c] == stored[f"shared_screen_{c}"] for c in COMBOS), fold
        sets = shared_sets(fold, n)
        entries = [["rule", f"shared_screen_{c}", sets[c]] for c in COMBOS]
        d = out_dir(n) / f"fold{fold}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "heads.json").write_text(json.dumps({"simple": {t: {"sets": entries} for t in simple.TASKS}}))
        distinct = len({frozenset(map(tuple, hs)) for hs in sets.values()})
        print(f"fold {fold}: check at 40 heads passed; {len(COMBOS)} shared sets of {n} heads ({distinct} distinct)", flush=True)


def scores(n: int, part: str, column: str, alpha: float | None) -> dict[int, dict[str, float]]:
    """score[fold][combo]: per task the template-mean value (at ``alpha``, or the better alpha), mean over tasks."""
    out = {}
    for fold in FOLDS:
        cells = defaultdict(lambda: defaultdict(list))
        for path in sorted((out_dir(n) / f"fold{fold}").glob(f"patching_simple.part-{part}-*.csv")):
            for r in csv.DictReader(path.open()):
                if r["source"] == "task" and r["heads"].startswith("shared_screen_") and (alpha is None or float(r["alpha"]) == alpha):
                    cells[(r["task"], r["heads"][len("shared_screen_"):])][r["alpha"]].append(float(r[column]))
        per = defaultdict(list)
        for (t, c), by_alpha in cells.items():
            per[c].append(max(np.mean(v) for v in by_alpha.values()))
        out[fold] = {c: float(np.mean(v)) for c, v in per.items() if len(v) == len(simple.TASKS)}
    return out


def cmd_shortlist(n: int) -> None:
    q = scores(n, "quick", "first_token", 2.0)
    assert all(len(q[f]) == len(COMBOS) for f in FOLDS), {f: len(q[f]) for f in FOLDS}
    report = {}
    for fold in FOLDS:
        other = 3 - fold
        ranked = sorted(COMBOS, key=lambda c: (-q[other][c], len(c), COMBOS.index(c)))[:K]
        report[str(fold)] = [[c, round(q[other][c], 4)] for c in ranked]
        print(f"fold {fold} shortlist (quick pass of fold {other}): {', '.join(ranked)}", flush=True)
    report["union"] = sorted({c for f in FOLDS for c, _ in report[str(f)]}, key=COMBOS.index)
    (out_dir(n) / "shortlist.json").write_text(json.dumps(report, indent=1))
    for c in report["union"]:
        print(f"shared_screen_{c}")


def cmd_select(n: int) -> None:
    short = json.loads((out_dir(n) / "shortlist.json").read_text())
    full = scores(n, "full", "accuracy", None)
    rebuild_report.DIR = out_dir(n)
    cf = rebuild_report.crossfit(rebuild_report.load("full"))
    test = lambda fold, c: float(np.mean([cf[(fold, t, f"shared_screen_{c}", "task")][1] for t in simple.TASKS]))  # noqa: E731
    report, rows = {}, []
    for fold in FOLDS:
        other = 3 - fold
        cands = [c for c, _ in short[str(fold)]]
        pick = min(cands, key=lambda c: (-full[other][c], len(c), COMBOS.index(c)))
        report[str(fold)] = dict(combo=pick, score_other_fold=round(full[other][pick], 4), test=round(test(fold, pick), 4),
                                 candidates=[[c, round(full[other][c], 4)] for c in sorted(cands, key=lambda c: -full[other][c])])
        print(f"fold {fold}: picks {pick} (fold {other} {full[other][pick]:.3f}); its accuracy on fold {fold}'s test half {test(fold, pick):.3f}", flush=True)
    report["selected_test_mean"] = round(float(np.mean([report[str(f)]["test"] for f in FOLDS])), 4)
    for fold in FOLDS:  # the pick and its random sets, for the run with control means
        path = out_dir(n) / f"fold{fold}" / "heads.json"
        heads = json.loads(path.read_text())
        listed = {nm: hs for _, nm, hs in heads["simple"][simple.TASKS[0]]["sets"]}
        sel = listed[f"shared_screen_{report[str(fold)]['combo']}"]
        label = f"shared_selected{n}"
        add = [["rule", label, sel]] + [["rule", f"random_{label}_{i}", r] for i, r in
                                        enumerate(random_sets(sel, f"shared_n{n}/{fold}/shared/{label}"))]
        assert len(add) == N_RANDOM + 1
        for t in simple.TASKS:
            sets = [e for e in heads["simple"][t]["sets"] if "selected" not in e[1]]
            heads["simple"][t]["sets"] = sets + add
        path.write_text(json.dumps(heads))
    (out_dir(n) / "selected.json").write_text(json.dumps(report, indent=1))
    for c in short["union"]:
        rows.append(dict(combo=c, shortlisted=",".join(str(f) for f in FOLDS if c in [x for x, _ in short[str(f)]]),
                         selected=",".join(str(f) for f in FOLDS if report[str(f)]["combo"] == c),
                         accuracy=round(float(np.mean([test(f, c) for f in FOLDS])), 4),
                         fold1=round(test(1, c), 4), fold2=round(test(2, c), 4)))
    rows.sort(key=lambda r: -r["accuracy"])
    with (out_dir(n) / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Selected at {n} heads (each fold's pick on its own test half, mean): {report['selected_test_mean']:.3f}")
    for r in rows:
        print(f"   {r['combo']:9s} {r['accuracy']:.3f} ({r['fold1']:.3f} / {r['fold2']:.3f}) shortlist {r['shortlisted']:4s} picked {r['selected']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=("sets", "shortlist", "select"))
    parser.add_argument("--n", type=int, required=True)
    args = parser.parse_args()
    {"sets": cmd_sets, "shortlist": cmd_shortlist, "select": cmd_select}[args.step](args.n)


if __name__ == "__main__":
    main()
