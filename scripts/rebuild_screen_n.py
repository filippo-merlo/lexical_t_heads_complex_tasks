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
``summary.csv``. ``report``: checks that the pick's rows of the run with control means equal its rows of the full
pass (same heads, same options), and prints its accuracy with alpha cross-fitted by rule (``rebuild_report.py``,
as reported) and by combination (the other fold's runs of the same combination; the two differ only when the
folds pick different combinations).

Stage 3 with the shared Selected chosen at each size (user 2026-11-23), ``--pool``:

- ``examples`` (Sec. 6.2, re-identified on example-based prompts): the 127 combinations of the example pool's
  seven features (``results/rebuild/examples/fold<f>/features.npz``) at N heads, the same two stages with the
  P-format means; the runner then patches the pick with the Q-format means (part test; control means, random
  sets) and sums it (part sum, ``scripts/e26_summed.py``, Q-format means). ``results/rebuild/examples_n<N>/``.
- ``wordings`` (Sec. 6.1, re-identified on Yang et al.'s five templates and the P wordings): step
  ``candidates``: fold f's candidates are the combinations among which stage 2 chose at N (fold f's shortlist,
  ``shared_n<N>/selected.json``), as 3.2 at 40 heads takes stage 2's ten best; rebuilt on the wordings pool's
  features, patched in full with that pool's means (no quick pass), chosen as above; the pick is patched with
  the Q-wording means (part test). ``results/rebuild/wordings_n<N>/``.
- ``transfer`` (Secs. 6.1, 6.2, 6.3, the stage-2 sets on other prompts): step ``transfer-sets`` lists every
  size's stage-2 shared pick with its random sets (all sizes with a screen but 40, whose pick stage 3 already
  tests as shared_selected40) under every task of ``results/rebuild/transfer_n/fold<f>/heads.json``; the runner
  patches them with the Q-wording means (part wt), the Q-format means (part et) and summed with Yang et al.'s
  five template means (part tsum).

Each pool's sets are checked first: at 40 heads the same code gives the pool's stored ``shared_screen_<combo>``.

    .venv/bin/python scripts/rebuild_screen_n.py sets --n 80
    .venv/bin/python scripts/rebuild_screen_n.py sets --n 80 --pool examples
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
from rebuild_sets import (COMBOS, FEATURES, FOLDS, N_RANDOM, SCREEN_N, SIZES, order_of, rankdata,  # noqa: E402
                          random_sets, ranks, top)
from rebuild_sets_s3 import EX_FEATURES, combos  # noqa: E402

K = 10
R = RESULTS / "rebuild"
POOLS = {"additions": (R / "additions", "shared_n", FEATURES, COMBOS),  # features and stored sets, output prefix
         "wordings": (R / "wordings", "wordings_n", FEATURES, COMBOS),
         "examples": (R / "examples", "examples_n", EX_FEATURES, combos(EX_FEATURES))}
ADD, PREFIX, FEATS, COMBS = POOLS["additions"]
TRANSFER = R / "transfer_n"


def out_dir(n: int) -> Path:
    return R / f"{PREFIX}{n}"


def shared_sets(fold: int, n: int) -> dict[str, list[list[int]]]:
    z = np.load(ADD / f"fold{fold}" / "features.npz")
    single = {t: {c: ranks(z[f"{t}/{c}"]) for c in FEATS} for t in simple.TASKS}
    shared_single = {c: rankdata(np.mean([single[t][c] for t in simple.TASKS], axis=0)) for c in FEATS}
    return {c: top(order_of(np.mean([shared_single[x] for x in c], axis=0)), n) for c in COMBS}


def cmd_sets(n: int) -> None:
    for fold in FOLDS:
        stored = {nm: hs for _, nm, hs in json.loads((ADD / f"fold{fold}" / "heads.json").read_text())["simple"][simple.TASKS[0]]["sets"]}
        at40 = shared_sets(fold, 40)
        assert all(at40[c] == stored[f"shared_screen_{c}"] for c in COMBS), fold
        sets = shared_sets(fold, n)
        entries = [["rule", f"shared_screen_{c}", sets[c]] for c in COMBS]
        d = out_dir(n) / f"fold{fold}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "heads.json").write_text(json.dumps({"simple": {t: {"sets": entries} for t in simple.TASKS}}))
        distinct = len({frozenset(map(tuple, hs)) for hs in sets.values()})
        print(f"fold {fold}: check at 40 heads passed; {len(COMBS)} shared sets of {n} heads ({distinct} distinct)", flush=True)


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
    assert all(len(q[f]) == len(COMBS) for f in FOLDS), {f: len(q[f]) for f in FOLDS}
    report = {}
    for fold in FOLDS:
        other = 3 - fold
        ranked = sorted(COMBS, key=lambda c: (-q[other][c], len(c), COMBS.index(c)))[:K]
        report[str(fold)] = [[c, round(q[other][c], 4)] for c in ranked]
        print(f"fold {fold} shortlist (quick pass of fold {other}): {', '.join(ranked)}", flush=True)
    report["union"] = sorted({c for f in FOLDS for c, _ in report[str(f)]}, key=COMBS.index)
    (out_dir(n) / "shortlist.json").write_text(json.dumps(report, indent=1))
    for c in report["union"]:
        print(f"shared_screen_{c}")


def cmd_candidates(n: int) -> None:
    """Wordings pool: fold f's candidates are stage 2's shortlist of fold f at this size, written as the shortlist."""
    stage2 = json.loads((R / f"shared_n{n}" / "selected.json").read_text())
    report = {str(f): stage2[str(f)]["candidates"] for f in FOLDS}
    assert all(len(report[str(f)]) == K for f in FOLDS)
    report["union"] = sorted({c for f in FOLDS for c, _ in report[str(f)]}, key=COMBS.index)
    (out_dir(n) / "shortlist.json").write_text(json.dumps(report, indent=1))
    for fold in FOLDS:
        print(f"fold {fold} candidates (stage 2's shortlist at {n} heads): {', '.join(c for c, _ in report[str(fold)])}", flush=True)
    for c in report["union"]:
        print(f"shared_screen_{c}")


def cmd_transfer_sets(_n: int) -> None:
    sizes = [n for n in SIZES if n != SCREEN_N and (R / f"shared_n{n}" / "selected.json").exists()]
    for fold in FOLDS:
        entries = []
        for n in sizes:
            sets = json.loads((R / f"shared_n{n}" / f"fold{fold}" / "heads.json").read_text())["simple"]
            first = [e for e in sets[simple.TASKS[0]]["sets"] if e[1] == f"shared_selected{n}" or e[1].startswith(f"random_shared_selected{n}_")]
            assert len(first) == N_RANDOM + 1 and all(
                [e for e in sets[t]["sets"] if e in first] == first for t in simple.TASKS), (fold, n)
            entries += first
        d = TRANSFER / f"fold{fold}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "heads.json").write_text(json.dumps({"simple": {t: {"sets": entries} for t in simple.TASKS}}))
    print(f"transfer sets: shared Selected and its random sets at {sizes} heads, both folds", flush=True)


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
        pick = min(cands, key=lambda c: (-full[other][c], len(c), COMBS.index(c)))
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
                                        enumerate(random_sets(sel, f"{PREFIX}{n}/{fold}/shared/{label}"))]
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


def cmd_report(n: int) -> None:
    sel = json.loads((out_dir(n) / "selected.json").read_text())
    rebuild_report.DIR = out_dir(n)
    full, run = rebuild_report.load("full"), rebuild_report.load("sel")
    by_rule = rebuild_report.crossfit(run)
    label = f"shared_selected{n}"
    for fold in FOLDS:
        combo, acc_rule, acc_combo = sel[str(fold)]["combo"], [], []
        for t in simple.TASKS:
            for alpha, a in run[(fold, t, label, "task")].items():  # same heads, same options: same rows
                assert abs(a - full[(fold, t, f"shared_screen_{combo}", "task")][alpha]) < 1e-9, (fold, t, alpha)
            other = full[(3 - fold, t, f"shared_screen_{combo}", "task")]
            alpha = max(sorted(other), key=lambda a: (other[a], -a))  # ties to the smaller alpha, as rebuild_report
            acc_combo.append(run[(fold, t, label, "task")][alpha])
            acc_rule.append(by_rule[(fold, t, label, "task")][1])
        print(f"{n} heads, fold {fold} ({combo}): alpha by rule {np.mean(acc_rule):.3f}, by combination {np.mean(acc_combo):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=("sets", "shortlist", "candidates", "select", "report", "transfer-sets"))
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--pool", choices=tuple(POOLS), default="additions")
    args = parser.parse_args()
    global ADD, PREFIX, FEATS, COMBS
    ADD, PREFIX, FEATS, COMBS = POOLS[args.pool]
    assert args.n or args.step == "transfer-sets", "--n"
    {"sets": cmd_sets, "shortlist": cmd_shortlist, "candidates": cmd_candidates, "select": cmd_select,
     "report": cmd_report, "transfer-sets": cmd_transfer_sets}[args.step](args.n)


if __name__ == "__main__":
    main()
