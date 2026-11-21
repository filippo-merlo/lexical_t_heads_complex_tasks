"""Rebuilt design (paper/head-sets-simple-design.tex, Sec. 5.4): pick Selected from the combination screen
by cross-fitting and add its sets at every size to both folds' ``heads.json``. No GPU.

Screen accuracies: ``results/rebuild/additions/fold{1,2}/patching_simple.part-s2-*.csv``
(``scripts/rebuild_stage2.sh``), check 1, task means only, alphas 1 and 2. A candidate's score on a
fold is, for every task, its accuracy averaged over the template means at the better alpha of that
fold and task, then averaged over the 14 tasks. For fold 1, Selected is the candidate with the best
score on fold 2 (whose test items are fold 1's choosing half), and the reverse; separately per task
(one combination for all tasks, each task ranking heads by its own features) and shared. Ties go to
fewer features, then to the order of the combinations.

Selected's sets are rebuilt from ``features.npz`` exactly as the screen built them (checked against
the screen set at 40 heads) at every size, with their random sets (``rebuild_sets.random_sets``), and
written into ``heads.json`` as ``selected<N>`` / ``shared_selected<N>`` (earlier Selected entries are
replaced). Writes ``selected.json`` (the combinations, their scores, the ten best candidates per scope
and fold).

Stage 3 (Secs. 6.1-6.2): ``--pool wordings`` or ``--pool examples`` does the same for the rules re-chosen
on those pools (``scripts/rebuild_sets_s3.py``), from their screens (``results/rebuild/<pool>/fold{1,2}/
patching_simple.part-s3<w|e>-screen-*.csv``, patched with the pool's own means).

    .venv/bin/python scripts/rebuild_select.py
    .venv/bin/python scripts/rebuild_select.py --pool examples
    .venv/bin/python scripts/rebuild_select.py --pool wordings --candidates 5

``--candidates K`` (stage 3.2, user 2026-11-18): instead of all combinations, fold f chooses among the K
best stage-2 combinations of its scope (``results/rebuild/additions/selected.json``, ranked on the other
fold, so Selected first); the screen then needs only the union of both folds' candidates.

``--shortlist`` (two-stage screen, scripts/rebuild_quick_screen.sh): fold f chooses among its shortlist in
``results/rebuild/<pool>/shortlist.json`` (``scripts/rebuild_shortlist.py``, the best combinations of fold
3 - f's quick pass), scored on the full pass of those combinations as above. ``--dry-run`` prints the
choice and writes nothing (used to check the two-stage screen against the full one).

    .venv/bin/python scripts/rebuild_select.py --pool additions --shortlist --dry-run
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

from lth import simple  # noqa: E402
from rebuild_sets import (COMBOS, FEATURES, FOLDS, OUT, SCREEN_N, SIZES, order_of, random_sets,  # noqa: E402
                          rankdata, ranks, top)

POOLS = {"additions": ("patching_simple.part-s2-*.csv", ""), "wordings": ("patching_simple.part-s3w-screen-*.csv", "wordings/"),
         "examples": ("patching_simple.part-s3e-screen-*.csv", "examples/")}  # screen files, random-set key prefix


def features_combos(pool: str):
    if pool == "examples":
        from rebuild_sets_s3 import EX_FEATURES, combos
        return EX_FEATURES, combos(EX_FEATURES)
    return FEATURES, COMBOS

SCOPES = {"per task": "screen_", "shared": "shared_screen_"}


def screen_scores(fold: int, pool: str = "additions") -> dict[str, dict[str, float]]:
    """score[scope][combo] on ``fold``: per task best alpha of the template-mean accuracy, mean over tasks."""
    _, combos_ = features_combos(pool)
    cells = defaultdict(lambda: defaultdict(list))  # (task, set) -> alpha -> accuracies of template means
    for path in sorted((OUT.parent / pool / f"fold{fold}").glob(POOLS[pool][0])):
        for r in csv.DictReader(path.open()):
            if r["source"] == "task" and "screen_" in r["heads"]:
                cells[(r["task"], r["heads"])][float(r["alpha"])].append(float(r["accuracy"]))
    out = {}
    for scope, prefix in SCOPES.items():
        out[scope] = {}
        for combo in combos_:
            per_task = []
            for t in simple.TASKS:
                by_alpha = cells.get((t, prefix + combo))
                if not by_alpha:
                    break
                per_task.append(max(np.mean(v) for v in by_alpha.values()))
            if len(per_task) == len(simple.TASKS):
                out[scope][combo] = float(np.mean(per_task))
    return out


def best(scores: dict[str, float], combos_: list[str]) -> str:
    return min(scores, key=lambda c: (-scores[c], len(c), combos_.index(c)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", choices=tuple(POOLS), default="additions")
    parser.add_argument("--candidates", type=int, default=None,
                        help="choose among the K best stage-2 combinations of each fold and scope (default: all)")
    parser.add_argument("--shortlist", action="store_true",
                        help="choose among the shortlist of the quick pass (<pool>/shortlist.json)")
    parser.add_argument("--dry-run", action="store_true", help="print the choice, write nothing")
    args = parser.parse_args()
    assert not (args.candidates and args.shortlist), "--candidates or --shortlist, not both"
    stage2 = json.loads((OUT / "selected.json").read_text()) if args.candidates else None
    short = json.loads((OUT.parent / args.pool / "shortlist.json").read_text()) if args.shortlist else None

    def candidates(fold: int, scope: str) -> list[str]:
        if short is not None:
            return [c for c, _ in short[str(fold)][scope]]
        if stage2 is None:
            return combos_
        return [c for c, _ in stage2[str(fold)][scope]["top10_other_fold"][:args.candidates]]
    features, combos_ = features_combos(args.pool)
    base = OUT.parent / args.pool
    scores = {fold: screen_scores(fold, args.pool) for fold in FOLDS}
    for fold in FOLDS:
        for scope in SCOPES:
            missing = [c for f in FOLDS for c in candidates(f, scope) if c not in scores[fold][scope]]
            assert not missing, f"fold {fold} {scope}: {missing} without all 14 tasks"
    report = {}
    for fold in FOLDS:
        other = 3 - fold
        pick = {scope: {c: scores[other][scope][c] for c in candidates(fold, scope)} for scope in SCOPES}
        chosen = {scope: best(pick[scope], combos_) for scope in SCOPES}
        report[fold] = {scope: dict(
            combo=chosen[scope], features=[features[c][0] for c in chosen[scope]],
            score_other_fold=round(scores[other][scope][chosen[scope]], 4),
            score_own_fold=round(scores[fold][scope][chosen[scope]], 4),
            top10_other_fold=[[c, round(scores[other][scope][c], 4)]
                              for c in sorted(pick[scope], key=lambda c: (-pick[scope][c], len(c)))[:10]],
            **({"candidates": candidates(fold, scope)} if args.candidates or args.shortlist else {}))
            for scope in SCOPES}
        if args.dry_run:
            print(f"fold {fold}: " + "; ".join(f"{scope} {report[fold][scope]['combo']} "
                                               f"(fold {other} {report[fold][scope]['score_other_fold']})"
                                               for scope in SCOPES) + "  [dry run]", flush=True)
            continue

        out = base / f"fold{fold}"
        z = np.load(out / "features.npz")
        single = {t: {c: ranks(z[f"{t}/{c}"]) for c in features} for t in simple.TASKS}
        shared_single = {c: rankdata(np.mean([single[t][c] for t in simple.TASKS], axis=0)) for c in features}
        heads = json.loads((out / "heads.json").read_text())

        def entries(label: str, sets: list[list[int]], key: str) -> list:
            return [["rule", label, sets]] + [["rule", f"random_{label}_{i}", r]
                                               for i, r in enumerate(random_sets(sets, f"{POOLS[args.pool][1]}{fold}/{key}/{label}"))]

        shared_entries = []
        r = np.mean([shared_single[c] for c in chosen["shared"]], axis=0)
        for n in SIZES:
            shared_entries += entries(f"shared_selected{n}", top(order_of(r), n), "shared")
        for t in simple.TASKS:
            sets = [e for e in heads["simple"][t]["sets"] if "selected" not in e[1]]
            listed = {name: hs for _, name, hs in sets}
            rt = np.mean([single[t][c] for c in chosen["per task"]], axis=0)
            own = []
            for n in SIZES:
                own += entries(f"selected{n}", top(order_of(rt), n), t)
            at40 = {name: hs for _, name, hs in own + shared_entries}
            assert at40[f"selected{SCREEN_N}"] == listed[f"screen_{chosen['per task']}"], (fold, t)
            assert at40[f"shared_selected{SCREEN_N}"] == listed[f"shared_screen_{chosen['shared']}"], (fold, t)
            heads["simple"][t]["sets"] = sets + own + shared_entries
        (out / "heads.json").write_text(json.dumps(heads))
        print(f"fold {fold}: " + "; ".join(f"{scope} {report[fold][scope]['combo']} "
                                           f"(fold {other} {report[fold][scope]['score_other_fold']})"
                                           for scope in SCOPES), flush=True)
    if not args.dry_run:
        (base / "selected.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
