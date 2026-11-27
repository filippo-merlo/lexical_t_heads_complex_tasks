"""Two-stage screen, step 2 (scripts/rebuild_quick_screen.sh): the shortlist of combinations from the quick
pass. No GPU.

Quick pass: ``results/rebuild/<pool>/fold{1,2}/patching_simple.part-<quick>-<task>.csv``, every combination's
set at 40 heads patched at alpha 2 into a seeded sample of 10 test items per task, with replies of one token,
task means only. A combination's quick score on a fold is its first-token accuracy averaged over the
template (or format) means, then over the 14 tasks. As ``scripts/rebuild_select.py`` chooses fold f's
combination on fold 3 - f, fold f's shortlist is the K best combinations on fold 3 - f's quick pass, per
scope; ties go to fewer features, then to the order of the combinations.

Writes ``results/rebuild/<pool>/shortlist.json``: ``{fold: {scope: [[combo, quick score], ...]}}`` and,
under ``"union"``, each scope's combinations over both folds, which the full pass patches on both folds.

    .venv/bin/python scripts/rebuild_shortlist.py --pool additions
    .venv/bin/python scripts/rebuild_shortlist.py --pool examples --k 10
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
from rebuild_select import SCOPES, features_combos  # noqa: E402
from rebuild_sets import FOLDS, OUT  # noqa: E402

QUICK = {"additions": "quick-s2", "examples": "quick-s3e", "tfidf_screen": "quick-s2", "lrp_screen": "quick-s2"}  # part name of the quick pass (no stage prefix matches it)
QUICK_ALPHA = 2.0


def quick_paths(fold: int, pool: str) -> list[Path]:
    return sorted((OUT.parent / pool / f"fold{fold}").glob(f"patching_simple.part-{QUICK[pool]}-*.csv"))


def has_quick(pool: str = "additions") -> bool:
    """Whether the screen of ``pool`` ran in two stages (a quick pass in both folds)."""
    return all(quick_paths(fold, pool) for fold in FOLDS)


def quick_per_task(fold: int, pool: str) -> dict[str, dict[str, dict[str, float]]]:
    """score[scope][combo][task] on ``fold``: first-token accuracy at alpha 2, mean over the means."""
    _, combos_ = features_combos(pool)
    cells = defaultdict(list)  # (task, set) -> first-token accuracy of each mean
    paths = quick_paths(fold, pool)
    assert paths, f"no quick pass for {pool} fold {fold}"
    for path in paths:
        for r in csv.DictReader(path.open()):
            if r["source"] == "task" and "screen_" in r["heads"] and float(r["alpha"]) == QUICK_ALPHA:
                cells[(r["task"], r["heads"])].append(float(r["first_token"]))
    out = {}
    for scope, prefix in SCOPES.items():
        out[scope] = {}
        for combo in combos_:
            per_task = {t: float(np.mean(cells[(t, prefix + combo)])) for t in simple.TASKS if cells.get((t, prefix + combo))}
            assert len(per_task) == len(simple.TASKS), f"fold {fold} {scope} {combo}: {len(per_task)} tasks"
            out[scope][combo] = per_task
    return out


def quick_scores(fold: int, pool: str) -> dict[str, dict[str, float]]:
    """score[scope][combo] on ``fold``: first-token accuracy at alpha 2, mean over the means, then the tasks."""
    return {scope: {c: float(np.mean(list(v.values()))) for c, v in d.items()}
            for scope, d in quick_per_task(fold, pool).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", choices=tuple(QUICK), default="additions")
    parser.add_argument("--k", type=int, default=10, help="combinations per fold and scope (default 10)")
    args = parser.parse_args()
    _, combos_ = features_combos(args.pool)
    scores = {fold: quick_scores(fold, args.pool) for fold in FOLDS}
    report: dict = {}
    for fold in FOLDS:
        other = 3 - fold
        report[str(fold)] = {}
        for scope in SCOPES:
            ranked = sorted(combos_, key=lambda c: (-scores[other][scope][c], len(c), combos_.index(c)))[:args.k]
            report[str(fold)][scope] = [[c, round(scores[other][scope][c], 4)] for c in ranked]
            print(f"fold {fold} {scope} (quick pass of fold {other}): " + ", ".join(c for c in ranked), flush=True)
    report["union"] = {scope: sorted({c for f in FOLDS for c, _ in report[str(f)][scope]}, key=combos_.index)
                       for scope in SCOPES}
    report["k"] = args.k
    path = OUT.parent / args.pool / "shortlist.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"{path}: union " + "; ".join(f"{s} {len(report['union'][s])}" for s in SCOPES))


if __name__ == "__main__":
    main()
