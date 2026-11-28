"""One shared combination for every size (user 2026-11-28): chosen by cross-fit on the quick passes of the shared
attention screens (``results/rebuild/additions`` at 40 heads, ``results/rebuild/shared_n<N>`` at the other sizes),
and its sets at the sizes where it was not patched in full. No GPU.

- **Choice.** On each fold, a combination's shortfall at size N is its quick-pass score (first-token accuracy at
  alpha 2 on 10 test items per task, mean over the template means and the 14 tasks) minus the best quick-pass
  score of that fold and size; its general score is the mean shortfall over the seven sizes. Fold f takes the
  combination with the best general score on fold 3 - f (ties to fewer features, then combination order).
- **Sets.** The pick's shared sets at ``--sizes`` heads, ranked as the shared screen ranks them (each feature's
  within-task ranks averaged over the 14 tasks and ranked again, a combination the mean of its features' ranks,
  ``results/rebuild/additions/fold<f>/features.npz``); checked equal to the stored screen set of the same size
  (``shared_screen_<combo>``). Named ``shared_general<N>``, with ``N_RANDOM`` random sets of the same layer counts
  (``rebuild_sets.random_sets``, key ``general_n/<fold>/shared/shared_general<N>``), listed under every task.

Writes ``results/rebuild/general_n/fold<f>/heads.json`` and ``results/rebuild/general_n/selected.json``. Patched by
``scripts/general_patch.sh`` (control means; random sets with task means only), reported by
``scripts/rebuild_report.py --dir general_n --stage gen``.

    .venv/bin/python scripts/general_sets.py --sizes 10 40 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rebuild_screen_n as rs  # noqa: E402
import rebuild_shortlist as rsl  # noqa: E402
from lth import RESULTS, simple  # noqa: E402
from rebuild_sets import COMBOS, FEATURES, FOLDS, N_RANDOM, SIZES, order_of, random_sets, rankdata, ranks, top  # noqa: E402

R = RESULTS / "rebuild"
OUT = R / "general_n"


def quick(n: int) -> dict[int, dict[str, float]]:
    if n == 40:
        return {f: rsl.quick_scores(f, "additions")["shared"] for f in FOLDS}
    return rs.scores(n, "quick", "first_token", 2.0)


def stored(fold: int, n: int, combo: str) -> list[list[int]]:
    d = R / "additions" if n == 40 else R / f"shared_n{n}"
    sets = json.loads((d / f"fold{fold}" / "heads.json").read_text())["simple"][simple.TASKS[0]]["sets"]
    (heads,) = [hs for _, nm, hs in sets if nm == f"shared_screen_{combo}"]
    return heads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    args = parser.parse_args()
    rs.ADD, rs.PREFIX, rs.FEATS, rs.COMBS = rs.POOLS["additions"]
    q = {n: quick(n) for n in SIZES}
    assert all(len(q[n][f]) == len(COMBOS) for n in SIZES for f in FOLDS)
    general = {f: {c: float(np.mean([q[n][f][c] - max(q[n][f].values()) for n in SIZES])) for c in COMBOS} for f in FOLDS}
    report = {}
    for fold in FOLDS:
        other = 3 - fold
        ranked = sorted(COMBOS, key=lambda c: (-general[other][c], len(c), COMBOS.index(c)))
        report[str(fold)] = dict(combo=ranked[0], top10_other_fold=[[c, round(general[other][c], 4)] for c in ranked[:10]])
        print(f"fold {fold}: picks {ranked[0]} on fold {other} (mean shortfall {general[other][ranked[0]]:+.3f}); "
              f"next {', '.join(ranked[1:5])}", flush=True)
    for fold in FOLDS:
        combo = report[str(fold)]["combo"]
        z = np.load(R / "additions" / f"fold{fold}" / "features.npz")
        single = {t: {c: ranks(z[f"{t}/{c}"]) for c in FEATURES} for t in simple.TASKS}
        shared_single = {c: rankdata(np.mean([single[t][c] for t in simple.TASKS], axis=0)) for c in FEATURES}
        order = order_of(np.mean([shared_single[c] for c in combo], axis=0))
        entries = []
        for n in args.sizes:
            heads = top(order, n)
            assert heads == stored(fold, n, combo), (fold, n, combo)
            label = f"shared_general{n}"
            entries += [["rule", label, heads]] + [["rule", f"random_{label}_{i}", r]
                                                   for i, r in enumerate(random_sets(heads, f"general_n/{fold}/shared/{label}"))]
        assert len(entries) == len(args.sizes) * (N_RANDOM + 1)
        d = OUT / f"fold{fold}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "heads.json").write_text(json.dumps({"simple": {t: {"sets": entries} for t in simple.TASKS}}))
        print(f"fold {fold}: {combo} at {args.sizes} heads equal to the stored screen sets; written with random sets", flush=True)
    report["sizes"] = args.sizes
    (OUT / "selected.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
