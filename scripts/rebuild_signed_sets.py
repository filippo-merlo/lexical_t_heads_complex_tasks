"""Signed screen (user 2026-11-22): Selected with features subtracted. No GPU.

Why: weights fitted to the TF-IDF stage-2 screen (offline, three ways: per-head causal score, head credit,
set accuracy from the mean features of the set) all give the query segment Q a negative weight (and O, other
words, a smaller one), while D, B and T stay positive. The screen cannot test this, since a combination can
only add features. Here a feature can also be subtracted.

Rule: a subtracted feature ``-X`` ranks heads by X from the bottom (rank 1025 - rank of X, ties averaged as in
``rebuild_sets.ranks``) and enters the combination's mean of ranks like any other feature; per task from the
task's own ranks, shared from the within-task ranks averaged over the 14 tasks and ranked again, as in
``rebuild_sets.build``. Ties to the lower layer, then head.

Candidates, at 40 heads, both folds (the TF-IDF lists, ``results/rebuild/tfidf_screen/fold<f>/features.npz``):
- per task: Selected DBTSPL minus every non-empty subset of the features it does not use, {O, M, Q} (7 sets);
- shared: Selected DBTM minus every non-empty subset of {O, S, P, Q, L} (31 sets);
- Selected itself (``signed_DBTSPL``, ``shared_signed_DBTM``), checked equal to the stored screen sets.
Selected's combinations are read from ``results/rebuild/tfidf_screen/selected.json`` and must be the same in
both folds.

Writes ``results/rebuild/signed/fold<f>/heads.json`` (``scripts/patch.py --set-list`` format; names
``signed_<combo>`` and ``shared_signed_<combo>``, combo as ``DBTSPL-MQ``) and ``results/rebuild/signed/sets.json``
(candidates, overlap of each set with Selected-40).

    .venv/bin/python scripts/rebuild_signed_sets.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import RESULTS, simple  # noqa: E402
from rebuild_sets import FEATURES, FOLDS, N_ALL, SCREEN_N, order_of, rankdata, ranks, top  # noqa: E402

SOURCE = RESULTS / "rebuild" / "tfidf_screen"
OUT = RESULTS / "rebuild" / "signed"
SCOPES = {"per task": ("screen_", "signed_"), "shared": ("shared_screen_", "shared_signed_")}


def subsets(features: list[str]) -> list[str]:
    return ["".join(c) for k in range(1, len(features) + 1) for c in itertools.combinations(features, k)]


def candidates(base: str) -> list[str]:
    """Selected, then Selected minus each non-empty subset of the features it does not use."""
    unused = [c for c in FEATURES if c not in base]
    return [base] + [f"{base}-{minus}" for minus in subsets(unused)]


def parse(combo: str) -> tuple[str, str]:
    plus, _, minus = combo.partition("-")
    return plus, minus


def combo_rank(single: dict[str, np.ndarray], rev: dict[str, np.ndarray], combo: str) -> np.ndarray:
    plus, minus = parse(combo)
    return np.mean([single[c] for c in plus] + [rev[c] for c in minus], axis=0)


def main() -> None:
    selected = json.loads((SOURCE / "selected.json").read_text())
    base = {scope: {selected[str(f)][scope]["combo"] for f in FOLDS} for scope in SCOPES}
    assert all(len(v) == 1 for v in base.values()), f"Selected differs between the folds: {base}"
    base = {scope: v.pop() for scope, v in base.items()}
    combos = {scope: candidates(base[scope]) for scope in SCOPES}
    report = {"selected": base, "candidates": combos, "overlap_with_selected": {}}
    for fold in FOLDS:
        z = np.load(SOURCE / f"fold{fold}" / "features.npz")
        stored = json.loads((SOURCE / f"fold{fold}" / "heads.json").read_text())["simple"]
        single = {t: {c: ranks(z[f"{t}/{c}"]) for c in FEATURES} for t in simple.TASKS}
        rev = {t: {c: rankdata(z[f"{t}/{c}"]) for c in FEATURES} for t in simple.TASKS}  # 1 = smallest
        for t in simple.TASKS:
            for c in FEATURES:
                assert np.allclose(rev[t][c], N_ALL + 1 - single[t][c]), (fold, t, c)
        shared_single = {c: rankdata(np.mean([single[t][c] for t in simple.TASKS], axis=0)) for c in FEATURES}
        shared_rev = {c: rankdata(np.mean([rev[t][c] for t in simple.TASKS], axis=0)) for c in FEATURES}
        shared = [["rule", SCOPES["shared"][1] + combo,
                   top(order_of(combo_rank(shared_single, shared_rev, combo)), SCREEN_N)] for combo in combos["shared"]]
        heads, overlap = {"simple": {}}, {"per task": {}, "shared": {}}
        for t in simple.TASKS:
            own = [["rule", SCOPES["per task"][1] + combo,
                    top(order_of(combo_rank(single[t], rev[t], combo)), SCREEN_N)] for combo in combos["per task"]]
            listed = {name: hs for _, name, hs in stored[t]["sets"]}
            # Check: Selected rebuilt here = the stored screen set of the same combination.
            assert own[0][2] == listed[SCOPES["per task"][0] + base["per task"]], (fold, t)
            assert shared[0][2] == listed[SCOPES["shared"][0] + base["shared"]], (fold, t)
            heads["simple"][t] = {"sets": own + shared}
            sel = {tuple(h) for h in own[0][2]}
            for (_, name, hs) in own:
                overlap["per task"].setdefault(name, []).append(len(sel & {tuple(h) for h in hs}))
        sel = {tuple(h) for h in shared[0][2]}
        overlap["shared"] = {name: len(sel & {tuple(h) for h in hs}) for _, name, hs in shared}
        overlap["per task"] = {name: float(np.mean(v)) for name, v in overlap["per task"].items()}
        report["overlap_with_selected"][str(fold)] = overlap
        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "heads.json").write_text(json.dumps(heads))
        print(f"fold {fold}: {len(combos['per task'])} per-task and {len(combos['shared'])} shared sets per task; "
              f"Selected matches the stored screen sets", flush=True)
    (OUT / "sets.json").write_text(json.dumps(report, indent=1))
    for scope in SCOPES:
        print(f"{scope}: overlap with Selected-40 (fold 1 / fold 2): " + ", ".join(
            f"{name.split('_')[-1]} {report['overlap_with_selected']['1'][scope][name]:.1f}/"
            f"{report['overlap_with_selected']['2'][scope][name]:.1f}"
            for name in report["overlap_with_selected"]["1"][scope]))


if __name__ == "__main__":
    main()
