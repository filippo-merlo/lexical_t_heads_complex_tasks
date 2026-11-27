"""LRP-weighted combination screen (user 2026-11-27): the stage-2 screen of paper/head-sets-simple-design.tex
Sec. 5.4 with every attention feature replaced by its AttnLRP twin (Pham et al. 2026). No GPU.

Features, per head, from the choosing half's identification pool (``rebuild_sets.pool``) only:

- **D** unchanged (``results/rebuild/additions/fold<f>/features.npz``): it reads the head's decoded tokens, not its
  attention, so it has no LRP twin.
- **B, T, O, S, P, M, Q, L**: the AttnLRP twins stored by ``scripts/lrp_sets.py`` in ``lrp_features.npz``: on
  each segment, the relevance of the last token's attention weights for the logit of the answer's first token,
  ``max(0, A[n, j] * dlogit/dA[n, j])`` per key, summed over the segment's keys and averaged over the pool prompts
  where the segment is not empty (B: the relevance on every position but the first). T is LRP-T.

Sets exactly as ``rebuild_sets.build`` builds the screen (through ``rebuild_tfidf_sets.screen_sets``): every one
of the 511 non-empty combinations at 40 heads; per task (``screen_<combo>``) a combination averages its
features' within-task ranks; shared (``shared_screen_<combo>``) each feature's within-task ranks are averaged
over the 14 tasks and ranked again, then averaged over the combination's features.

Checks before anything is written, per fold:
(a) the same builder fed the attention features reproduces every stored stage-2 screen set
    (``results/rebuild/additions/fold<f>/heads.json``, 1,022 per task);
(b) with the LRP features, ``screen_D`` equals the stored stage-2 ``screen_D``, and ``screen_T`` /
    ``shared_screen_T`` equal the patched LRP-T sets ``lrpT40`` / ``shared_lrpT40`` (``scripts/lrp_sets.py``).

Writes ``results/rebuild/lrp_screen/fold<f>/``: ``heads.json`` (the 1,022 screen sets of each task, the
``scripts/patch.py --set-list`` format), ``features.npz`` (the nine features used, ``<task>/<code>``, for
``scripts/rebuild_select.py``), ``screen.json`` (distinct sets per scope) and ``sets.csv`` (per combination and
scope: median layer and heads in common with the attention screen's set of the same combination).
Patch and choose with the two-stage screen: ``scripts/rebuild_quick_screen.sh lrp_screen all GPU...``.

    .venv/bin/python scripts/lrp_screen_sets.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import RESULTS, simple  # noqa: E402
from rebuild_sets import COMBOS, FEATURES, FOLDS  # noqa: E402
from rebuild_tfidf_sets import screen_sets  # noqa: E402

ADD = RESULTS / "rebuild" / "additions"
OUT = RESULTS / "rebuild" / "lrp_screen"
LRP_CODES = [c for c in FEATURES if c != "D"]


def as_sets(entries: list) -> dict[str, list]:
    return {name: heads for _, name, heads in entries}


def main() -> None:
    assert len(COMBOS) == 511 and list(FEATURES) == ["D", "B", "T", "O", "S", "P", "M", "Q", "L"]
    for fold in FOLDS:
        att = np.load(ADD / f"fold{fold}" / "features.npz")
        lrp = np.load(ADD / f"fold{fold}" / "lrp_features.npz")
        stored = json.loads((ADD / f"fold{fold}" / "heads.json").read_text())["simple"]

        # (a) the builder reproduces the stage-2 screen from the attention features.
        att_feats = {t: {c: att[f"{t}/{c}"] for c in FEATURES} for t in simple.TASKS}
        att_sets = screen_sets(att_feats)
        for t in simple.TASKS:
            mine = as_sets(att_sets[t])
            theirs = {name: hs for _, name, hs in stored[t]["sets"] if name.startswith(("screen_", "shared_screen_"))}
            assert mine == theirs, (fold, t, len(mine), len(theirs))
        print(f"fold {fold}: check (a) passed, all {2 * len(COMBOS)} stage-2 screen sets of each task rebuilt", flush=True)

        # LRP features: D unchanged, every attention feature replaced by its AttnLRP twin.
        feats = {t: {"D": att[f"{t}/D"], **{c: lrp[f"{t}/{c}"] for c in LRP_CODES}} for t in simple.TASKS}
        sets = screen_sets(feats)

        # (b) single features against the stored sets.
        for t in simple.TASKS:
            mine, theirs = as_sets(sets[t]), {name: hs for _, name, hs in stored[t]["sets"]}
            assert mine["screen_D"] == theirs["screen_D"], (fold, t, "D")
            assert mine["screen_T"] == theirs["lrpT40"], (fold, t, "T")
            assert mine["shared_screen_T"] == theirs["shared_lrpT40"], (fold, t, "shared T")
        print(f"fold {fold}: check (b) passed (screen_D = stage-2 screen_D; screen_T, shared_screen_T = lrpT40, "
              f"shared_lrpT40)", flush=True)

        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "heads.json").write_text(json.dumps({"simple": {t: {"sets": sets[t]} for t in simple.TASKS}}))
        np.savez_compressed(out / "features.npz", **{f"{t}/{c}": feats[t][c] for t in simple.TASKS for c in FEATURES})

        distinct = {"per_task": {t: len({frozenset(map(tuple, hs)) for _, nm, hs in sets[t] if nm.startswith("screen_")})
                                 for t in simple.TASKS},
                    "shared": len({frozenset(map(tuple, hs)) for _, nm, hs in sets[simple.TASKS[0]]
                                   if nm.startswith("shared_screen_")}),
                    "combos": len(COMBOS)}
        (out / "screen.json").write_text(json.dumps(distinct, indent=1))

        rows = []
        for scope, prefix, tasks in (("per task", "screen_", simple.TASKS), ("shared", "shared_screen_", simple.TASKS[:1])):
            for t in tasks:
                mine, other = as_sets(sets[t]), as_sets(att_sets[t])
                for combo in COMBOS:
                    a = {tuple(h) for h in mine[prefix + combo]}
                    b = {tuple(h) for h in other[prefix + combo]}
                    rows.append(dict(scope=scope, task=t if scope == "per task" else "all", combo=combo,
                                     layer_median=float(np.median([l for l, _ in a])), common_attention=len(a & b)))
        with (out / "sets.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        per_task_common = np.mean([r["common_attention"] for r in rows if r["scope"] == "per task"])
        shared_common = np.mean([r["common_attention"] for r in rows if r["scope"] == "shared"])
        print(f"fold {fold}: written to {out}; distinct sets per task {min(distinct['per_task'].values())}-"
              f"{max(distinct['per_task'].values())}, shared {distinct['shared']}; heads in common with the "
              f"attention screen's set of the same combination: {per_task_common:.1f}/40 per task, "
              f"{shared_common:.1f}/40 shared", flush=True)


if __name__ == "__main__":
    main()
