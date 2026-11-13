"""E23 step 1 (no GPU): the 15 best E22 combinations of each scope at every E13 size.

Selection (user, 2026-11-13): per scope (per task, shared), the 15 combinations of D, B, T, L, M, S, N
with the highest E22 check-1 accuracy at 40 heads (``results/e22_all_combos/all_combos.csv``:
cross-fitted alpha, mean of the two folds, mean of 14 tasks; ties to the earlier combination in
E22's order). The choice uses the same held-out items the size curves are then scored on, so the
curves of the chosen combinations are optimistic.

Sets as ``scripts/e22_sets.py`` (scores and combination rule of E13/E21/E22), at N in 20, 60, 80,
100, 120 (40 is in E13/E21/E22). Combinations E13 ran at every size (D, B, T, DB, DT, BT, DBT) are
not rerun. Checks: the rebuilt 40-head sets equal the patched ones (E13, E21 or E22), and the
rebuilt sets of E13's combinations equal E13's at every size.

Writes ``results/e23_best_sizes/fold{f}/heads.json`` (``scripts/patch.py`` set-list layout),
``chosen.csv`` and ``fold{f}/sets.csv``.

    .venv/bin/python scripts/e23_sets.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS, simple  # noqa: E402

from e13_sets import SIZES, ranks, top  # noqa: E402
from e20_bai_attention import CATS  # noqa: E402
from e21_sets import PARTS, SIGN  # noqa: E402
from e22_sets import COMBOS, DONE, LETTERS  # noqa: E402

OUT = RESULTS / "e23_best_sizes"
HALF = {1: "A", 2: "B"}
K = 15
E13_ALL = DONE["e13_capped"]  # run by E13 at every size
NEW_SIZES = tuple(n for n in SIZES if n != 40)
SCOPES = {"task": "", "shared": "shared_"}


def chosen() -> dict[str, list[str]]:
    s = pd.read_csv(RESULTS / "e22_all_combos" / "all_combos.csv")
    order = {c: i for i, c in enumerate(COMBOS)}
    out = {}
    for scope in SCOPES:
        x = s[s.scope == scope].assign(o=lambda d: d.condition.map(order))
        assert len(x) == len(COMBOS) and x.o.notna().all()
        out[scope] = x.sort_values(["check1_task", "o"], ascending=[False, True]).condition.head(K).tolist()
    return out


def main() -> None:
    pick = chosen()
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(RESULTS / "e22_all_combos" / "all_combos.csv").set_index(["scope", "condition"])
    with (OUT / "chosen.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "rank", "combo", "check1_task_40", "rerun"])
        for scope, cs in pick.items():
            for i, c in enumerate(cs, 1):
                w.writerow([scope, i, c, round(float(s.loc[(scope, c), "check1_task"]), 4), c not in E13_ALL])
    for scope, cs in pick.items():
        print(f"{scope}: {' '.join(cs)}  (rerun: {sum(c not in E13_ALL for c in cs)})")

    patched40 = {}
    for fold in HALF:
        patched40[fold] = {}
        for exp in ("e13_capped", "e21_bai_rules", "e22_all_combos"):
            for t, v in json.loads((RESULTS / exp / f"fold{fold}" / "heads.json").read_text())["simple"].items():
                for _, name, hs in v["sets"]:
                    patched40[fold].setdefault(t, {})[name] = hs

    for fold, half in HALF.items():
        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        dsc = np.load(RESULTS / "e13_capped" / f"fold{fold}" / "d_scores.npz")
        att = {t: np.load(DATA / "readings" / "heldout" / f"attention20_{half}" / f"{t}.npy").astype(np.float64).mean(0)
               for t in simple.TASKS}
        score = {t: {"D": dsc[t], **{f: att[t][:, [CATS.index(c) for c in cs]].sum(1) for f, cs in PARTS.items()}}
                 for t in simple.TASKS}
        within = {f: np.stack([ranks(SIGN[f] * score[t][f]) for t in simple.TASKS]).mean(0) for f in LETTERS}
        combined = {f: ranks(-within[f]) for f in LETTERS}
        shared_order = {c: np.argsort(np.mean([combined[f] for f in c], axis=0), kind="stable") for c in COMBOS}

        sets = {t: [] for t in simple.TASKS}
        rows = []
        for t in simple.TASKS:
            r = {f: ranks(SIGN[f] * score[t][f]) for f in LETTERS}
            for scope, prefix in SCOPES.items():
                for c in pick[scope]:
                    order = (np.argsort(np.mean([r[f] for f in c], axis=0), kind="stable") if scope == "task"
                             else shared_order[c])
                    assert top(order, 40) == patched40[fold][t][f"{prefix}{c}40"], (fold, t, scope, c)
                    for n in NEW_SIZES:
                        hs = top(order, n)
                        if c in E13_ALL:
                            assert hs == patched40[fold][t][f"{prefix}{c}{n}"], (fold, t, scope, c, n)
                            continue
                        sets[t].append(["rule", f"{prefix}{c}{n}", hs])
                        rows.append(dict(fold=fold, task=t, scope=scope, set=f"{prefix}{c}{n}", combo=c, n=n,
                                         n_heads=len(hs), layer_median=float(np.median([l for l, _ in hs]))))
        (out / "heads.json").write_text(json.dumps({"simple": {t: {"sets": v} for t, v in sets.items()}}, indent=1))
        with (out / "sets.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"fold {fold}: 40-head sets reproduced; {len(sets[simple.TASKS[0]])} sets per task written")


if __name__ == "__main__":
    main()
