"""E16 step 2 (no GPU): B replaced by the Avg Head Output Norm score of Sandoval-Segura et al. (2025,
arXiv:2504.03889), layer-normalised (AHON (LN), their Sec. 3, Eq. 1), in E13's capped grid.

Score ``O`` per task over the 500 pooled rule prompts of the fold's choosing half: per prompt, a head's
AvgNorm (``e16_norms.py``) divided by the mean AvgNorm of the 32 heads of its layer, then averaged over
the prompts. Higher is better (a head that writes more than its layer's average). Only the score is
taken from the paper; their thresholds and zero-ablation evaluation are not used.

Sets: ``O, DO, OT, DOT``, the counterparts of E13's ``B, DB, BT, DBT``, with E13's rule (per task:
within-task ranks averaged over the combination's filters; shared: E10's rule), N in 20/40/60/80, ties
to the lower layer, then head. ``D`` and ``T`` are E13's scores. Code check: the same builder on
E13's B score reproduces E13's B, DB, BT, DBT sets.

Writes ``results/e16_ahon/fold{f}/heads.json`` (``scripts/patch.py`` set-list layout), ``scores.npz``
(O per task) and ``sets.csv`` (size, layer median, overlap with E13's B counterpart, Spearman of O
with -B over the 1,024 heads).

    .venv/bin/python scripts/e16_sets.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS, simple  # noqa: E402

OUT = RESULTS / "e16_ahon"
E13 = RESULTS / "e13_capped"
HALF = {1: "A", 2: "B"}
SIZES = (20, 40, 60, 80)
N_HEADS, N_LAYERS = 32, 32
PAIRS = {"O": "B", "DO": "DB", "OT": "BT", "DOT": "DBT"}  # E16 combination -> E13 counterpart
ATT = {"T": 0, "B": 3}
SIGN = {"D": 1, "T": 1, "B": -1, "O": 1}


def ahon_ln(norms: np.ndarray) -> np.ndarray:
    """[prompts, 1024] AvgNorm -> [1024] AHON (LN): per prompt over the layer's mean, averaged."""
    a = norms.reshape(len(norms), N_LAYERS, N_HEADS)
    return (a / a.mean(-1, keepdims=True)).mean(0).ravel()


def main() -> None:
    from e10_sets import ranks
    from e13_sets import top

    def spearmanr(a, b) -> float:
        return float(np.corrcoef(ranks(np.asarray(a)), ranks(np.asarray(b)))[0, 1])

    for fold, half in HALF.items():
        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        dsc = np.load(E13 / f"fold{fold}" / "d_scores.npz")
        att_dir = DATA / "readings" / "heldout" / f"attention13_{half}"
        nrm_dir = DATA / "readings" / "heldout" / f"headnorm16_{half}"
        score = {}
        for t in simple.TASKS:
            att = np.load(att_dir / f"{t}.npy").astype(np.float64).mean(0)
            nrm = np.load(nrm_dir / f"{t}.npy").astype(np.float64)
            assert len(nrm) == len(np.load(att_dir / f"{t}.npy", mmap_mode="r")), t  # same prompts as B and T
            score[t] = {"D": dsc[t], "B": att[:, ATT["B"]], "T": att[:, ATT["T"]], "O": ahon_ln(nrm[:, :, 0])}
        np.savez(out / "scores.npz", **{t: score[t]["O"] for t in simple.TASKS})

        def per_task_sets(t, combos):
            r = {f: ranks(SIGN[f] * score[t][f]) for f in "DBTO"}
            return {f"{c}{n}": top(np.argsort(np.mean([r[f] for f in c], axis=0), kind="stable"), n)
                    for n in SIZES for c in combos}

        within = {f: np.stack([ranks(SIGN[f] * score[t][f]) for t in simple.TASKS]).mean(0) for f in "DBTO"}
        combined = {f: ranks(-within[f]) for f in "DBTO"}

        def shared_sets(combos):
            return {f"shared_{c}{n}": top(np.argsort(np.mean([combined[f] for f in c], axis=0), kind="stable"), n)
                    for n in SIZES for c in combos}

        # Code check: E13's B-family sets come out of this builder unchanged.
        e13 = json.loads((E13 / f"fold{fold}" / "heads.json").read_text())["simple"]
        for t in simple.TASKS:
            stored = {name: hs for _, name, hs in e13[t]["sets"]}
            mine = per_task_sets(t, PAIRS.values()) | shared_sets(PAIRS.values())
            assert all(stored[k] == v for k, v in mine.items()), (fold, t)

        shared = shared_sets(PAIRS)
        heads, rows = {"simple": {}}, []
        for t in simple.TASKS:
            sets = per_task_sets(t, PAIRS) | shared
            heads["simple"][t] = {"sets": [["rule", k, v] for k, v in sets.items()]}
            stored = {name: hs for _, name, hs in e13[t]["sets"]}
            rho = spearmanr(score[t]["O"], -score[t]["B"])
            for k, hs in sets.items():
                core = k.removeprefix("shared_").rstrip("0123456789")
                ref = k.replace(core, PAIRS[core], 1)
                rows.append(dict(fold=fold, task=t, set=k, n=len(hs), b_counterpart=ref,
                                 overlap_b=len({tuple(h) for h in hs} & {tuple(h) for h in stored[ref]}),
                                 layer_median=float(np.median([l for l, _ in hs])),
                                 spearman_o_minus_b=round(float(rho), 4)))
        (out / "heads.json").write_text(json.dumps(heads, indent=1))
        with (out / "sets.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        rho_shared = spearmanr(within["O"], within["B"])  # both mean ranks, 1 = best
        print(f"fold {fold}: {len(heads['simple']['antonym']['sets'])} sets per task; E13 B sets reproduced; "
              f"Spearman(O, -B) per task {np.mean([r['spearman_o_minus_b'] for r in rows]):.2f}, "
              f"shared mean ranks {rho_shared:.2f}")


if __name__ == "__main__":
    main()
