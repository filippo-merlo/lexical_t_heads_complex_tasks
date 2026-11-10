"""E10: one shared head set for all 14 simple tasks, from E8's five per-task head scores. J-head only,
one E4 fold, all scores on the choosing half.

E8 (``scripts/e8_sets.py``) scores every head on every task, on the choosing half's rule prompts:
``D`` decoding (share of prompts whose J-head top 10 holds a word of the task's 30-word list), and
four last-token attention masses, ``B`` BOS, ``T`` task words, ``I`` input, ``R`` rest. E8 kept, per
task, the heads passing a threshold on each filter of a combination. Here instead (user decisions,
2026-11-10):

1. each filter's score is combined over the 14 tasks into one score per head, in three ways:
   ``mean`` (average over tasks), ``worst`` (the weakest task), ``rank`` (the head's rank within each
   task, averaged over tasks); higher is better for D, T and I, lower for B and R, as in E8 (the
   weakest task of B and R is the largest value; ranks are 1 = best, ties averaged);
2. for each of E8's 31 filter combinations, the heads are ranked on each of its filters' combined
   scores (1 = best, ties averaged) and those ranks are averaged; no thresholds;
3. the 30 heads with the best average rank form the set (ties: lower layer, then head).

So 3 x 31 = 93 sets per fold, named ``<way>_<combination>`` (e.g. ``mean_DBT``), each the same for
every task; each task still gets its own mean when patched. Sets equal to an earlier one are listed
anyway (``scripts/patch.py`` patches them once and copies the rows).

Writes ``--out``: ``heads.json`` (set-list layout of ``scripts/rank_heads.py``, the 93 sets under
every task), ``combined.npz`` (per way and filter, the combined score of every head; ``rank`` holds
the average rank) and ``sets.csv`` (per set: heads, layers, overlap with Davidson et al.'s 30 FV
heads, with their 6 generic heads, and with E8's per-task set of the same combination).

    .venv/bin/python scripts/e10_sets.py --fold 1 --out results/e10_shared/fold1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e8_sets import COMBOS, FILTERS, N_HEADS  # noqa: E402
from lth import RESULTS, simple  # noqa: E402

N_SET = 30
WAYS = ("mean", "worst", "rank")
ATT = {"T": 0, "I": 1, "R": 2, "B": 3}  # columns of E8's ``att``: task words, query, rest, BOS
SIGN = {"D": 1, "T": 1, "I": 1, "B": -1, "R": -1}  # +1: higher is better
GENERIC6 = {(14, 29), (15, 5), (14, 11), (13, 31), (18, 22), (31, 14)}  # scripts/e3/fv_generic_heads.py


def ranks(better: np.ndarray) -> np.ndarray:
    """Rank along the last axis, 1 = largest ``better``, ties averaged."""
    import pandas as pd

    flat = better.reshape(-1, better.shape[-1])
    out = np.stack([pd.Series(-row).rank(method="average").to_numpy() for row in flat])
    return out.reshape(better.shape)


def combined(scores) -> dict[str, dict[str, np.ndarray]]:
    """way -> filter -> per-head 'higher is better' score combined over the tasks."""
    out = {w: {} for w in WAYS}
    for f in FILTERS:
        raw = np.stack([scores[f"{t}/dec"] if f == "D" else scores[f"{t}/att"][:, ATT[f]]
                        for t in simple.TASKS]).astype(np.float64)  # tasks x heads
        better = SIGN[f] * raw
        out["mean"][f] = better.mean(0)
        out["worst"][f] = better.min(0)
        out["rank"][f] = -ranks(better).mean(0)  # negated: higher is better
    return out


def choose(scores: dict[str, np.ndarray], combo: str) -> list[list[int]]:
    """The ``N_SET`` heads with the best average rank over the filters of ``combo``."""
    mean_rank = np.mean([ranks(scores[f]) for f in combo], axis=0)
    order = np.argsort(mean_rank, kind="stable")[:N_SET]  # ties: lower index = lower layer, head
    return sorted([int(i) // N_HEADS, int(i) % N_HEADS] for i in order)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    e8 = RESULTS / "e8_filters" / f"fold{args.fold}"
    scores = np.load(e8 / "scores.npz")
    per_task = json.loads((e8 / "heads.json").read_text())["simple"]
    davidson = {(l, h) for l, h, _ in json.loads((RESULTS / "davidson_fv_all14_30" / "heads.json").read_text())["heads"]}

    comb = combined(scores)
    sets, rows = [], []
    for way in WAYS:
        for combo in COMBOS:
            heads = choose(comb[way], combo)
            name = f"{way}_{combo}"
            sets.append(["rule", name, heads])
            hs = {tuple(h) for h in heads}
            e8_sets = [{tuple(h) for h in v} for t in simple.TASKS
                       for st, n, v in per_task[t]["sets"] if n == combo]
            layers = [l for l, _ in heads]
            rows.append(dict(set=name, way=way, combination=combo, n_heads=len(heads),
                             layer_min=min(layers), layer_median=float(np.median(layers)), layer_max=max(layers),
                             davidson30=len(hs & davidson), generic6=len(hs & GENERIC6),
                             e8_tasks=len(e8_sets),
                             e8_share=round(float(np.mean([len(hs & s) / len(hs) for s in e8_sets])), 3)
                             if e8_sets else "",
                             heads=" ".join(f"L{l}H{h}" for l, h in heads)))
    first = {}
    for r, (_, name, heads) in zip(rows, sets):
        r["same_as"] = first.setdefault(frozenset(map(tuple, heads)), name)
        r["same_as"] = "" if r["same_as"] == name else r["same_as"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "heads.json").write_text(json.dumps({"simple": {t: {"sets": sets} for t in simple.TASKS}}, indent=1))
    np.savez_compressed(args.out / "combined.npz", **{f"{w}/{f}": comb[w][f] for w in WAYS for f in FILTERS})
    with (args.out / "sets.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"fold {args.fold}: {len(sets)} sets, {len({frozenset(map(tuple, s[2])) for s in sets})} distinct")


if __name__ == "__main__":
    main()
