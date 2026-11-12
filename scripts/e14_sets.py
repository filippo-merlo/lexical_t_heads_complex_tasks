"""E14 step 5 (no GPU): the head sets of the all-wordings re-run, per fold (fold 1 chooses on A, fold 2 on B).

Two kinds of set, both later patched with the means of all 35 rule templates (``scripts/e14_patch.sh``):

- ``y5_<name>``: E13's sets, chosen on Yang et al.'s five templates, copied unchanged from
  ``results/e13_capped/fold{f}/heads.json``.
- ``a35_<name>``: the same rules (``e13_sets.py``) with every score recomputed on all 35 templates of the
  choosing half: D from ``e14_lists.py``, B and T from ``e13_attention.py --e14``, Davidson et al.'s
  causal score from ``davidson_fv.py --instructions all`` (the 14 question-answer templates), and the
  generic heads from the J-head generic score (``e12_generic_halves``) on the 35 templates, cut among
  the fold's Davidson top 30 by the two-group split (none if more than 15).

Names (per task; shared ones start ``shared_``): DBT, causal and causal_nogen at N = 40, 80, 100 and 120
(100 and 120 added 2026-11-12), and random draws with as many heads per layer as the a35 DBT-80 and DBT-120
sets, outside them (``e11_sets.layer_matched``, seeds 0-4): ``random_a35_dbt{80,120}_<i>`` and
``random_shared_a35_dbt{80,120}_<i>``. Ties: lower layer, then head.

Writes ``results/e14_wordings/fold{f}/heads.json`` (``scripts/patch.py`` set-list layout),
``generic.json`` and ``sets.csv`` (sizes, layers, generic heads, overlap of each a35 set with its y5 twin).

    .venv/bin/python scripts/e14_sets.py
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e13_sets import ATT, JH, KINDS, N_HEADS, N_RANDOM, SIGN, heads_of, ranks, top, two_group_cut  # noqa: E402
from lth import DATA, RESULTS, simple  # noqa: E402

OUT = RESULTS / "e14_wordings"
HALF = {1: "A", 2: "B"}
SIZES = (40, 80, 100, 120)  # 100 and 120 added 2026-11-12
RANDOM_NS = (80, 120)  # 120 added 2026-11-12
TASKS = list(simple.TASKS)
# E13 set names kept (per task; the shared ones with the prefix ``shared_``).
CORE = [m for n in SIZES for m in (f"DBT{n}", f"causal{n}", f"causal{n}_nogen")]


def name(x: int) -> str:
    return f"L{x // N_HEADS}H{x % N_HEADS}"


def generic_scores(half: str, heads) -> dict[int, float]:
    """E12's J-head generic score of each head in ``heads`` on the 35 templates' half readings: per task the
    head's token counts over its top 10 (each token once per prompt), cosine of every task pair, averaged."""
    import e12_generic_halves as e12

    from lth.store import load

    root = DATA / "readings" / "heldout" / half / "simple"
    top_ = {}
    for t in TASKS:
        parts = [load(root / t.replace("/", "__"), w)["top"] for w in simple.ALL_RULE]
        top_[t] = np.concatenate([p[:, JH] for p in parts if len(p)])
    out = {}
    for s in heads:
        c = {t: e12.counts(top_[t][:, s]) for t in TASKS}
        out[s] = float(np.mean([e12.cos(c[a], c[b]) for a, b in combinations(TASKS, 2)]))
    return out


def main() -> None:
    from e11_sets import layer_matched

    for fold, half in HALF.items():
        out = OUT / f"fold{fold}"
        dsc = np.load(out / "d_scores.npz")
        att_dir = DATA / "readings" / "heldout" / f"attention14_{half}"
        att = {t: np.load(att_dir / f"{t}.npy").astype(np.float64).mean(0) for t in TASKS}  # 1024 x 4
        score = {t: {"D": dsc[t], "B": att[t][:, ATT["B"]], "T": att[t][:, ATT["T"]]} for t in TASKS}

        cie = np.load(out / "fv" / "cie.npz")
        causal = {t: np.mean([cie[f"{t}/{k}"] for k in KINDS], axis=0).ravel() for t in TASKS}
        causal_order = {t: np.argsort(-causal[t], kind="stable") for t in TASKS}
        shared_causal_order = np.argsort(-np.mean([causal[t] for t in TASKS], axis=0), kind="stable")

        # Generic heads: two-group split of the fold's (new) Davidson top 30 by the 35-template score.
        fv30 = [int(i) for i in shared_causal_order[:30]]
        g = generic_scores(half, fv30)
        by_score = sorted(fv30, key=lambda x: -g[x])
        k = two_group_cut([g[x] for x in by_score])
        generic = set(by_score[:k]) if k <= 15 else set()
        e13_gen = set(json.loads((RESULTS / "e13_capped" / f"fold{fold}" / "generic.json").read_text())["generic"])
        names = {name(x) for x in generic}
        e13_fv30 = set(int(l) * N_HEADS + int(h) for l, h, _ in json.loads(
            (RESULTS / "heldout" / f"fold{fold}" / "fv" / "heads.json").read_text())["heads"])
        print(f"fold {fold}: Davidson top 30 shares {len(e13_fv30 & set(fv30))}/30 with E13's; "
              f"{len(generic)} generic heads {sorted(names)}; E13's {sorted(e13_gen)}")
        (out / "generic.json").write_text(json.dumps(dict(
            fold=fold, choosing_half=half, templates=list(simple.ALL_RULE),
            rule="two-group split (least within-group sum of squares) of the fold's Davidson top 30 by J-head "
                 "generic score on the 35 templates; none if more than 15",
            fv30_by_score=[[name(x), round(g[x], 4)] for x in by_score], generic=sorted(names),
            e13_generic=sorted(e13_gen), fv30_shared_with_e13=len(e13_fv30 & set(fv30))), indent=1))

        # a35 sets, per task.
        per_task = {t: [] for t in TASKS}
        for t in TASKS:
            r = {f: ranks(SIGN[f] * score[t][f]) for f in "DBT"}
            dbt_order = np.argsort(np.mean([r[f] for f in "DBT"], axis=0), kind="stable")
            for n in SIZES:
                per_task[t].append(["rule", f"a35_DBT{n}", top(dbt_order, n)])
                per_task[t].append(["rule", f"a35_causal{n}", top(causal_order[t], n)])
                per_task[t].append(["rule", f"a35_causal{n}_nogen", top(causal_order[t], n, generic)])
            for rn in RANDOM_NS:
                base = top(dbt_order, rn)
                per_task[t] += [["rule", f"random_a35_dbt{rn}_{i}", layer_matched(base, i, (t, f"dbt{rn}"))]
                                for i in range(N_RANDOM)]
        # a35 sets, shared (E10's rank rule for the filters, Davidson's mean score over tasks).
        within = {f: np.stack([ranks(SIGN[f] * score[t][f]) for t in TASKS]).mean(0) for f in "DBT"}
        combined = {f: ranks(-within[f]) for f in "DBT"}
        shared_dbt_order = np.argsort(np.mean([combined[f] for f in "DBT"], axis=0), kind="stable")
        shared = []
        for n in SIZES:
            shared.append(["rule", f"a35_shared_DBT{n}", top(shared_dbt_order, n)])
            shared.append(["rule", f"a35_shared_causal{n}", top(shared_causal_order, n)])
            shared.append(["rule", f"a35_shared_causal{n}_nogen", top(shared_causal_order, n, generic)])
        shared += [["rule", f"random_shared_a35_dbt{rn}_{i}", layer_matched(top(shared_dbt_order, rn), i)]
                   for rn in RANDOM_NS for i in range(N_RANDOM)]

        # y5 sets: E13's, unchanged.
        e13 = json.loads((RESULTS / "e13_capped" / f"fold{fold}" / "heads.json").read_text())["simple"]
        wanted = set(CORE) | {f"shared_{c}" for c in CORE}
        y5 = {t: [["rule", f"y5_{m}", hs] for _, m, hs in e13[t]["sets"] if m in wanted] for t in TASKS}
        assert all(len(v) == 2 * len(CORE) for v in y5.values()), "an E13 set is missing"

        heads = {"simple": {t: {"sets": y5[t] + per_task[t] + shared} for t in TASKS}}
        (out / "heads.json").write_text(json.dumps(heads, indent=1))

        rows = []
        gen = {(x // N_HEADS, x % N_HEADS) for x in generic}
        for t in TASKS:
            sets = {m: {tuple(h) for h in hs} for _, m, hs in heads["simple"][t]["sets"]}
            for m, hs in sets.items():
                if not m.startswith("a35_"):
                    continue
                twin = sets["y5_" + m.removeprefix("a35_")]
                rows.append(dict(fold=fold, task=t, set=m, n_heads=len(hs),
                                 layer_median=float(np.median([l for l, _ in hs])),
                                 generic=len(hs & gen), overlap_y5=len(hs & twin)))
        with (out / "sets.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"fold {fold}: {len(heads['simple']['antonym']['sets'])} sets per task")


if __name__ == "__main__":
    main()
