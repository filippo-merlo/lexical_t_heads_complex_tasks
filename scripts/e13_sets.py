"""E13 step 3 (no GPU): every head set of the capped grid, per fold. J-head, rule style; every score
on the fold's choosing half (fold 1: A, fold 2: B).

Grid: scope (per task: ranked on the task; shared: one set for all 14 tasks) x N in ``SIZES`` x
condition. Each set is the top N of a ranking; ties go to the lower layer, then head.

- **Filter scores**, per task over the 500 pooled rule prompts at the last token: ``D`` the head's
  share of prompts whose top 10 holds a word of the E13 list (``e13_lists.py``); ``B`` mean
  attention on BOS (lower is better); ``T`` mean attention on the task-word positions
  (``e13_attention.py``).
- **D, B, T, DB, DT, BT, DBT**: per task, each filter's within-task rank (1 = best, ties averaged),
  averaged over the combination's filters; shared, E10's rule: each filter's within-task ranks
  averaged over the 14 tasks, then ranked again and averaged over the combination's filters.
- **causal**: Davidson et al.'s score (E4's ``cie.npz``, averaged over the three kinds of
  uninformative instruction); per task the task's score, shared averaged over the tasks.
- **_nogen** (DBT and causal): the generic heads removed from the ranking before the top N.
- **generic heads**: the J-head generic score (``e12_generic_halves``: cosine of the 14 tasks'
  top-10 token counts, averaged over task pairs) on all five rule templates; among the fold's
  Davidson top 30, the cut that leaves the least within-group sum of squares in the two groups
  (no free parameter); if it would mark more than 15, none (E12's safeguard).
- **yang**: the share of the 500 prompts whose top 10 holds a token of Yang et al.'s term list
  (their expansion, ``lth.terms.expand_to_token_ids``); heads with a share of 0 are not used, so a
  set may hold fewer than N heads (its real size is recorded); ties: mean number of list tokens in
  the top 10; from N = 100 on, a per-task set with fewer than N heads is not built. Shared: the
  within-task ranks of the share averaged over tasks, among heads with a
  share above 0 on some task.
- **random**: as many heads in every layer as the DBT set of the same scope and N, drawn outside
  it (E4's rule, ``e11_sets.layer_matched``), seeds 0-4; shared: one draw for all tasks.

Writes ``results/e13_capped/fold{f}/heads.json`` (``scripts/patch.py`` set-list layout: per task its
own sets and all shared sets), ``fv_sets.json`` (shared causal N with and without generic heads,
for Davidson et al.'s summed intervention), ``generic.json`` and ``sets.csv``.

    .venv/bin/python scripts/e13_sets.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, READOUTS, RESULTS, simple  # noqa: E402

OUT = RESULTS / "e13_capped"
HALF = {1: "A", 2: "B"}
SIZES = (20, 40, 60, 80, 100, 120)  # 100 and 120 added 2026-11-12
COMBOS = ("D", "B", "T", "DB", "DT", "BT", "DBT")
N_RANDOM = 5
N_HEADS, N_ALL = 32, 1024
KINDS = ("equiprobable", "wikitext", "other_task")
ATT = {"T": 0, "I": 1, "R": 2, "B": 3}  # columns of the attention masses
SIGN = {"D": 1, "T": 1, "B": -1}
JH = READOUTS.index("jhead")


def ranks(better: np.ndarray) -> np.ndarray:
    """1 = largest ``better`` along the last axis, ties averaged (E10)."""
    from e10_sets import ranks as r

    return r(better)


def heads_of(order) -> list[list[int]]:
    return sorted([int(i) // N_HEADS, int(i) % N_HEADS] for i in order)


def top(order, n, drop=frozenset()) -> list[list[int]]:
    return heads_of([i for i in order if int(i) not in drop][:n])


def two_group_cut(values: list[float]) -> int:
    """Number of values (sorted high to low) in the high group of the split with the least
    within-group sum of squares; the first such cut on a tie."""
    v = np.asarray(values, dtype=np.float64)
    cost = [k * v[:k].var() + (len(v) - k) * v[k:].var() for k in range(1, len(v))]
    return int(np.argmin(cost)) + 1


def main() -> None:
    import e12_generic_halves as e12
    from e11_sets import layer_matched

    from lth.evaluation import tokenizer
    from lth.store import load
    from lth.terms import expand_to_token_ids, simple_term_lists

    tok = tokenizer()
    e12.WORDINGS = tuple(simple.WORDINGS["rule"])  # E13: the generic score on all five templates
    for fold, half in HALF.items():
        out = OUT / f"fold{fold}"
        dsc = np.load(out / "d_scores.npz")
        att_dir = DATA / "readings" / "heldout" / f"attention13_{half}"
        att = {t: np.load(att_dir / f"{t}.npy").astype(np.float64).mean(0) for t in simple.TASKS}  # 1024 x 4
        score = {t: {"D": dsc[t], "B": att[t][:, ATT["B"]], "T": att[t][:, ATT["T"]]} for t in simple.TASKS}

        # Davidson et al.'s score and the fold's top 30 (E4).
        cie = np.load(RESULTS / "heldout" / f"fold{fold}" / "fv" / "cie.npz")
        causal = {t: np.mean([cie[f"{t}/{k}"] for k in KINDS], axis=0).ravel() for t in simple.TASKS}
        causal_shared = np.mean([causal[t] for t in simple.TASKS], axis=0)
        causal_order = {t: np.argsort(-causal[t], kind="stable") for t in simple.TASKS}
        shared_causal_order = np.argsort(-causal_shared, kind="stable")
        fv30 = [l * N_HEADS + h for l, h, _ in json.loads((RESULTS / "heldout" / f"fold{fold}" / "fv" / "heads.json")
                                                         .read_text())["heads"]]
        assert set(fv30) == set(int(i) for i in shared_causal_order[:30]), "Davidson top 30 not reproduced"
        e4 = json.loads((RESULTS / "heldout" / f"fold{fold}" / "heads.json").read_text())["simple"]
        for t in simple.TASKS:
            sets4 = {m: h for _, m, h in e4[t]["sets"]}
            for n in (20, 40):
                assert top(causal_order[t], n) == sets4[f"fv{n}"], (t, n)
                assert top(shared_causal_order, n) == sets4[f"fvuniv{n}"], n

        # Generic heads: two-group split of the fold's Davidson top 30 by the 5-template score.
        g = e12.generic_scores(DATA / "readings" / "heldout" / half / "simple", fv30)
        by_score = sorted(fv30, key=lambda x: -g[x])
        k = two_group_cut([g[x] for x in by_score])
        generic = set(by_score[:k]) if k <= 15 else set()
        e12_list = set(json.loads((RESULTS / "e12_generic_halves" / f"fold{fold}" / "generic.json").read_text())["generic"])
        names = {e12.name(x) for x in generic}
        print(f"fold {fold}: {len(generic)} generic heads {sorted(names)}; same as E12: {names == e12_list}")
        (out / "generic.json").write_text(json.dumps(dict(
            fold=fold, choosing_half=half, templates=list(e12.WORDINGS),
            rule="two-group split (least within-group sum of squares) of the fold's Davidson top 30 by J-head "
                 "generic score; none if more than 15",
            fv30_by_score=[[e12.name(x), round(g[x], 4)] for x in by_score], generic=sorted(names),
            same_as_e12=names == e12_list), indent=1))

        # Yang et al.'s term lists: share of prompts and mean number of list tokens in the top 10.
        yang_rate, yang_count = {}, {}
        for t in simple.TASKS:
            root = DATA / "readings" / "heldout" / half / "simple" / t.replace("/", "__")
            topk = np.concatenate([load(root, w)["top"][:, JH] for w in simple.WORDINGS["rule"]])
            hit = np.isin(topk, np.array(sorted(set(expand_to_token_ids(tok, simple_term_lists()[t])))))
            yang_rate[t], yang_count[t] = hit.any(-1).mean(0), hit.sum(-1).mean(0)

        # Per-task rankings.
        per_task, rows = {t: [] for t in simple.TASKS}, []
        for t in simple.TASKS:
            r = {f: ranks(SIGN[f] * score[t][f]) for f in "DBT"}
            for n in SIZES:
                for c in COMBOS:
                    order = np.argsort(np.mean([r[f] for f in c], axis=0), kind="stable")
                    per_task[t].append(["rule", f"{c}{n}", top(order, n)])
                    if c == "DBT":
                        dbt = per_task[t][-1][2]
                        per_task[t].append(["rule", f"DBT{n}_nogen", top(order, n, generic)])
                per_task[t].append(["rule", f"causal{n}", top(causal_order[t], n)])
                per_task[t].append(["rule", f"causal{n}_nogen", top(causal_order[t], n, generic)])
                eligible = np.flatnonzero(yang_rate[t] > 0)
                y = eligible[np.lexsort((eligible, -yang_count[t][eligible], -yang_rate[t][eligible]))]
                # From N = 100 on (added later), a Yang set with fewer than N heads is not built (user,
                # 2026-11-12: not enough heads, do not run it again); at 20-80 it is kept as before.
                if n <= 80 or len(y) >= n:
                    per_task[t].append(["rule", f"yang{n}", heads_of(y[:n])])
                per_task[t] += [["rule", f"random_dbt{n}_{i}", layer_matched(dbt, i, (t, f"dbt{n}"))]
                                for i in range(N_RANDOM)]

        # Shared rankings (E10's rank rule for the filters).
        within = {f: np.stack([ranks(SIGN[f] * score[t][f]) for t in simple.TASKS]).mean(0) for f in "DBT"}
        combined = {f: ranks(-within[f]) for f in "DBT"}  # rank of the average within-task rank
        y_rank = np.stack([ranks(yang_rate[t]) for t in simple.TASKS]).mean(0)
        y_eligible = np.flatnonzero(np.stack([yang_rate[t] for t in simple.TASKS]).max(0) > 0)
        y_order = y_eligible[np.argsort(y_rank[y_eligible], kind="stable")]
        shared, fv_sets = [], {}
        for n in SIZES:
            for c in COMBOS:
                order = np.argsort(np.mean([combined[f] for f in c], axis=0), kind="stable")
                shared.append(["rule", f"shared_{c}{n}", top(order, n)])
                if c == "DBT":
                    dbt = shared[-1][2]
                    shared.append(["rule", f"shared_DBT{n}_nogen", top(order, n, generic)])
            shared.append(["rule", f"shared_causal{n}", top(shared_causal_order, n)])
            shared.append(["rule", f"shared_causal{n}_nogen", top(shared_causal_order, n, generic)])
            shared.append(["rule", f"shared_yang{n}", heads_of(y_order[:n])])
            shared += [["rule", f"random_shared_dbt{n}_{i}", layer_matched(dbt, i)] for i in range(N_RANDOM)]
            fv_sets[f"fvsum{n}"] = top(shared_causal_order, n)
            fv_sets[f"fvsum{n}_nogen"] = top(shared_causal_order, n, generic)
        # E10's shared DBT-30 ordering must come out of the same rule on E8's scores (code check).
        e10 = np.load(RESULTS / "e10_shared" / f"fold{fold}" / "combined.npz")
        e10_order = np.argsort(np.mean([ranks(e10[f"rank/{f}"]) for f in "DBT"], axis=0), kind="stable")
        e8 = np.load(RESULTS / "e8_filters" / f"fold{fold}" / "scores.npz")
        w8 = {f: np.stack([ranks(SIGN[f] * (e8[f"{t}/dec"] if f == "D" else e8[f"{t}/att"][:, ATT[f]]))
                           for t in simple.TASKS]).mean(0) for f in "DBT"}
        mine = np.argsort(np.mean([ranks(-w8[f]) for f in "DBT"], axis=0), kind="stable")
        assert top(mine, 30) == top(e10_order, 30), "shared rank rule does not reproduce E10's rank_DBT"

        heads = {"simple": {t: {"sets": per_task[t] + shared} for t in simple.TASKS}}
        (out / "heads.json").write_text(json.dumps(heads, indent=1))
        (out / "fv_sets.json").write_text(json.dumps(dict(fold=fold, sets=fv_sets), indent=1))

        # Report rows: size, layers, generic heads, overlap with causal of the same scope and N.
        gen = {(x // N_HEADS, x % N_HEADS) for x in generic}
        for t in simple.TASKS:
            sets = {name: {tuple(h) for h in hs} for _, name, hs in per_task[t] + shared}
            for name, hs in sets.items():
                if name.startswith("random"):
                    continue
                n = int(re.search(r"(\d+)(?:_nogen)?$", name).group(1))
                ref = sets[f"shared_causal{n}" if name.startswith("shared") else f"causal{n}"]
                rows.append(dict(fold=fold, task=t, set=name, n=n, n_heads=len(hs),
                                 layer_median=float(np.median([l for l, _ in hs])) if hs else "",
                                 generic=len(hs & gen), overlap_causal=len(hs & ref)))
        with (out / "sets.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        n_sets = len(per_task["antonym"]) + len(shared)
        print(f"fold {fold}: {n_sets} sets per task ({len(shared)} shared), {len(fv_sets)} summed-vector sets")


if __name__ == "__main__":
    main()
