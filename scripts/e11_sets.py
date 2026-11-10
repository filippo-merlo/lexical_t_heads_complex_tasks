"""E11: the sets that complete the comparison of paper/head-sets-simple.tex (its "Conditions still to
run"). J-head, rule style, one E4 fold; every set comes from an existing ranking or an existing
random-draw rule, nothing is chosen anew.

Shared sets (the same heads for every task; each task still patched with its own means):

- ``fvuniv{5,10,20,30,40}``: the first N heads by Davidson et al.'s score averaged over the three
  kinds of uninformative instruction and the 14 tasks, on the choosing half (E4's
  ``scripts/heldout_fv_sets.py``; ``fvuniv20``/``40`` equal E4's sets);
- ``fvuniv20_nogen``: ``fvuniv20`` without the FV generic heads it holds (``e10_sets.GENERIC6``);
- ``e10dbt{5,10,20,40}``: the first N heads of E10's ``rank_DBT`` ordering (``e10dbt30`` would be
  E10's ``rank_DBT`` itself, already run, so it is not listed);
- ``randomuniv20_{i}``, ``randomdbt30_{i}``, i = 0..4: as many heads in every layer as ``fvuniv20``
  / E10's ``rank_DBT``, drawn outside it (seed i), the same draw for every task (E4's rule).

Per task, from E8's sets of the fold (``results/e8_filters/fold{f}/heads.json``):

- ``random_layer_{DB,DBT,T}_{i}``: as many heads in every layer as the E8 set, drawn outside it
  (seed i; E4's and E7's rule);
- ``random_drop_{DB,DBT}_{i}``: as many heads as the E8 set, drawn from E8's ``D`` set of the task
  (seed i; E7's random-drop: random decoders instead of the filtered ones). T is not a subset of
  D, so it has no random-drop control.

Check 2 only: E4's ``random20_{i}``, ``random40_{i}``, ``randomNt_{i}``, copied unchanged from
``results/heldout/fold{f}/heads.json``, to be patched at alpha 1 and 2 (E4 ran them only at their
ten-alpha cross-fitted alpha).

Writes ``results/e11_complete/fold{f}/heads.json`` (set-list layout) and ``sets.csv``.

    .venv/bin/python scripts/e11_sets.py --fold 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e10_sets import GENERIC6, ranks  # noqa: E402
from lth import RESULTS, simple  # noqa: E402

N_LAYERS = N_HEADS = 32
KINDS = ("equiprobable", "wikitext", "other_task")
FV_SIZES = (5, 10, 20, 30, 40)
DBT_SIZES = (5, 10, 20, 40)
N_RANDOM = 5
OUT = RESULTS / "e11_complete"


def heads_of(order) -> list[list[int]]:
    return sorted([int(i) // N_HEADS, int(i) % N_HEADS] for i in order)


SHORT = Counter()  # (task, set) -> layers drawn from the whole layer (too few heads outside the set)


def layer_matched(base: list[list[int]], seed: int, key=None) -> list[list[int]]:
    """As many heads in every layer as ``base``, drawn outside it (E4's ``heldout_sets.py`` rule).
    A layer with fewer heads outside the set than it needs (large T sets) is drawn from all its
    heads instead, so the draw may share heads with the set there."""
    inside = {tuple(h) for h in base}
    per_layer = Counter(l for l, _ in inside)
    rng = np.random.default_rng(seed)
    pick = []
    for layer in sorted(per_layer):
        pool = [(layer, x) for x in range(N_HEADS) if (layer, x) not in inside]
        if len(pool) < per_layer[layer]:
            pool = [(layer, x) for x in range(N_HEADS)]
            SHORT[key] += 1
        pick += [list(pool[int(j)]) for j in rng.choice(len(pool), per_layer[layer], replace=False)]
    return sorted(pick)


def drawn_from(pool: list[list[int]], n: int, seed: int) -> list[list[int]]:
    """``n`` heads drawn from ``pool`` (E7's random-drop rule)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), n, replace=False)
    return sorted([list(pool[int(i)]) for i in idx])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    f = args.fold

    # Davidson et al.'s score averaged over kinds and tasks, on the choosing half (E4).
    cie = np.load(RESULTS / "heldout" / f"fold{f}" / "fv" / "cie.npz")
    univ = np.mean([np.mean([cie[f"{t}/{k}"] for k in KINDS], axis=0) for t in simple.TASKS], axis=0).ravel()
    fv_order = np.argsort(-univ, kind="stable")
    e4 = json.loads((RESULTS / "heldout" / f"fold{f}" / "heads.json").read_text())["simple"]
    shared = [["rule", f"fvuniv{n}", heads_of(fv_order[:n])] for n in FV_SIZES]
    for n in (20, 40):  # equal to E4's sets of the same name
        assert shared[FV_SIZES.index(n)][2] == next(h for _, m, h in e4["antonym"]["sets"] if m == f"fvuniv{n}")
    fvuniv20 = shared[FV_SIZES.index(20)][2]
    nogen = [h for h in fvuniv20 if tuple(h) not in GENERIC6]
    shared.append(["rule", "fvuniv20_nogen", nogen])

    # E10's rank_DBT ordering: average of the D, B, T ranks of the within-task-rank scores.
    comb = np.load(RESULTS / "e10_shared" / f"fold{f}" / "combined.npz")
    mean_rank = np.mean([ranks(comb[f"rank/{x}"]) for x in "DBT"], axis=0)
    dbt_order = np.argsort(mean_rank, kind="stable")
    e10 = json.loads((RESULTS / "e10_shared" / f"fold{f}" / "heads.json").read_text())["simple"]
    rank_dbt = next(h for _, m, h in e10["antonym"]["sets"] if m == "rank_DBT")
    assert heads_of(dbt_order[:30]) == rank_dbt, "E10 rank_DBT not reproduced"
    shared += [["rule", f"e10dbt{n}", heads_of(dbt_order[:n])] for n in DBT_SIZES]
    shared += [["rule", f"randomuniv20_{i}", layer_matched(fvuniv20, i)] for i in range(N_RANDOM)]
    shared += [["rule", f"randomdbt30_{i}", layer_matched(rank_dbt, i)] for i in range(N_RANDOM)]

    e8 = json.loads((RESULTS / "e8_filters" / f"fold{f}" / "heads.json").read_text())["simple"]
    out, rows = {}, []
    for t in simple.TASKS:
        own = {m: h for _, m, h in e8[t]["sets"]}
        sets = [list(s) for s in shared]
        for name in ("DB", "DBT", "T"):
            base = own.get(name) or []
            if not base:
                continue
            sets += [["rule", f"random_layer_{name}_{i}", layer_matched(base, i, (t, name))] for i in range(N_RANDOM)]
            if name != "T":
                assert {tuple(h) for h in base} <= {tuple(h) for h in own["D"]}
                sets += [["rule", f"random_drop_{name}_{i}", drawn_from(own["D"], len(base), i)]
                         for i in range(N_RANDOM)]
        sets += [list(s) for s in e4[t]["sets"] if s[1].startswith(("random20_", "random40_", "randomNt_"))]
        out[t] = {"sets": sets}
        for _, name, heads in sets:
            base_name = name.rsplit("_", 1)[0].replace("random_layer_", "") if name.startswith("random_layer_") else ""
            rows.append(dict(task=t, set=name, n_heads=len(heads),
                             whole_layer_draws=SHORT.get((t, base_name), 0) // N_RANDOM if base_name else 0,
                             generic6=sum(tuple(h) in GENERIC6 for h in heads),
                             heads=" ".join(f"L{l}H{h}" for l, h in heads)))
    d = OUT / f"fold{f}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "heads.json").write_text(json.dumps({"simple": out}, indent=1))
    with (d / "sets.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"fold {f}: fvuniv20 generic {len(fvuniv20) - len(nogen)} ({sorted(map(tuple, fvuniv20)) and ''}"
          f"{[f'L{l}H{h}' for l, h in fvuniv20 if tuple((l, h)) in GENERIC6]}), "
          f"sets per task {min(len(v['sets']) for v in out.values())}-{max(len(v['sets']) for v in out.values())}; "
          f"layers drawn whole (per draw): {dict((k, v // N_RANDOM) for k, v in SHORT.items())}")


if __name__ == "__main__":
    main()
