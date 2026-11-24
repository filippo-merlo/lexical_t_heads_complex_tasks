"""Causal (zero-shot) sets (paper/head-sets-simple-design.tex, Sec. 4.2), added to each fold's
``results/rebuild/additions/fold<f>/heads.json``. No GPU.

The zero-shot causal score (``scripts/causal_context_cie.py``, ``results/heldout/fold<f>/fv/cie_context.npz``,
context ``zero``) takes the place of Causal's score averaged over the three kinds of uninformative text;
everything else is ``scripts/rebuild_sets.py``'s, through its own functions:

- ``causalzs<N>``: the top N of the task's ranking; ``shared_causalzs<N>``: the top N by within-task ranks
  averaged over tasks; N in 10..120;
- ``causalzsnogen<N>``, ``shared_causalzsnogen<N>``: the same without the generic heads, found by Sec. 5.5's
  rule among the 30 top heads of the shared zero-shot ranking (J-head readings of the fold's pool);
- ``shared_davidson_zs``: Davidson et al.'s own form, the 20 top heads by the score averaged over tasks
  (patched summed, ``scripts/e26_summed.py``);
- three random sets per set, ``random_<name>_<i>``, with as many heads per layer (same seeding rule).

Self-check before anything is written: Causal's sets (``causal*``, ``causalnogen*``, ``shared_causal*``,
``shared_causalnogen*``, ``shared_davidson_own`` and their random sets) rebuilt by the same code from
``cie.npz`` must equal those in heads.json. Existing ``*causalzs*`` and ``*davidson_zs*`` sets are replaced;
no other set is touched.

Also writes ``results/rebuild/additions/fold<f>/causal_context.json``: the generic heads of the zero-shot
ranking and, for the overlap analysis, the per-task and shared top N of every context (Causal, zero,
frame) as head indices.

    .venv/bin/python scripts/causal_context_sets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import RESULTS, simple  # noqa: E402
from rebuild_sets import (DAVIDSON_N, FOLDS, GENERIC_MAX, GENERIC_TOP, KINDS, OUT, SIZES, generic_scores,  # noqa: E402
                          name, order_of, pool, random_sets, rankdata, ranks, top, two_group_cut)

NEW = ("causalzs", "davidson_zs")


def rule_sets(fold: int, score: dict, tops: dict, prefix: str, own: str):
    """Per-task and shared sets of a causal ranking, as ``rebuild_sets.build`` makes Causal's."""
    single = {t: ranks(score[t]) for t in simple.TASKS}
    shared_order = order_of(rankdata(np.mean([single[t] for t in simple.TASKS], axis=0), method="average"))
    top30 = [int(i) for i in shared_order[:GENERIC_TOP]]
    g = generic_scores(tops, top30)
    by_score = sorted(top30, key=lambda x: -g[x])
    k = two_group_cut([g[x] for x in by_score])
    generic = set(by_score[:k]) if k <= GENERIC_MAX else set()

    per_task, shared = {t: [] for t in simple.TASKS}, []

    def add(target: list, rule: str, n, heads, key: str):
        if heads is None:
            return
        label = f"{rule}{n}" if n != "" else rule
        target.append(["rule", label, heads])
        target += [["rule", f"random_{label}_{i}", r] for i, r in enumerate(random_sets(heads, f"{fold}/{key}/{label}"))]

    for t in simple.TASKS:
        order = order_of(single[t])
        for n in SIZES:
            add(per_task[t], prefix, n, top(order, n), t)
            add(per_task[t], f"{prefix}nogen", n, top(order, n, generic), t)
    davidson = order_of(ranks(np.mean([score[t] for t in simple.TASKS], axis=0)))
    add(shared, f"shared_davidson_{own}", "", top(davidson, DAVIDSON_N), "shared")
    for n in SIZES:
        add(shared, f"shared_{prefix}", n, top(shared_order, n), "shared")
        add(shared, f"shared_{prefix}nogen", n, top(shared_order, n, generic), "shared")
    return per_task, shared, generic, single, shared_order


def main() -> None:
    for fold, half in FOLDS.items():
        out = OUT / f"fold{fold}"
        fv = RESULTS / "heldout" / f"fold{fold}" / "fv"
        cie = np.load(fv / "cie.npz")
        ctx = np.load(fv / "cie_context.npz")
        tops = {t: pool(half, t)[1] for t in simple.TASKS}
        causal = {t: np.mean([cie[f"{t}/{k}"] for k in KINDS], axis=0).ravel() for t in simple.TASKS}
        heads = json.loads((out / "heads.json").read_text())

        # Self-check: Causal's sets, rebuilt by this code, equal the stored ones.
        per_task, shared, generic_c, single_c, shared_c = rule_sets(fold, causal, tops, "causal", "own")
        stored = {(t, nm): hs for t in simple.TASKS for _, nm, hs in heads["simple"][t]["sets"]}
        n_checked = 0
        for t in simple.TASKS:
            for _, nm, hs in per_task[t] + shared:
                assert stored[(t, nm)] == hs, (fold, t, nm)
                n_checked += 1
        print(f"fold {fold}: {n_checked} stored Causal sets rebuilt identically; generic "
              f"{sorted(name(x) for x in generic_c)}", flush=True)

        zero = {t: ctx[f"{t}/zero"].ravel() for t in simple.TASKS}
        frame = {t: ctx[f"{t}/frame"].ravel() for t in simple.TASKS}
        per_task, shared, generic_z, single_z, shared_z = rule_sets(fold, zero, tops, "causalzs", "zs")
        for t in simple.TASKS:
            kept = [s for s in heads["simple"][t]["sets"] if not any(k in s[1] for k in NEW)]
            heads["simple"][t]["sets"] = kept + per_task[t] + shared
        partial = out / "heads.json.partial"  # replaced at once: patch runs may be reading it
        partial.write_text(json.dumps(heads))
        partial.replace(out / "heads.json")

        _, _, _, single_f, shared_f = rule_sets(fold, frame, tops, "causalframe", "frame")
        orders = {"causal": (single_c, shared_c), "zero": (single_z, shared_z), "frame": (single_f, shared_f)}
        info = dict(
            fold=fold, choosing_half=half, generic_zero=sorted(name(x) for x in generic_z),
            generic_causal=sorted(name(x) for x in generic_c),
            top={c: {"per_task": {t: [int(i) for i in order_of(s[t])[:max(SIZES)]] for t in simple.TASKS},
                     "shared": [int(i) for i in sh[:max(SIZES)]]} for c, (s, sh) in orders.items()},
            top20_mean={c: {t: float(np.sort(v[t])[-20:].mean()) for t in simple.TASKS}
                        for c, v in (("causal", causal), ("zero", zero), ("frame", frame))})
        (out / "causal_context.json").write_text(json.dumps(info, indent=1))
        print(f"fold {fold}: zero-shot sets added; generic {info['generic_zero']}", flush=True)


if __name__ == "__main__":
    main()
