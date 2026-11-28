"""Sets for testing the core heads on the four unseen tasks. No GPU.

Core (7): the heads helpful on 10-13 of the 14 tasks in both folds and with a positive random-set slope on
13-14 (offline per-head credit analysis of 2026-11-28: ridge regression of every rebuilt set's accuracy on head
membership, checked on the layer-matched random sets; not in the repo): L15H7, L17H5, L14H12, L14H11, L12H9, L13H8, L15H5. Same list in both folds.
Comparisons at the same size, per fold, the top 7 of the shared rankings of rebuild_sets.py / lrp_sets.py:
  shared_DBTL7  (DBTL = shared Combined chosen by both folds at 10 heads), shared_lrpT7, shared_causal7,
  shared_causalnogen7. Random: 3 layer-matched draws of the core (rebuild_sets.random_sets).
Self-check: the same code reproduces the stored 10-head shared sets (selected10 = DBTL, lrpT10, causal10,
causalnogen10) of results/rebuild/additions/fold<f>/heads.json and shared_n10.
Writes results/rebuild/newtasks_core/fold<f>/heads.json (or <out>/fold<f>/) for the unseen tasks.

    .venv/bin/python scripts/core_unseen_sets.py [out]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from lth import simple  # noqa: E402
from rebuild_sets import FOLDS, heads_of, order_of, random_sets, rankdata, ranks, top  # noqa: E402

CORE = ["L15H7", "L17H5", "L14H12", "L14H11", "L12H9", "L13H8", "L15H5"]
N = len(CORE)


def idx(nm):
    l, h = nm[1:].split("H")
    return int(l) * 32 + int(h)


def shared_order(scores):
    return order_of(rankdata(np.mean([ranks(scores[t]) for t in simple.TASKS], axis=0), method="average"))


def main(out: Path):
    for fold in FOLDS:
        add = ROOT / "results" / "rebuild" / "additions" / f"fold{fold}"
        fz, lz = np.load(add / "features.npz"), np.load(add / "lrp_features.npz")
        gen = {idx(n) for n in json.loads((add / "generic.json").read_text())["generic"]}
        stored = {nm: hs for _, nm, hs in json.loads((add / "heads.json").read_text())["simple"]["antonym"]["sets"]}
        s10 = {nm: hs for _, nm, hs in json.loads((ROOT / "results" / "rebuild" / "shared_n10" / f"fold{fold}" / "heads.json").read_text())["simple"]["antonym"]["sets"]}
        # shared single-feature ranks, as rebuild_sets.build
        single = {c: rankdata(np.mean([ranks(fz[f"{t}/{c}"]) for t in simple.TASKS], axis=0), method="average")
                  for c in "DBTL"}
        dbtl = order_of(np.mean([single[c] for c in "DBTL"], axis=0))
        lrpt = shared_order({t: lz[f"{t}/T"] for t in simple.TASKS})
        causal = shared_order({t: fz[f"{t}/causal"] for t in simple.TASKS})
        checks = {"shared_selected10 (DBTL)": (top(dbtl, 10), s10.get("shared_selected10", stored.get("shared_selected10"))),
                  "shared_lrpT10": (top(lrpt, 10), stored["shared_lrpT10"]),
                  "shared_causal10": (top(causal, 10), stored["shared_causal10"]),
                  "shared_causalnogen10": (top(causal, 10, gen), stored["shared_causalnogen10"])}
        for k, (a, b) in checks.items():
            assert a == b, (fold, k, a, b)
        print(f"fold {fold}: rankings reproduce {', '.join(checks)}")
        core = heads_of([idx(n) for n in CORE])
        sets = [["rule", f"core{N}", core]]
        sets += [["rule", f"random_core{N}_{i}", r] for i, r in enumerate(random_sets(core, f"core_unseen/{fold}/core{N}"))]
        for name, order, drop in (("shared_DBTL", dbtl, frozenset()), ("shared_lrpT", lrpt, frozenset()),
                                  ("shared_causal", causal, frozenset()), ("shared_causalnogen", causal, frozenset(gen))):
            sets.append(["rule", f"{name}{N}", top(order, N, drop)])
        for _, nm, hs in sets:
            inter = len({tuple(h) for h in hs} & {tuple(h) for h in core})
            print(f"   {nm:22s} {' '.join(f'L{l}H{h}' for l, h in hs):60s} shares {inter}/{N} with core")
        d = out / f"fold{fold}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "heads.json").write_text(json.dumps({"simple": {t: {"sets": sets} for t in simple.UNSEEN_TASKS}}))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "rebuild" / "newtasks_core")
