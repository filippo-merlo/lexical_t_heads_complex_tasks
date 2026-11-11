"""E12 steps 1-2: the FV generic heads identified on each choosing half only (no GPU).

The no-generic sets of E11 removed ``e10_sets.GENERIC6``, six heads read off a generic score
computed on prompts of both halves (``scripts/e3/fv_generic_heads.py``, ``data/readings/simple``).
Here the same score is computed on the choosing half of each fold only (fold 1: A, fold 2: B;
``data/readings/heldout/{A,B}/simple``), so that the testing half is used for nothing but testing.

Score (unchanged): per head and task, token counts over the head's J-head top 10 at the last token
of the correct inst1-inst4 prompts (each token once per prompt); the cosine of every pair of the 14
tasks' count vectors, averaged (1 = the same tokens on every task).

Rule (fixed before looking, user 2026-11-11): sort the fold's 30 FV heads
(``results/heldout/fold{f}/fv/heads.json``) by the score and cut at the largest drop between
neighbours; the heads above it are generic (on a tie, the first drop, i.e. fewer heads). If the cut
would mark more than half of the 30, nothing is written and the script stops.

Check before use: the same code on ``data/readings/simple`` must reproduce the saved all-item
J-head scores of ``results/davidson_fv_diff/generic_heads.json``.

Writes ``results/e12_generic_halves/fold{f}/generic.json``: every head's score, the fold's FV-30
order with scores and drops, the generic heads, and the two sets built from them (FV-30 and
FV-univ-20 of E11, minus the generic heads), next to E11's sets; and ``heads.json``, FV-univ-20
minus the generic heads in ``scripts/patch.py``'s set-list layout (the FV-30 set is read by
``scripts/e11_fv_residual.py --sets-from``).

    .venv/bin/python scripts/e12_generic_halves.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from lth import DATA, READOUTS, RESULTS, simple  # noqa: E402
from lth.store import load  # noqa: E402
from e10_sets import GENERIC6  # noqa: E402

N_ALL, N_H = 1024, 32
WORDINGS = ("inst1", "inst2", "inst3", "inst4")
JH = READOUTS.index("jhead")
CHOOSE = {1: "A", 2: "B"}
OUT = RESULTS / "e12_generic_halves"
TASKS = list(simple.TASKS)


def name(s: int) -> str:
    return f"L{s // N_H}H{s % N_H}"


def counts(top) -> Counter:  # top [P, 10] -> Counter of tokens (once per prompt)
    c = Counter()
    for row in top:
        c.update(set(row.tolist()))
    return c


def cos(a: Counter, b: Counter) -> float:
    num = sum(v * b.get(k, 0) for k, v in a.items())
    den = np.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return num / den if den else 0.0


def generic_scores(root: Path, heads) -> dict[int, float]:
    """J-head generic score of each head in ``heads`` from the readings under ``root``."""
    top = {t: np.concatenate([load(root / t.replace("/", "__"), w)["top"][:, JH] for w in WORDINGS])
           for t in TASKS}  # P, 1024, 10
    out = {}
    for s in heads:
        c = {t: counts(top[t][:, s]) for t in TASKS}
        out[s] = float(np.mean([cos(c[a], c[b]) for a, b in combinations(TASKS, 2)]))
    return out


def check_all_items() -> None:
    saved = json.loads((RESULTS / "davidson_fv_diff/generic_heads.json").read_text())["heads"]
    probe = [s for s in range(N_ALL) if saved[name(s)]["fv_rank"] is not None]
    mine = generic_scores(DATA / "readings/simple", probe)
    worst = max(abs(mine[s] - saved[name(s)]["jhead_generic"]) for s in probe)
    assert worst < 1e-9, f"all-item scores not reproduced (max diff {worst})"
    print(f"check: all-item J-head scores of the {len(probe)} all-item FV heads reproduced (max diff {worst:.1e})")


def cut(order: list[int], score: dict[int, float]) -> tuple[list[int], list[float]]:
    s = sorted(order, key=lambda x: -score[x])
    drops = [score[s[i]] - score[s[i + 1]] for i in range(len(s) - 1)]
    k = int(np.argmax(drops))  # first largest drop
    return s[:k + 1], drops


def main() -> None:
    check_all_items()
    for f, half in CHOOSE.items():
        fv = json.loads((RESULTS / "heldout" / f"fold{f}" / "fv/heads.json").read_text())["heads"]
        fv30 = [l * N_H + h for l, h, _ in fv]
        assert len(fv30) == 30
        e11 = {m: [l * N_H + h for l, h in hs] for _, m, hs in
               json.loads((RESULTS / "e11_complete" / f"fold{f}" / "heads.json").read_text())["simple"]["antonym"]["sets"]
               if m in ("fvuniv20", "fvuniv20_nogen")}
        assert set(e11["fvuniv20"]) <= set(fv30), "FV-univ-20 is expected inside the fold's FV-30"
        score = generic_scores(DATA / "readings/heldout" / half / "simple", range(N_ALL))
        generic, drops = cut(fv30, score)
        if len(generic) > 15:
            sys.exit(f"fold {f}: the largest drop marks {len(generic)} of 30 heads as generic; stopping")
        gen = set(generic)
        e11_nogen30 = [s for s in fv30 if (s // N_H, s % N_H) not in GENERIC6]
        sets = {
            "fv30_nogen_half": [s for s in fv30 if s not in gen],
            "fvuniv20_nogen_half": [s for s in e11["fvuniv20"] if s not in gen],
        }
        ordered = sorted(fv30, key=lambda x: -score[x])
        d = OUT / f"fold{f}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "generic.json").write_text(json.dumps(dict(
            fold=f, choosing_half=half, rule="largest drop among the fold's FV-30 by J-head generic score",
            fv30_by_score=[[name(s), round(score[s], 4), fv30.index(s) + 1] for s in ordered],
            drops=[round(x, 4) for x in drops],
            generic=[name(s) for s in generic],
            sets={k: [[s // N_H, s % N_H] for s in v] for k, v in sets.items()},
            e11_sets=dict(fv30_nogen=[[s // N_H, s % N_H] for s in e11_nogen30],
                          fvuniv20_nogen=[[s // N_H, s % N_H] for s in e11["fvuniv20_nogen"]]),
            same_as_e11=dict(fv30_nogen=set(sets["fv30_nogen_half"]) == set(e11_nogen30),
                             fvuniv20_nogen=set(sets["fvuniv20_nogen_half"]) == set(e11["fvuniv20_nogen"])),
            scores={name(s): round(v, 4) for s, v in score.items()},
        ), indent=1))
        # set-list layout of scripts/patch.py (``--set-list``), the same shared set for every task
        hs = [[s // N_H, s % N_H] for s in sets["fvuniv20_nogen_half"]]
        (d / "heads.json").write_text(json.dumps(
            {"simple": {t: {"sets": [["rule", "fvuniv20_nogen_half", hs]]} for t in TASKS}}, indent=1))
        print(f"\nfold {f} (choose on {half}): generic = {[name(s) for s in generic]}")
        for i, s in enumerate(ordered[:12]):
            tag = " <- generic" if s in gen else ""
            old = " (old six)" if (s // N_H, s % N_H) in GENERIC6 else ""
            drop = f"  drop to next {drops[i]:.3f}" if i < len(drops) else ""
            print(f"  {name(s):8s} score {score[s]:.3f}  FV rank {fv30.index(s) + 1:2d}{drop}{tag}{old}")
        print(f"  sets: fv30_nogen_half {len(sets['fv30_nogen_half'])} heads (E11 {len(e11_nogen30)}), "
              f"fvuniv20_nogen_half {len(sets['fvuniv20_nogen_half'])} (E11 {len(e11['fvuniv20_nogen'])})")


if __name__ == "__main__":
    main()
