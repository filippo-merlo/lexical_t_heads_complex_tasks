"""E22 step 1 (no GPU): every combination of the seven filters D, B, T, L, M, S, N at N = 40, per task
and shared, except those holding N with T or S: N is attention to the whole instruction, which
already counts T's and S's positions (user, 2026-11-13). 79 combinations x 2 scopes.

Scores and combination rule as ``scripts/e21_sets.py`` (E13's: per task, each filter's within-task
rank averaged over the combination's filters; shared, E10's rule), ties to the lower layer, then
head. Letter order in names: D B T L M S N (e.g. ``DBS40``, ``shared_BTLN40``).

The 19 combinations already run (E13: D, B, T, DB, DT, BT, DBT; E21: X, BX, DBX for X in L, M, S, N)
are rebuilt and must equal the patched sets exactly; only the other 60 are written.

Writes ``results/e22_all_combos/fold{f}/heads.json`` (``scripts/patch.py`` set-list layout) and
``sets.csv`` (size, median layer, identical sets among all 79 per task and fold).

    .venv/bin/python scripts/e22_sets.py
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS, simple  # noqa: E402

from e13_sets import ranks, top  # noqa: E402
from e20_bai_attention import CATS  # noqa: E402
from e21_sets import PARTS, SIGN  # noqa: E402

OUT = RESULTS / "e22_all_combos"
HALF = {1: "A", 2: "B"}
N = 40
LETTERS = "DBTLMSN"
COMBOS = ["".join(c) for k in range(1, len(LETTERS) + 1) for c in itertools.combinations(LETTERS, k)
          if not ("N" in c and ("T" in c or "S" in c))]
DONE = {"e13_capped": ("D", "B", "T", "DB", "DT", "BT", "DBT"),
        "e21_bai_rules": tuple(p + x for x in "LMSN" for p in ("", "B", "DB"))}


def main() -> None:
    assert len(COMBOS) == 79
    done = {c: exp for exp, cs in DONE.items() for c in cs}
    for fold, half in HALF.items():
        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        dsc = np.load(RESULTS / "e13_capped" / f"fold{fold}" / "d_scores.npz")
        att = {t: np.load(DATA / "readings" / "heldout" / f"attention20_{half}" / f"{t}.npy").astype(np.float64).mean(0)
               for t in simple.TASKS}
        score = {t: {"D": dsc[t], **{f: att[t][:, [CATS.index(c) for c in cs]].sum(1) for f, cs in PARTS.items()}}
                 for t in simple.TASKS}
        old = {exp: {t: {name: hs for _, name, hs in json.loads((RESULTS / exp / f"fold{fold}" / "heads.json")
                                                               .read_text())["simple"][t]["sets"]}
                     for t in simple.TASKS} for exp in DONE}

        per_task = {t: {} for t in simple.TASKS}
        for t in simple.TASKS:
            r = {f: ranks(SIGN[f] * score[t][f]) for f in LETTERS}
            for c in COMBOS:
                per_task[t][f"{c}{N}"] = top(np.argsort(np.mean([r[f] for f in c], axis=0), kind="stable"), N)
        within = {f: np.stack([ranks(SIGN[f] * score[t][f]) for t in simple.TASKS]).mean(0) for f in LETTERS}
        combined = {f: ranks(-within[f]) for f in LETTERS}
        shared = {f"shared_{c}{N}": top(np.argsort(np.mean([combined[f] for f in c], axis=0), kind="stable"), N)
                  for c in COMBOS}

        for c, exp in done.items():
            for t in simple.TASKS:
                assert per_task[t][f"{c}{N}"] == old[exp][t][f"{c}{N}"], (fold, t, c)
                assert shared[f"shared_{c}{N}"] == old[exp][t][f"shared_{c}{N}"], (fold, c)
        print(f"fold {fold}: the {len(done)} combinations already run are reproduced (per task and shared)")

        new = [c for c in COMBOS if c not in done]
        heads = {"simple": {t: {"sets": [["rule", f"{c}{N}", per_task[t][f"{c}{N}"]] for c in new]
                                + [["rule", f"shared_{c}{N}", shared[f"shared_{c}{N}"]] for c in new]}
                            for t in simple.TASKS}}
        (out / "heads.json").write_text(json.dumps(heads, indent=1))

        rows = []
        for t in simple.TASKS:
            for scope, sets in (("task", per_task[t]), ("shared", shared)):
                key = {name: tuple(map(tuple, hs)) for name, hs in sets.items()}
                for name, hs in key.items():
                    c = name.removeprefix("shared_")[:-len(str(N))]
                    twins = sorted(n for n, h in key.items() if h == hs and n != name)
                    rows.append(dict(fold=fold, task=t, scope=scope, set=name, combo=c, run_in=done.get(c, "e22_all_combos"),
                                     n_heads=len(hs), layer_median=float(np.median([l for l, _ in hs])),
                                     identical_to=" ".join(twins)))
        with (out / "sets.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"fold {fold}: {len(new)} new combinations x 2 scopes written")


if __name__ == "__main__":
    main()
