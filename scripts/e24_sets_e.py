"""E24 step E (no GPU): the other good filter combinations at every size, in both scopes.

User (2026-11-16): add every good combination of at least four letters. "Good": among E23's 15 best
of either scope (``results/e23_best_sizes/chosen.csv``); at least four letters; not already in step
D (DBMN, DBTMS, DBTLS, DBTM). Twelve: DBLN, DBTS, DBLS (per-task list), DBTLM, DBTLMS, DBMS, DBLMN,
DLMN, DBLMS, DBTL, DTLMS, DTLM (shared list; DBLN is on both); and DBS (user: also DBS). Each in both
scopes at N in 20-120.

Sets are built by E22/E23's rule (``scripts/e22_sets.py``: E13/E20 scores of the choosing half, ranks
averaged per task, E10's rule shared) and checked against every set E22 (40 heads) and E23 (other
sizes) patched. They are listed

- under the example style, all of them (patched with the example mean);
- under the rule style, those with no rule-mean run yet: a combination's sets at 20, 60, 80, 100 and
  120 in the scope whose top 15 it is not in (E22 ran every combination at 40 in both scopes).

Random sets are step D's. Writes ``results/e24_cross_style/e/fold{f}/heads.json``.

    .venv/bin/python scripts/e24_sets_e.py
"""

from __future__ import annotations

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
from e22_sets import LETTERS  # noqa: E402

OUT = RESULTS / "e24_cross_style" / "e"
HALF = {1: "A", 2: "B"}
IN_D = ("DBMN", "DBTMS", "DBTLS", "DBTM")
EXTRA = ("DBS",)  # user, 2026-11-16: also DBS (three letters; per-task list, best per-task set at 20 heads)
SCOPES = {"task": "", "shared": "shared_"}


def combos() -> tuple[list[str], dict[str, set[str]]]:
    c = pd.read_csv(RESULTS / "e23_best_sizes" / "chosen.csv")
    top15 = {scope: set(c[c.scope == scope].combo) for scope in SCOPES}
    good = sorted({x for s in top15.values() for x in s if (len(x) >= 4 or x in EXTRA) and x not in IN_D},
                  key=lambda x: (len(x), x))
    return good, top15


def main() -> None:
    good, top15 = combos()
    print(f"{len(good)} combinations: {' '.join(good)}")
    for fold, half in HALF.items():
        dsc = np.load(RESULTS / "e13_capped" / f"fold{fold}" / "d_scores.npz")
        att = {t: np.load(DATA / "readings" / "heldout" / f"attention20_{half}" / f"{t}.npy").astype(np.float64).mean(0)
               for t in simple.TASKS}
        score = {t: {"D": dsc[t], **{f: att[t][:, [CATS.index(c) for c in cs]].sum(1) for f, cs in PARTS.items()}}
                 for t in simple.TASKS}
        known = {}
        for exp in ("e22_all_combos", "e23_best_sizes"):
            for t, v in json.loads((RESULTS / exp / f"fold{fold}" / "heads.json").read_text())["simple"].items():
                for _, name, hs in v["sets"]:
                    known.setdefault(t, {})[name] = hs
        within = {f: np.stack([ranks(SIGN[f] * score[t][f]) for t in simple.TASKS]).mean(0) for f in LETTERS}
        combined = {f: ranks(-within[f]) for f in LETTERS}
        sets = {t: [] for t in simple.TASKS}
        checked = n_rule = 0
        for c in good:
            shared_order = np.argsort(np.mean([combined[f] for f in c], axis=0), kind="stable")
            for t in simple.TASKS:
                r = {f: ranks(SIGN[f] * score[t][f]) for f in c}
                task_order = np.argsort(np.mean([r[f] for f in c], axis=0), kind="stable")
                for scope, prefix in SCOPES.items():
                    order = task_order if scope == "task" else shared_order
                    for n in SIZES:
                        name, hs = f"{prefix}{c}{n}", top(order, n)
                        if name in known[t]:
                            assert hs == known[t][name], (fold, t, name)
                            checked += 1
                        sets[t].append(["example", name, hs])
                        if n != 40 and c not in top15[scope]:
                            assert name not in known[t], name
                            sets[t].append(["rule", name, hs])
                            n_rule += 1
        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "heads.json").write_text(json.dumps({"simple": {t: {"sets": v} for t, v in sets.items()}}, indent=1))
        print(f"fold {fold}: {checked} sets equal E22/E23's; {len(sets[simple.TASKS[0]])} set-styles per task "
              f"({n_rule // len(simple.TASKS)} under the rule style)")


if __name__ == "__main__":
    main()
