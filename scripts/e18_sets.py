"""E18 step 3 (no GPU): E13's filter sets rebuilt on the 20- and 40-word lists (``e18_lists.py``).

E13's rules (``e13_sets.py``) unchanged; only the list behind D and T changes length. D from
``results/e18_list_length/w{n}/fold{f}/d_scores.npz``. B does not use the list: E13's
(``attention13_<half>``), checked against the new pass. T from the new attention pass
(``attention18w{n}_<half>``), except on a task where the list words found in the prompts are the
same as with E13's list: the task-word positions are then the same, and E13's T is used (checked
against the rerun), so that a set equal to E13's is not changed by rerun noise.

Sets: D, T, DB, DT, BT, DBT and DBT minus the fold's E13 generic heads, per task and shared, N 20-80,
E13's names. Only the sets that differ from E13's set of the same name (same fold and task) go into
``heads.json`` for patching; an equal set takes E13's results (same-alpha reruns are identical).

Checks: the same code on E13's scores gives E13's sets; B and the reused T agree with the rerun.

Writes ``results/e18_list_length/w{n}/fold{f}/heads.json`` (``scripts/patch.py`` set-list layout,
differing sets only), ``sets.csv`` (every set: size, median layer, overlap with E13's, whether it
equals E13's) and ``t_source.json`` (per task: E13's T or the new pass, and the words found).

    .venv/bin/python scripts/e18_sets.py
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

OUT = RESULTS / "e18_list_length"
E13 = RESULTS / "e13_capped"
HALF = {1: "A", 2: "B"}
LENGTHS = (20, 40)
SIZES = (20, 40, 60, 80)
COMBOS = ("D", "T", "DB", "DT", "BT", "DBT")
ATT = {"T": 0, "B": 3}
SIGN = {"D": 1, "T": 1, "B": -1}
N_HEADS = 32
ATOL = 1e-3


def build(score: dict, generic: set) -> tuple[dict, list]:
    """E13's per-task and shared rules for the filter sets that use the list."""
    from e13_sets import ranks, top

    per_task = {t: [] for t in simple.TASKS}
    for t in simple.TASKS:
        r = {f: ranks(SIGN[f] * score[t][f]) for f in "DBT"}
        for n in SIZES:
            for c in COMBOS:
                order = np.argsort(np.mean([r[f] for f in c], axis=0), kind="stable")
                per_task[t].append(["rule", f"{c}{n}", top(order, n)])
                if c == "DBT":
                    per_task[t].append(["rule", f"DBT{n}_nogen", top(order, n, generic)])
    within = {f: np.stack([ranks(SIGN[f] * score[t][f]) for t in simple.TASKS]).mean(0) for f in "DBT"}
    combined = {f: ranks(-within[f]) for f in "DBT"}
    shared = []
    for n in SIZES:
        for c in COMBOS:
            order = np.argsort(np.mean([combined[f] for f in c], axis=0), kind="stable")
            shared.append(["rule", f"shared_{c}{n}", top(order, n)])
            if c == "DBT":
                shared.append(["rule", f"shared_DBT{n}_nogen", top(order, n, generic)])
    return per_task, shared


def masses(att_dir: Path, t: str) -> np.ndarray:
    return np.load(att_dir / f"{t}.npy").astype(np.float64).mean(0)  # 1024 x 4


def found_words(att_dir: Path, t: str) -> set[str]:
    info = json.loads((att_dir / "info.json").read_text())[t]["found"]
    return {w for per_template in info.values() for w in per_template}


def main() -> None:
    for fold, half in HALF.items():
        names = json.loads((E13 / f"fold{fold}" / "generic.json").read_text())["generic"]
        generic = {int(s[1:].split("H")[0]) * N_HEADS + int(s.split("H")[1]) for s in names}
        att13 = DATA / "readings" / "heldout" / f"attention13_{half}"
        d13 = np.load(E13 / f"fold{fold}" / "d_scores.npz")
        old = {t: {"D": d13[t], "B": masses(att13, t)[:, ATT["B"]], "T": masses(att13, t)[:, ATT["T"]]}
               for t in simple.TASKS}
        e13_sets = {t: {name: hs for _, name, hs in v["sets"]}
                    for t, v in json.loads((E13 / f"fold{fold}" / "heads.json").read_text())["simple"].items()}
        o_task, o_shared = build(old, generic)
        for t in simple.TASKS:
            for _, name, hs in o_task[t] + o_shared:
                assert e13_sets[t][name] == hs, f"fold {fold}: E13 set {name} on {t} not reproduced"

        for n in LENGTHS:
            out = OUT / f"w{n}" / f"fold{fold}"
            att = DATA / "readings" / "heldout" / f"attention18w{n}_{half}"
            dsc = np.load(out / "d_scores.npz")
            new, t_source = {}, {}
            for t in simple.TASKS:
                m = masses(att, t)
                assert np.allclose(m[:, ATT["B"]], old[t]["B"], atol=ATOL), f"w{n} fold {fold}: B differs on {t}"
                same_positions = found_words(att, t) == found_words(att13, t)
                if same_positions:
                    assert np.allclose(m[:, ATT["T"]], old[t]["T"], atol=ATOL), f"w{n} fold {fold}: T rerun differs on {t}"
                new[t] = {"D": dsc[t], "B": old[t]["B"], "T": old[t]["T"] if same_positions else m[:, ATT["T"]]}
                t_source[t] = dict(source="e13" if same_positions else "new pass",
                                   found=sorted(found_words(att, t)), found_e13=sorted(found_words(att13, t)))
            per_task, shared = build(new, generic)
            todo, rows = {}, []
            for t in simple.TASKS:
                sets = per_task[t] + shared
                differ = [s for s in sets if s[2] != e13_sets[t][s[1]]]
                if differ:
                    todo[t] = {"sets": differ}
                for _, name, hs in sets:
                    a, b = {tuple(h) for h in hs}, {tuple(h) for h in e13_sets[t][name]}
                    rows.append(dict(fold=fold, task=t, set=name, n_heads=len(a),
                                     layer_median=float(np.median([l for l, _ in a])),
                                     overlap_e13=len(a & b), equal_e13=int(a == b)))
            (out / "heads.json").write_text(json.dumps({"simple": todo}, indent=1))
            (out / "t_source.json").write_text(json.dumps(t_source, ensure_ascii=False, indent=1))
            with (out / "sets.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            n_diff = sum(len(v["sets"]) for v in todo.values())
            print(f"w{n} fold {fold}: T from new pass on {[t for t, v in t_source.items() if v['source'] != 'e13']}; "
                  f"{n_diff} of {len(rows)} task x set pairs differ from E13 and will be patched")


if __name__ == "__main__":
    main()
