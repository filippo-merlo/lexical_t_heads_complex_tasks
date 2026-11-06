"""E4 (paper/e4_heldout.tex): the item split of each simple task and the counts it leaves.

The items of each task (file order, ``lth.simple.dataset``) are shuffled with seed 0 and cut
into two halves, A and B (one more in A when odd). Fold 1 chooses on A and tests on B, fold 2
the reverse. Writes ``results/heldout/split.json`` ({task: {"A": [...], "B": [...]}}, item ids
sorted) and ``results/heldout/counts.csv``: per task and half, the items, the correct prompts
of each rule wording (identification prompts and means, at most ``CAP`` used), the items
answered correctly and wrongly 1-shot (patched items of checks 1 and 2 when the half tests, at
most ``CAP``), and the control task's items correct under inst1 in that half (control mean).

    .venv/bin/python scripts/heldout_split.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import RESULTS, simple  # noqa: E402
from lth.store import load, readings_dir  # noqa: E402

SEED = 0
OUT = RESULTS / "heldout"


def split(n: int) -> dict[str, list[int]]:
    order = np.random.default_rng(SEED).permutation(n)
    cut = (n + 1) // 2
    return {"A": sorted(map(int, order[:cut])), "B": sorted(map(int, order[cut:]))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    halves = {t: split(len(simple.dataset(t))) for t in simple.TASKS}
    for t, h in halves.items():
        assert not set(h["A"]) & set(h["B"]) and len(h["A"]) + len(h["B"]) == len(simple.dataset(t))
    (OUT / "split.json").write_text(json.dumps(halves))

    def correct(task, wording):
        return {int(r["item"]): bool(r["correct"]) for r in load(readings_dir("simple", task), wording)["records"]}

    rows = []
    for t in simple.TASKS:
        shot = correct(t, simple.FIXING)
        control = correct(simple.CONTROL[t], simple.CONTROL_ITEMS["rule"])
        rule = {w: correct(t, w) for w in simple.WORDINGS["rule"]}
        for half in ("A", "B"):
            items = halves[t][half]
            ctrl_items = set(halves[simple.CONTROL[t]][half])
            row = dict(task=t, half=half, items=len(items))
            row.update({f"correct_{w}": sum(rule[w][i] for i in items) for w in simple.WORDINGS["rule"]})
            row["correct_shot1"] = sum(shot[i] for i in items)
            row["wrong_shot1"] = sum(not shot[i] for i in items)
            row["control_task"] = simple.CONTROL[t]
            row["control_items"] = sum(v for i, v in control.items() if i in ctrl_items)
            rows.append(row)
    with (OUT / "counts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import pandas as pd
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
