"""E26: the head sets whose means are summed into one vector (``scripts/e26_summed.py``).

The sets of the capped comparison, unchanged, per fold and task: from E13 (``results/e13_capped``) Yang
terms, BT, DBT, DBT - generic, Causal, Causal - generic and the random sets layer-matched to DBT (five
draws), per task and shared, N = 20 to 120; from E22 (40 heads) and E23 (other sizes) the best further
combinations DBMN, DBTMS, DBTLS, per task and shared. A set with fewer heads than its N (per-task Yang terms
at 60 and 80 on some tasks) is left out, so it has no 14-task mean.

Writes ``results/e26_summed/fold{1,2}/heads.json`` (the ``sets`` list of ``scripts/patch.py --set-list``)
and ``sets.csv``.

    .venv/bin/python scripts/e26_sets.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import RESULTS  # noqa: E402

OUT = RESULTS / "e26_summed"
E13 = re.compile(r"(shared_)?(yang|BT)\d+|(shared_)?(DBT|causal)\d+(_nogen)?|random_(shared_)?dbt\d+_\d")
BEST = re.compile(r"(shared_)?(DBMN|DBTMS|DBTLS)\d+")


def size(name: str) -> int:
    return int(re.search(r"dbt(\d+)_\d$", name).group(1)) if name.startswith("random") else int(
        re.search(r"(\d+)(_nogen)?$", name).group(1))


def main() -> None:
    for fold in (1, 2):
        src = {x: json.loads((RESULTS / x / f"fold{fold}" / "heads.json").read_text())["simple"]
               for x in ("e13_capped", "e22_all_combos", "e23_best_sizes")}
        out, rows = {}, []
        for task in src["e13_capped"]:
            sets = [s for s in src["e13_capped"][task]["sets"] if E13.fullmatch(s[1])]
            sets += [s for s in src["e22_all_combos"][task]["sets"] if BEST.fullmatch(s[1]) and size(s[1]) == 40]
            sets += [s for s in src["e23_best_sizes"][task]["sets"] if BEST.fullmatch(s[1]) and size(s[1]) != 40]
            names = [s[1] for s in sets]
            assert len(names) == len(set(names)), task
            kept = [s for s in sets if len(s[2]) == size(s[1])]
            out[task] = {"sets": kept}
            rows += [dict(fold=fold, task=task, set=name, n=size(name), n_heads=len(heads),
                          kept=int(len(heads) == size(name))) for _, name, heads in sets]
            print(fold, task, len(kept), "sets; left out:", [s[1] for s in sets if s not in kept])
        d = OUT / f"fold{fold}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "heads.json").write_text(json.dumps({"simple": out}))
        with (d / "sets.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
