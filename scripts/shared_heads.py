"""E1b, shared heads: each task's J-head set for k = 1 (``results/consensus_k1``), restricted to
the heads that the k = 1 J-head sets of at least ``--min-tasks`` tasks contain (no GPU).

Writes ``results/consensus_k1_shared/heads.json`` in the layout of ``results/heads.json``
(family ``simple``, style ``rule``) with the restricted J-head sets and empty Logit-head sets,
so that ``scripts/pipeline_consensus.sh`` runs both simple checks on it with
``LTH_RESULTS=results/consensus_k1_shared LTH_SETS=jhead``; and ``measures.csv``: per task,
the heads of the k = 1 set and the shared heads kept.

``--unshared`` keeps the complement instead, the heads of each set that fewer than
``--min-tasks`` sets contain, and writes ``results/consensus_k1_unshared``.

    .venv/bin/python scripts/shared_heads.py [--min-tasks 2] [--unshared]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import RESULTS, simple  # noqa: E402

SOURCE = RESULTS / "consensus_k1"
OUT = RESULTS / "consensus_k1_shared"
STYLE, READOUT = "rule", "jhead"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--min-tasks", type=int, default=2)
    parser.add_argument("--unshared", action="store_true")
    args = parser.parse_args()
    min_tasks = args.min_tasks
    out = RESULTS / "consensus_k1_unshared" if args.unshared else OUT
    source = json.loads((SOURCE / "heads.json").read_text())["simple"]
    sets = {task: source[task][STYLE][READOUT] for task in simple.TASKS}
    tasks_per_head = Counter((l, h) for s in sets.values() for l, h, _ in s["heads"])
    heads_out, measures = {}, []
    for task, s in sets.items():
        kept = [head for head in s["heads"] if (tasks_per_head[(head[0], head[1])] >= min_tasks) != args.unshared]
        rule = f"in the sets of {'fewer than' if args.unshared else 'at least'} {min_tasks} tasks"
        heads_out.setdefault("simple", {}).setdefault(task, {})[STYLE] = {
            READOUT: {**s, "rule": f"{s['rule']}, {rule}", "heads": kept},
            "logithead": {"heads": []}}
        measures.append({"task": task, "readout": READOUT, "n_k1": len(s["heads"]), "n_heads": len(kept)})
        print(f"{task}: {len(kept)} of {len(s['heads'])}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "heads.json").write_text(json.dumps(heads_out, indent=1))
    with (out / "measures.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(measures[0]))
        writer.writeheader()
        writer.writerows(measures)
    print(f"wrote {out}/heads.json, measures.csv")


if __name__ == "__main__":
    main()
