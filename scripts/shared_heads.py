"""E1b, shared heads: each task's J-head set for k = 1 (``results/consensus_k1``), restricted to
the heads that the k = 1 J-head sets of at least ``--min-tasks`` tasks contain (no GPU).

Writes ``results/consensus_k1_shared/heads.json`` in the layout of ``results/heads.json``
(family ``simple``, style ``rule``) with the restricted J-head sets and empty Logit-head sets,
so that ``scripts/pipeline_consensus.sh`` runs both simple checks on it with
``LTH_RESULTS=results/consensus_k1_shared LTH_SETS=jhead``; and ``measures.csv``: per task,
the heads of the k = 1 set and the shared heads kept.

    .venv/bin/python scripts/shared_heads.py [--min-tasks 2]
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
    min_tasks = parser.parse_args().min_tasks
    source = json.loads((SOURCE / "heads.json").read_text())["simple"]
    sets = {task: source[task][STYLE][READOUT] for task in simple.TASKS}
    tasks_per_head = Counter((l, h) for s in sets.values() for l, h, _ in s["heads"])
    heads_out, measures = {}, []
    for task, s in sets.items():
        kept = [head for head in s["heads"] if tasks_per_head[(head[0], head[1])] >= min_tasks]
        heads_out.setdefault("simple", {}).setdefault(task, {})[STYLE] = {
            READOUT: {**s, "rule": f"{s['rule']}, in the sets of at least {min_tasks} tasks", "heads": kept},
            "logithead": {"heads": []}}
        measures.append({"task": task, "readout": READOUT, "n_k1": len(s["heads"]), "n_heads": len(kept)})
        print(f"{task}: {len(kept)} of {len(s['heads'])}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "heads.json").write_text(json.dumps(heads_out, indent=1))
    with (OUT / "measures.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(measures[0]))
        writer.writeheader()
        writer.writerows(measures)
    print(f"wrote {OUT}/heads.json, measures.csv")


if __name__ == "__main__":
    main()
