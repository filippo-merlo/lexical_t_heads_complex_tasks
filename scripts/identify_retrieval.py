"""Retrieval heads of the simple tasks (Yang et al.'s ``retrieval_heads``), and which lexical task
heads are also retrieval heads (no GPU).

Yang et al. identify retrieval heads as lexical task heads, with one change: a head matches on a
prompt if the first token of that prompt's own answer is among its top-10 decoded tokens
(``calculate_component_metrics_per_head`` with ``component_type="retrieval_heads"``), instead of
any token of the task's term list. The prompts, ``k = 10``, ``n_match = 1`` and the threshold
``RHO`` are those of Definition 1: E1's identification prompts of each task and style (the most
accurate instruction template, or 5 shots), read from ``data/readings`` under both readouts.

Writes ``results/retrieval/heads.json`` (family ``simple``, the layout of ``results/heads.json``)
and ``results/retrieval/overlap.csv``: for each head-set procedure (E1; E1b k = 1, 3, 5 and the
k = 1 shared and unshared J-head sets, rule style), task and readout, the set's size and how many
of its heads are retrieval heads of the same task, style and readout.

    .venv/bin/python scripts/identify_retrieval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from identify_heads import N_HEADS, RHO  # noqa: E402

from lth import READOUTS, RESULTS, simple  # noqa: E402
from lth.store import load, readings_dir  # noqa: E402

OUT = RESULTS / "retrieval"
PROCEDURES = {"E1": RESULTS, "k=1": RESULTS / "consensus_k1", "k=3": RESULTS / "consensus_k3",
              "k=5": RESULTS / "consensus", "k=1 shared": RESULTS / "consensus_k1_shared",
              "k=1 unshared": RESULTS / "consensus_k1_unshared"}


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    from transformers import AutoTokenizer

    from src.config import LENSES

    tokenizer = AutoTokenizer.from_pretrained(LENSES["llama3.1-8b-it"].hf_model_id)
    e1 = json.loads((RESULTS / "heads.json").read_text())["simple"]
    heads_out = {}
    for task in simple.TASKS:
        for style, sets in e1[task].items():
            wording = sets["logithead"]["wordings"][0]  # E1's identification wording of the style
            run = load(readings_dir("simple", task), wording)
            answers = simple.prompts(task, wording)[1]
            first = np.array([tokenizer(answers[int(i)], add_special_tokens=False).input_ids[0]
                              for i in run["id_items"]])
            for r, readout in enumerate(READOUTS):
                rate = (run["top"][:, r] == first[:, None, None]).any(-1).mean(0)  # [head]
                heads = [int(h) for h in np.flatnonzero(rate >= RHO)]
                heads_out.setdefault("simple", {}).setdefault(task, {}).setdefault(style, {})[readout] = {
                    "wordings": [wording], "n_prompts": int(len(first)), "rule": "answer token in top 10",
                    "heads": [[h // N_HEADS, h % N_HEADS, round(float(rate[h]), 4)] for h in heads]}
        print(task, {s: {r: len(v["heads"]) for r, v in d.items()} for s, d in heads_out["simple"][task].items()},
              flush=True)

    rows = []
    for name, directory in PROCEDURES.items():
        path = directory / "heads.json"
        if not path.exists():
            continue
        sets = json.loads(path.read_text())["simple"]
        for task in simple.TASKS:
            for style in sets.get(task, {}):
                for readout in READOUTS:
                    heads = {(l, h) for l, h, _ in sets[task][style].get(readout, {}).get("heads", [])}
                    if not heads:
                        continue
                    retrieval = {(l, h) for l, h, _ in heads_out["simple"][task][style][readout]["heads"]}
                    both = sorted(heads & retrieval)
                    rows.append({"procedure": name, "task": task, "style": style, "readout": readout,
                                 "n_heads": len(heads), "n_retrieval": len(retrieval), "n_both": len(both),
                                 "both": " ".join(f"L{l}H{h}" for l, h in both)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "heads.json").write_text(json.dumps(heads_out, indent=1))
    with (OUT / "overlap.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT}/heads.json, overlap.csv")


if __name__ == "__main__":
    main()
