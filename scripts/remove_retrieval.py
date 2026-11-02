"""E1b, k = 1 without retrieval heads: each task's k = 1 rule-style sets (``results/consensus_k1``)
minus every head that is a retrieval head of the task (no GPU).

A retrieval head (Yang et al.'s ``retrieval_heads``, ``scripts/identify_retrieval.py``) has the
first token of the prompt's own answer among its top-10 decoded tokens on at least ``RHO`` of
the identification prompts. The k = 1 sets pool the five instruction templates, so a head is
removed if it is a retrieval head under any of the five templates and under either readout.

Writes ``results/consensus_k1_noretrieval/heads.json`` (layout of ``results/heads.json``), which
``scripts/pipeline_consensus.sh`` patches with ``LTH_RESULTS=results/consensus_k1_noretrieval
LTH_SETS="logithead jhead"``, and ``measures.csv``: per task and readout, the k = 1 set, the
heads removed, and the heads E1's retrieval heads (one template, same readout) would remove.

    .venv/bin/python scripts/remove_retrieval.py
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

SOURCE = RESULTS / "consensus_k1"
OUT = RESULTS / "consensus_k1_noretrieval"
STYLE = "rule"


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    from transformers import AutoTokenizer

    from src.config import LENSES

    tokenizer = AutoTokenizer.from_pretrained(LENSES["llama3.1-8b-it"].hf_model_id)
    k1 = json.loads((SOURCE / "heads.json").read_text())["simple"]
    e1_retrieval = json.loads((RESULTS / "retrieval" / "heads.json").read_text())["simple"]
    heads_out, measures = {}, []
    for task in simple.TASKS:
        retrieval = set()  # (layer, head) retrieval under any template and readout
        for wording in simple.WORDINGS[STYLE]:
            run = load(readings_dir("simple", task), wording)
            answers = simple.prompts(task, wording)[1]
            first = np.array([tokenizer(answers[int(i)], add_special_tokens=False).input_ids[0]
                              for i in run["id_items"]])
            for r in range(len(READOUTS)):
                rate = (run["top"][:, r] == first[:, None, None]).any(-1).mean(0)
                retrieval |= {(h // N_HEADS, h % N_HEADS) for h in np.flatnonzero(rate >= RHO)}
        for readout in READOUTS:
            s = k1[task][STYLE][readout]
            kept = [head for head in s["heads"] if (head[0], head[1]) not in retrieval]
            heads_out.setdefault("simple", {}).setdefault(task, {}).setdefault(STYLE, {})[readout] = {
                **s, "rule": f"{s['rule']}, without retrieval heads", "heads": kept}
            narrow = {(l, h) for l, h, _ in e1_retrieval[task][STYLE][readout]["heads"]}
            measures.append({"task": task, "readout": readout, "n_k1": len(s["heads"]), "n_heads": len(kept),
                             "n_removed": len(s["heads"]) - len(kept),
                             "n_removed_e1_rule": sum((l, h) in narrow for l, h, _ in s["heads"]),
                             "n_retrieval_any": len(retrieval)})
        print(task, {m["readout"]: f"{m['n_heads']} of {m['n_k1']}" for m in measures[-2:]}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "heads.json").write_text(json.dumps(heads_out, indent=1))
    with (OUT / "measures.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(measures[0]))
        writer.writeheader()
        writer.writerows(measures)
    print(f"wrote {OUT}/heads.json, measures.csv")


if __name__ == "__main__":
    main()
