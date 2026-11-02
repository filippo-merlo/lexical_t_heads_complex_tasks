"""E1b: rule-style lexical task heads that pass Definition 1 under every instruction template (no GPU).

For each simple task and readout, a head is in the consensus set when it matches the task's
term list on at least ``RHO`` of the identification prompts of each of the five instruction
templates separately (the first 100 prompts in P+ of each, ``scripts/build_readings.py``);
its rate is the lowest of the five. Only the rule style: E1 identifies its rule heads from
one template, the most accurate, and E1b asks whether requiring all five gives heads with
larger causal effects. The example style is not part of E1b.

``--min-templates k`` (default 5) relaxes "every template" to "at least k of the five"; the
head's rate is then its k-th highest. k = 1 is the union of the five templates' sets, which
contains E1's rule set; for k < 5 the set is written to ``results/consensus_k{k}``.

Writes ``results/consensus/heads.json`` in the layout of ``results/heads.json`` (family
``simple``, style ``rule``), so that ``scripts/patch.py`` runs on it with
``LTH_RESULTS=results/consensus``; and ``results/consensus/measures.csv``: set sizes, mean
layer, copy share over the five templates' prompts, and the overlap with E1's rule set.

    .venv/bin/python scripts/identify_consensus.py [--min-templates 3]
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

from identify_heads import N_HEADS, RHO, copy_share, output_prompts  # noqa: E402

from lth import READOUTS, RESULTS, simple  # noqa: E402
from lth.store import load, readings_dir  # noqa: E402

OUT = RESULTS / "consensus"
STYLE = "rule"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--min-templates", type=int, default=5, choices=range(1, 6))
    k = parser.parse_args().min_templates
    out = OUT if k == 5 else RESULTS / f"consensus_k{k}"
    from transformers import AutoTokenizer

    from lth.terms import expand_to_token_ids, simple_term_lists
    from src.config import LENSES

    tokenizer = AutoTokenizer.from_pretrained(LENSES["llama3.1-8b-it"].hf_model_id)
    single = json.loads((RESULTS / "heads.json").read_text())["simple"]
    terms = simple_term_lists()
    wordings = simple.WORDINGS[STYLE]
    heads_out, measures = {}, []
    for task in simple.TASKS:
        directory = readings_dir("simple", task)
        ids = np.array(sorted(set(expand_to_token_ids(tokenizer, terms[task]))))
        runs = {w: load(directory, w) for w in wordings}
        texts = []
        for w in wordings:
            by_item = output_prompts("simple", task, w)
            texts += [by_item[item] for item in runs[w]["id_items"]]
        top = np.concatenate([runs[w]["top"] for w in wordings])
        for r, readout in enumerate(READOUTS):
            rates = np.stack([np.isin(runs[w]["top"][:, r], ids).any(-1).mean(0) for w in wordings])  # [wording, head]
            lowest = np.sort(rates, 0)[-k]  # k-th highest rate over the templates
            heads = [int(h) for h in np.flatnonzero(lowest >= RHO)]
            heads_out.setdefault("simple", {}).setdefault(task, {}).setdefault(STYLE, {})[readout] = {
                "wordings": list(wordings), "n_prompts": int(len(top)), "rule": "all templates" if k == 5 else f"at least {k} templates",
                "heads": [[h // N_HEADS, h % N_HEADS, round(float(lowest[h]), 4)] for h in heads]}
            e1 = {l * N_HEADS + h for l, h, _ in single[task][STYLE][readout]["heads"]}
            matched = np.isin(top[:, r], ids).any(-1)
            measures.append({
                "task": task, "readout": readout, "n_prompts": len(top), "n_heads": len(heads),
                "n_e1": len(e1), "n_both": len(e1 & set(heads)),
                "mean_layer": float(np.mean([h // N_HEADS for h in heads])) if heads else float("nan"),
                "copy_share": copy_share(top[:, r], matched, texts, heads, set(ids.tolist()), tokenizer)})
        sets = heads_out["simple"][task][STYLE]
        print(f"{task}: " + ", ".join(f"{r[:5]} {len(sets[r]['heads'])}" for r in READOUTS), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "heads.json").write_text(json.dumps(heads_out, indent=1))
    with (out / "measures.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(measures[0]))
        writer.writeheader()
        writer.writerows(measures)
    print(f"wrote {out}/heads.json, measures.csv")


if __name__ == "__main__":
    main()
