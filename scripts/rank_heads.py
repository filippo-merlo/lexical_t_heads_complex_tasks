"""Head sets of equal size (no GPU): the top N heads of each selection rule and readout.

E1 and E1b select heads by a threshold (Definition 1: rate >= RHO), so their sets differ in
size between readouts and between rules. Here every head of the model is ranked by the
quantity the threshold is applied to, and the first N are kept, for N in ``SIZES``:
- ``e1``: the head's match rate on E1's identification prompts of the style (the most
  accurate instruction template for the rule style, 5 shots for the example style;
  ``results/heads.json``);
- ``k1``, ``k3``, ``k5`` (rule style): the k-th highest of its match rates under the five
  instruction templates, as ``scripts/identify_consensus.py``.
Ties are broken by the mean number of term tokens among the head's top 10 tokens on the same
prompts (for ``k``, the k-th highest of the five templates' means), then by layer and head.
Only heads with a rate above 0 are ranked: where fewer than N heads match on any prompt, the
set of size N is not formed (it would be filled with heads that never name the task).

The threshold sets of E1 and E1b are recomputed from the same rates and checked against
``results/heads.json`` and ``results/consensus*/heads.json``; the script stops if they differ.

Writes ``results/sized/heads.json``: per task, the list ``sets`` of ``[style, name, heads]``
with name ``readout|rule|N`` (for ``scripts/patch.py --set-list``); and
``results/sized/measures.csv``: per task, rule, readout and N, how many of the N heads meet
the rule's threshold, the threshold set's size, the lowest rate kept, the mean layer, and the
heads shared with the other readout's top N (empty where either set is not formed).

    .venv/bin/python scripts/rank_heads.py
"""

from __future__ import annotations

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

OUT = RESULTS / "sized"
SIZES = (5, 10, 20)
CONSENSUS = {1: "consensus_k1", 3: "consensus_k3", 5: "consensus"}


def scores(top: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per head: the share of prompts with a term among its top tokens, and the mean number
    of term tokens among them. ``top``: [prompts, heads, 10]."""
    hits = np.isin(top, ids)
    return hits.any(-1).mean(0), hits.sum(-1).mean(0)


def ranked(rate: np.ndarray, count: np.ndarray) -> list[int]:
    """Heads by rate, then term count, then index (``np.lexsort`` sorts by its last key first)."""
    return [int(h) for h in np.lexsort((np.arange(len(rate)), -count, -rate))]


def main() -> None:
    from transformers import AutoTokenizer

    from lth.terms import expand_to_token_ids, simple_term_lists
    from src.config import LENSES

    tokenizer = AutoTokenizer.from_pretrained(LENSES["llama3.1-8b-it"].hf_model_id)
    terms = simple_term_lists()
    e1 = json.loads((RESULTS / "heads.json").read_text())["simple"]
    consensus = {k: json.loads((RESULTS / d / "heads.json").read_text())["simple"] for k, d in CONSENSUS.items()}
    heads_out, measures = {}, []
    for task in simple.TASKS:
        directory = readings_dir("simple", task)
        ids = np.array(sorted(set(expand_to_token_ids(tokenizer, terms[task]))))
        templates = {w: load(directory, w)["top"] for w in simple.WORDINGS["rule"]}
        rules = []  # (style, rule, readout, rate, count, threshold set it must reproduce)
        for style in simple.WORDINGS:
            for r, readout in enumerate(READOUTS):
                recorded = e1[task][style][readout]
                (wording,) = recorded["wordings"]
                top = templates[wording] if style == "rule" else load(directory, wording)["top"]
                rate, count = scores(top[:, r], ids)
                rules.append((style, "e1", readout, rate, count, {(l, h) for l, h, _ in recorded["heads"]}))
        for k in CONSENSUS:
            for r, readout in enumerate(READOUTS):
                per = [scores(top[:, r], ids) for top in templates.values()]
                rate = np.sort(np.stack([p[0] for p in per]), 0)[-k]
                count = np.sort(np.stack([p[1] for p in per]), 0)[-k]
                recorded = consensus[k][task]["rule"][readout]
                rules.append(("rule", f"k{k}", readout, rate, count, {(l, h) for l, h, _ in recorded["heads"]}))

        sets, tops = [], {}
        for style, rule, readout, rate, count, recorded in rules:
            threshold = {(h // N_HEADS, h % N_HEADS) for h in np.flatnonzero(rate >= RHO)}
            if threshold != recorded:
                raise SystemExit(f"{task} {style} {rule} {readout}: threshold set differs from the recorded one")
            order = [h for h in ranked(rate, count) if rate[h] > 0]
            for n in SIZES:
                row = {"task": task, "style": style, "rule": rule, "readout": readout, "n": n,
                       "n_matching": len(order), "n_threshold": len(threshold), "formed": len(order) >= n}
                if row["formed"]:
                    chosen = [(h // N_HEADS, h % N_HEADS) for h in order[:n]]
                    tops[(style, rule, readout, n)] = set(chosen)
                    sets.append([style, f"{readout}|{rule}|{n}", [list(h) for h in chosen]])
                    row.update(n_pass=len(set(chosen) & threshold), lowest_rate=round(float(rate[order[n - 1]]), 4),
                               mean_layer=float(np.mean([l for l, _ in chosen])))
                measures.append(row)
        for row in measures:
            if row["task"] == task:
                other = "jhead" if row["readout"] == "logithead" else "logithead"
                key = (row["style"], row["rule"], row["readout"], row["n"])
                pair = (*key[:2], other, key[3])
                row["n_shared_other_readout"] = len(tops[key] & tops[pair]) if key in tops and pair in tops else ""
        heads_out.setdefault("simple", {})[task] = {"sets": sets}
        print(f"{task}: {len(sets)} sets", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "heads.json").write_text(json.dumps(heads_out, indent=1))
    with (OUT / "measures.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(measures[0]))
        writer.writeheader()
        writer.writerows(measures)
    print(f"wrote {OUT}/heads.json, measures.csv")


if __name__ == "__main__":
    main()
