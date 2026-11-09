"""E8: head sets from every combination of five per-head filters, one from the J-head lens and four
from last-token attention. J-head only, one E4 fold, all scores on the choosing half.

Per task and head, on the choosing half's rule prompts (five wordings pooled, 500 prompts;
``--choose A`` for fold 1, ``B`` for fold 2), at the last prompt token:

- ``D`` decoding: share of prompts whose J-head top 10 holds a token of the task's E7 ``raw``
  list (``results/e7_attfilter/fold<f>/word_lists.json``, the 10 most active decoded words);
  accept if >= 0.10 (the E7 / CC-pooled rule);
- ``B`` BOS: mean attention on BOS; accept if <= 0.5 (the E7 filter);
- ``T`` task words: mean attention on the instruction's task-word tokens; accept if >= 0.05;
- ``I`` input: mean attention on the query (the item the task is applied to); accept if >= 0.05;
- ``R`` rest: mean attention on the rest of the instruction; accept if <= 0.5.

The attention masses (task words, query, rest, BOS) are ``data/readings/heldout/attention_<half>``
(``scripts/e3/attention_pass.py``). A set is named by its filters in the order D, B, T, I, R
(``DB`` = D and B); a head is in it if it passes all of them. All 31 non-empty combinations, all
layers, no size cap; an empty set is left out for that task. No controls.

Writes ``--out``: ``heads.json`` (set-list layout of ``scripts/rank_heads.py``), ``scores.npz``
and ``set_sizes.csv``.

    .venv/bin/python scripts/e8_sets.py --choose A --out results/e8_filters/fold1
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, READOUTS, RESULTS, simple  # noqa: E402

N_HEADS = 32
R = READOUTS.index("jhead")
FILTERS = "DBTIR"
COMBOS = ["".join(c) for n in range(1, 6) for c in itertools.combinations(FILTERS, n)]


def passes(dec, att) -> dict[str, np.ndarray]:
    """att: heads x [task words, query, rest, BOS]."""
    return {"D": dec >= 0.10, "B": att[:, 3] <= 0.5, "T": att[:, 0] >= 0.05,
            "I": att[:, 1] >= 0.05, "R": att[:, 2] <= 0.5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--choose", choices=("A", "B"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from lth.evaluation import tokenizer as get_tok
    from lth.store import load
    from lth.terms import expand_to_token_ids

    tok = get_tok()
    fold = 1 if args.choose == "A" else 2
    lists = json.loads((RESULTS / "e7_attfilter" / f"fold{fold}" / "word_lists.json").read_text())["tasks"]
    root = DATA / "readings" / "heldout" / args.choose
    out, scores, sizes = {}, {}, []
    for t in simple.TASKS:
        pooled = np.concatenate([load(root / "simple" / t, w)["top"][:, R] for w in simple.WORDINGS["rule"]])
        ids = np.array(sorted(set(expand_to_token_ids(tok, lists[t]["raw"]))))
        dec = np.isin(pooled, ids).any(-1).mean(0)
        att = np.load(DATA / "readings" / "heldout" / f"attention_{args.choose}" / f"{t}.npy").astype(np.float32).mean(0)
        ok = passes(dec, att)
        sets, row = [], {"task": t}
        for c in COMBOS:
            idx = np.flatnonzero(np.logical_and.reduce([ok[f] for f in c]))
            row[c] = len(idx)
            if len(idx):
                sets.append(["rule", c, sorted([int(i) // N_HEADS, int(i) % N_HEADS] for i in idx)])
        out[t] = {"sets": sets}
        scores |= {f"{t}/dec": dec, f"{t}/att": att}
        sizes.append(row)
        print(t, {c: row[c] for c in ("D", "DB", "DBT", "DBTIR")})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "heads.json").write_text(json.dumps({"simple": out}, indent=1))
    np.savez_compressed(args.out / "scores.npz", **scores)
    with (args.out / "set_sizes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", *COMBOS])
        w.writeheader()
        w.writerows(sizes)


if __name__ == "__main__":
    main()
