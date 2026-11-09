"""E8: head sets from every combination of five per-head filters, one from the J-head lens and four
from last-token attention. J-head only, one E4 fold, all scores on the choosing half.

Per task and head, on the choosing half's rule prompts (five wordings pooled, 500 prompts;
``--choose A`` for fold 1, ``B`` for fold 2), at the last prompt token:

- ``D`` decoding: share of prompts whose J-head top 10 holds a token of the task's word list;
  accept if >= 0.10 (the E7 / CC-pooled rule). The list is the task's 30 most active decoded
  words, E7's word rules (``scripts/e7_sets.py``: top-10 tokens decoded, lowercased, alphabetic,
  at least three letters, not in ``STOP``, Zipf >= 3; activity = share of (prompt, head) pairs
  whose top 10 holds the word), nothing removed. Its tokens come from
  ``lth.terms.expand_lowercase_to_token_ids``, which adds `` Canada`` / `` CANADA`` to Yang et
  al.'s expansion. (First run: E7's 10-word ``raw`` list with Yang's expansion only, which never
  matched the leading-space capitalised tokens the lens decodes; commit 4d19133.)
- ``B`` BOS: mean attention on BOS; accept if <= 0.5 (the E7 filter);
- ``T`` task words: mean attention on the instruction's task-word tokens; accept if >= 0.05;
- ``I`` input: mean attention on the query (the item the task is applied to); accept if >= 0.05;
- ``R`` rest: mean attention on the rest of the instruction; accept if <= 0.5.

The attention masses (task words, query, rest, BOS) are ``data/readings/heldout/attention_<half>``
(``scripts/e3/attention_pass.py``). A set is named by its filters in the order D, B, T, I, R
(``DB`` = D and B); a head is in it if it passes all of them. All 31 non-empty combinations, all
layers, no size cap; an empty set is left out for that task. No controls.

Writes ``--out``: ``heads.json`` (set-list layout of ``scripts/rank_heads.py``), ``scores.npz``,
``word_lists.json`` and ``set_sizes.csv``.

    .venv/bin/python scripts/e8_sets.py --choose A --out results/e8_filters/fold1
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, READOUTS, RESULTS, simple  # noqa: E402

N_HEADS = 32
R = READOUTS.index("jhead")
FILTERS = "DBTIR"
COMBOS = ["".join(c) for n in range(1, 6) for c in itertools.combinations(FILTERS, n)]
N_WORDS = 30
STOP = set("the and for with that this from are was were you your has have had not but its".split())  # as E7
WORD = re.compile(r"^[a-z]{3,}$")


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
    from wordfreq import zipf_frequency

    from lth.terms import expand_lowercase_to_token_ids

    tok = get_tok()
    cache: dict[int, str | None] = {}

    def word(t: int):  # E7's word rules
        if t not in cache:
            w = tok.decode([t]).strip().lower()
            cache[t] = w if WORD.match(w) and w not in STOP and zipf_frequency(w, "en") >= 3 else None
        return cache[t]

    root = DATA / "readings" / "heldout" / args.choose
    out, scores, sizes, word_lists = {}, {}, [], {}
    for t in simple.TASKS:
        pooled = np.concatenate([load(root / "simple" / t, w)["top"][:, R] for w in simple.WORDINGS["rule"]])
        c = Counter()
        for p in range(pooled.shape[0]):
            for h in range(pooled.shape[1]):
                c.update({w for x in pooled[p, h] if (w := word(int(x)))})
        words = sorted(c, key=lambda w: (-c[w], w))[:N_WORDS]
        word_lists[t] = {w: round(c[w] / (pooled.shape[0] * pooled.shape[1]), 4) for w in words}
        ids = np.array(expand_lowercase_to_token_ids(tok, words))
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
    (args.out / "word_lists.json").write_text(json.dumps({"n_words": N_WORDS, "tasks": word_lists}, indent=1))
    np.savez_compressed(args.out / "scores.npz", **scores)
    with (args.out / "set_sizes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", *COMBOS])
        w.writeheader()
        w.writerows(sizes)


if __name__ == "__main__":
    main()
