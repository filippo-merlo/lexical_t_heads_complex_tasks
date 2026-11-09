"""E7: heads from each task's most active decoded words, with and without the WikiText cut,
then an attention filter that drops heads looking mostly at BOS. J-head only, one E4 fold.

Everything is chosen on the choosing half (``--choose A`` for fold 1, ``B`` for fold 2), with the
word rules of E4 (``scripts/heldout_sets.py``):

- words: a head's top-10 at the last prompt token, decoded, lowercased, alphabetic, at least
  three letters, not in ``STOP``, Zipf frequency at least 3; a word's activity in a task is the
  share of (prompt, head) pairs whose top 10 holds it, over the five rule wordings pooled
  (500 prompts per half);
- ``raw`` list: the task's 10 most active words, nothing removed, no contrast with other tasks;
- ``clean`` list: the same after removing the 50 words with the largest share on the 1,000
  WikiText-103 passages (``data/readings/neutral/wikitext.npz``);
- heads (CC-pooled rule): a list token in the head's top 10 on at least 10% of the 500 pooled
  prompts;
- attention filter (``_att``): drop the heads whose mean last-token attention on BOS over the
  task's rule prompts of this half (``data/readings/heldout/attention_<half>``) is above 0.5.

Controls for each filtered set (seeds 0-4, patched with the task means in check 1 only):
``random_drop_<list>_<i>``, as many heads as ``<list>_att`` drawn at random from the unfiltered
``<list>`` set (does the filter pick the right heads, or does a smaller set just do better?);
``random_layer_<list>_<i>``, the same number of heads in every layer as ``<list>_att``, drawn
outside it (as E4's random sets).

Also reported, not used: the set sizes with the filter at 0.4 and 0.6, and per set how many
heads reach 10% on only one of the five wordings.

Writes ``--out``: ``heads.json`` (set-list layout of ``scripts/rank_heads.py``),
``word_lists.json`` and ``set_sizes.csv``.

    .venv/bin/python scripts/e7_sets.py --choose A --out results/e7_attfilter/fold1
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, READOUTS, simple  # noqa: E402

N_WORDS, RHO, N_HEADS, K_NEUTRAL, BOS_MAX, N_RANDOM = 10, 0.10, 32, 50, 0.5, 5
BOS_SENSITIVITY = (0.4, 0.6)
R = READOUTS.index("jhead")
BOS = 3  # attention masses: task words, query, rest of the instruction, BOS
STOP = set("the and for with that this from are was were you your has have had not but its".split())
WORD = re.compile(r"^[a-z]{3,}$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--choose", choices=("A", "B"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from wordfreq import zipf_frequency

    from lth.evaluation import tokenizer as get_tok
    from lth.store import load
    from lth.terms import expand_to_token_ids

    tok = get_tok()
    root = DATA / "readings" / "heldout" / args.choose
    cache: dict[int, str | None] = {}

    def word(t: int):
        if t not in cache:
            w = tok.decode([t]).strip().lower()
            cache[t] = w if WORD.match(w) and w not in STOP and zipf_frequency(w, "en") >= 3 else None
        return cache[t]

    def shares(top):  # prompts, heads, 10
        c = Counter()
        for p in range(top.shape[0]):
            for h in range(top.shape[1]):
                c.update({w for x in top[p, h] if (w := word(int(x)))})
        return {w: n / (top.shape[0] * top.shape[1]) for w, n in c.items()}

    def rate_of(top, words):
        ids = np.array(sorted(set(expand_to_token_ids(tok, words))))
        return np.isin(top, ids).any(-1).mean(0)

    def heads_of(idx):
        return sorted([int(h) // N_HEADS, int(h) % N_HEADS] for h in idx)

    neutral = shares(np.load(DATA / "readings" / "neutral" / "wikitext.npz")["top"][:, R])
    removed = sorted(neutral, key=lambda w: (-neutral[w], w))[:K_NEUTRAL]

    out, word_lists, sizes = {}, {}, []
    for t in simple.TASKS:
        tops = {w: load(root / "simple" / t, w)["top"][:, R] for w in simple.WORDINGS["rule"]}
        pooled = np.concatenate(list(tops.values()))
        s = shares(pooled)
        ranked = sorted(s, key=lambda w: (-s[w], w))
        lists = {"raw": ranked[:N_WORDS], "clean": [w for w in ranked if w not in removed][:N_WORDS]}
        bos = np.load(DATA / "readings" / "heldout" / f"attention_{args.choose}" / f"{t}.npy")[:, :, BOS].mean(0)
        sets, row = [], dict(task=t, n_prompts=len(pooled))
        for name, words in lists.items():
            rate = rate_of(pooled, words)
            idx = np.flatnonzero(rate >= RHO)
            per_wording = np.stack([rate_of(top, words)[idx] >= RHO for top in tops.values()])
            kept = idx[bos[idx] <= BOS_MAX]
            sets += [["rule", name, heads_of(idx)], ["rule", f"{name}_att", heads_of(kept)]]
            per_layer = Counter(int(h) // N_HEADS for h in kept)
            for i in range(N_RANDOM):
                rng = np.random.default_rng(i)
                sets.append(["rule", f"random_drop_{name}_{i}", heads_of(rng.choice(idx, len(kept), replace=False))])
                pick = []
                for layer in sorted(per_layer):
                    pool = [layer * N_HEADS + x for x in range(N_HEADS) if layer * N_HEADS + x not in set(kept)]
                    pick += [pool[int(j)] for j in rng.choice(len(pool), per_layer[layer], replace=False)]
                sets.append(["rule", f"random_layer_{name}_{i}", heads_of(pick)])
            row |= {name: len(idx), f"{name}_att": len(kept),
                    **{f"{name}_att{x}": int((bos[idx] <= x).sum()) for x in BOS_SENSITIVITY},
                    f"{name}_one_wording": int((per_wording.sum(0) == 1).sum()),
                    f"{name}_bos_median": round(float(np.median(bos[idx])), 3) if len(idx) else None}
        out[t] = {"sets": sets}
        word_lists[t] = {**lists, "raw_shares": {w: round(s[w], 4) for w in lists["raw"]},
                         "clean_shares": {w: round(s[w], 4) for w in lists["clean"]}}
        sizes.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "heads.json").write_text(json.dumps({"simple": out}, indent=1))
    (args.out / "word_lists.json").write_text(json.dumps(dict(removed=removed, tasks=word_lists), indent=1))
    with (args.out / "set_sizes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sizes[0]))
        w.writeheader()
        w.writerows(sizes)
    for r in sizes:
        print(r)


if __name__ == "__main__":
    main()
