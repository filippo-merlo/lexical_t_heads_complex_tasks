"""E9: E8's DB set on the complex tasks. J-head only, rule style, easy level, no fold split.

Per task and head, on the identification prompts of the seven rule wordings pooled (``orig``,
``w1``..``w6``; the first 100 in P+ of each, 700 prompts, ``data/readings/complex``), at the last
prompt token, E8's two filters with E8's thresholds and word rules (``scripts/e8_sets.py``):

- ``D`` decoding: share of prompts whose J-head top 10 holds a token of the task's word list;
  accept if >= 0.10. The list is the task's 30 most active decoded words (top-10 tokens decoded,
  lowercased, alphabetic, at least three letters, not in ``STOP``, Zipf >= 3; activity = share of
  (prompt, head) pairs whose top 10 holds the word), nothing removed, tokens from
  ``lth.terms.expand_lowercase_to_token_ids``.
- ``B`` BOS: mean attention of the last prompt token on BOS; accept if <= 0.5
  (``data/readings/attention_complex``, ``scripts/e9_attention.py``).

``DB`` = heads that pass both; all layers, no size cap. Only DB is patched; D's and B's sizes are
kept for the report.

Writes ``--out``: ``heads.json`` (``{"complex": {task: {"sets": [["rule", "DB", heads]]}}}``,
read by ``scripts/patch.py --set-list``), ``scores.npz`` (per task: ``dec``, ``bos``),
``word_lists.json`` and ``set_sizes.csv``.

    .venv/bin/python scripts/e9_sets.py --out results/e9_complex_db
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e8_sets import N_HEADS, N_WORDS, STOP, WORD  # noqa: E402
from lth import DATA, READOUTS, fu  # noqa: E402

R = READOUTS.index("jhead")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from wordfreq import zipf_frequency

    from identify_heads import complex_styles
    from lth.evaluation import tokenizer as get_tok
    from lth.store import load, readings_dir
    from lth.terms import expand_lowercase_to_token_ids

    tok = get_tok()
    cache: dict[int, str | None] = {}

    def word(t: int):  # E7's word rules
        if t not in cache:
            w = tok.decode([t]).strip().lower()
            cache[t] = w if WORD.match(w) and w not in STOP and zipf_frequency(w, "en") >= 3 else None
        return cache[t]

    out, scores, sizes, word_lists = {}, {}, [], {}
    for key in fu.STUDY_LEVELS:
        directory = readings_dir("complex", key)
        wordings = complex_styles()["rule"]
        runs = [load(directory, w) for w in wordings]
        pooled = np.concatenate([run["top"][:, R] for run in runs])  # prompts, heads, 10
        # The attention rows must be the same prompts in the same order.
        name = key.replace("/", "__")
        order = json.loads((DATA / "readings" / "attention_complex" / f"{name}.json").read_text())
        assert [(o["wording"], o["item"]) for o in order] == \
            [(w, i) for w, run in zip(wordings, runs) for i in run["id_items"].tolist()]
        bos = np.load(DATA / "readings" / "attention_complex" / f"{name}.npy").astype(np.float32).mean(0)
        c = Counter()
        for p in range(pooled.shape[0]):
            for h in range(pooled.shape[1]):
                c.update({w for x in pooled[p, h] if (w := word(int(x)))})
        words = sorted(c, key=lambda w: (-c[w], w))[:N_WORDS]
        word_lists[key] = {w: round(c[w] / (pooled.shape[0] * pooled.shape[1]), 4) for w in words}
        ids = np.array(expand_lowercase_to_token_ids(tok, words))
        dec = np.isin(pooled, ids).any(-1).mean(0)
        d, b = dec >= 0.10, bos <= 0.5
        db = np.flatnonzero(d & b)
        out[key] = {"sets": [["rule", "DB", sorted([int(i) // N_HEADS, int(i) % N_HEADS] for i in db)]]}
        scores |= {f"{key}/dec": dec, f"{key}/bos": bos}
        sizes.append({"task": key, "n_prompts": pooled.shape[0], "D": int(d.sum()), "B": int(b.sum()),
                      "DB": len(db)})
        print(sizes[-1], flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "heads.json").write_text(json.dumps({"complex": out}, indent=1))
    (args.out / "word_lists.json").write_text(json.dumps({"n_words": N_WORDS, "tasks": word_lists}, indent=1))
    np.savez_compressed(args.out / "scores.npz", **scores)
    with (args.out / "set_sizes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sizes[0]))
        w.writeheader()
        w.writerows(sizes)


if __name__ == "__main__":
    main()
