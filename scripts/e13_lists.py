"""E13 step 1 (no GPU): the task-word lists and the D (decoding) score of every head, per fold.

All on the fold's choosing half (fold 1: A, fold 2: B): the J-head top 10 at the last prompt token of
the correct prompts of the five rule templates (100 each, 500 per task; ``data/readings/heldout/<half>``).

- **Word rule.** A token, decoded, trimmed and lower-cased, is a word if every character is a letter
  (any script) or a digit (``str.isalnum``), it is not one of the 18 words of ``STOP``, and it has at
  least three characters if it contains a Latin letter (other scripts and pure numbers: no minimum).
- **Share.** A word's share on a task is the fraction of (prompt, head) pairs whose top 10 holds a
  token that is that word (each word once per pair).
- **Generic words.** The ``N_GENERIC`` = 50 words with the largest share on the 1,000 WikiText-103
  passages of ``data/readings/neutral/wikitext.npz`` (cut to the rule prompts' lengths), same rule.
- **Task list.** The task's words by share, the generic words removed, the first ``N_WORDS`` = 30.
  One list per task and fold, used by both D and T.
- **D score.** A head's share of the 500 prompts whose top 10 holds a token that is a word of the list.

Writes ``results/e13_capped/fold{f}/word_lists.json`` (generic words and per-task lists with shares)
and ``d_scores.npz`` (task -> 1024 shares).

    .venv/bin/python scripts/e13_lists.py
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, READOUTS, RESULTS, simple  # noqa: E402

OUT = RESULTS / "e13_capped"
HALF = {1: "A", 2: "B"}
STOP = set("the and for with that this from are was were you your has have had not but its".split())
N_GENERIC, N_WORDS = 50, 30
JH = READOUTS.index("jhead")


def is_word(s: str) -> bool:
    if not s or not s.isalnum() or s in STOP:
        return False
    latin = any(ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN") for ch in s)
    return len(s) >= 3 or not latin


def token_words(tok) -> tuple[np.ndarray, list[str]]:
    """token id -> word index (-1: not a word), and the words."""
    index: dict[str, int] = {}
    tmap = np.full(len(tok), -1, dtype=np.int64)
    for i in range(len(tok)):
        s = tok.decode([i]).strip().lower()
        if is_word(s):
            tmap[i] = index.setdefault(s, len(index))
    return tmap, sorted(index, key=index.get)


def shares(top: np.ndarray, tmap: np.ndarray, n_words: int) -> np.ndarray:
    """top [P, H, 10] token ids -> share of (prompt, head) pairs holding each word."""
    w = np.sort(tmap[top.astype(np.int64)], axis=-1)
    first = np.ones_like(w, dtype=bool)
    first[..., 1:] = w[..., 1:] != w[..., :-1]
    return np.bincount(w[first & (w >= 0)], minlength=n_words) / (top.shape[0] * top.shape[1])


def pooled_top(half: str, task: str) -> np.ndarray:
    from lth.store import load

    root = DATA / "readings" / "heldout" / half / "simple" / task.replace("/", "__")
    return np.concatenate([load(root, w)["top"][:, JH] for w in simple.WORDINGS["rule"]])


def main() -> None:
    from lth.evaluation import tokenizer

    tok = tokenizer()
    tmap, words = token_words(tok)
    wiki = shares(np.load(DATA / "readings" / "neutral" / "wikitext.npz")["top"][:, JH], tmap, len(words))
    generic_idx = np.argsort(-wiki, kind="stable")[:N_GENERIC]
    generic = {words[i] for i in generic_idx}
    for fold, half in HALF.items():
        lists, d = {}, {}
        for task in simple.TASKS:
            top = pooled_top(half, task)
            s = shares(top, tmap, len(words))
            order = [i for i in np.argsort(-s, kind="stable") if s[i] > 0 and words[i] not in generic][:N_WORDS]
            lists[task] = {"words": [words[i] for i in order], "shares": [round(float(s[i]), 6) for i in order],
                           "n_prompts": int(top.shape[0])}
            member = np.zeros(len(words) + 1, dtype=bool)  # index -1 -> last slot, False
            member[order] = True
            d[task] = member[tmap[top.astype(np.int64)]].any(-1).mean(0)
        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "word_lists.json").write_text(json.dumps(dict(
            fold=fold, choosing_half=half, rule="isalnum, not STOP, >=3 characters if it contains a Latin letter",
            stop=sorted(STOP), generic=[[words[i], round(float(wiki[i]), 6)] for i in generic_idx],
            tasks=lists), ensure_ascii=False, indent=1))
        np.savez_compressed(out / "d_scores.npz", **d)
        print(f"fold {fold}: lists and D scores written; e.g. antonym: {', '.join(lists['antonym']['words'][:8])}")


if __name__ == "__main__":
    main()
