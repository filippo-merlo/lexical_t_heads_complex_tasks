"""E14 step 3 (no GPU): the task-word lists and the D (decoding) score of every head, per fold, on all
35 rule templates (Yang et al.'s five and the 30 new ones of ``lth.simple.E4B_WORDINGS``).

E13's rule unchanged (``e13_lists.py``: word rule, the 50 WikiText words removed, 30 words per task,
one list for D and T), on more prompts: the J-head top 10 at the last token of the correct prompts of
every template on the fold's choosing half (at most 100 each, ``data/readings/heldout/<half>``), pooled.
A template with fewer correct prompts gives fewer; the prompt count of each template is recorded.

Writes ``results/e14_wordings/fold{f}/word_lists.json`` and ``d_scores.npz`` (task -> 1024 shares).

    .venv/bin/python scripts/e14_lists.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e13_lists import JH, N_GENERIC, N_WORDS, STOP, shares, token_words  # noqa: E402
from lth import DATA, RESULTS, simple  # noqa: E402

OUT = RESULTS / "e14_wordings"
HALF = {1: "A", 2: "B"}


def pooled_top(half: str, task: str) -> tuple[np.ndarray, dict[str, int]]:
    """[prompts, 1024, 10] J-head top tokens of the 35 templates' half readings, and prompts per template."""
    from lth.store import load

    root = DATA / "readings" / "heldout" / half / "simple" / task.replace("/", "__")
    tops = {w: load(root, w)["top"] for w in simple.ALL_RULE}
    tops = {w: t[:, JH] for w, t in tops.items() if len(t)}
    counts = {w: (len(tops[w]) if w in tops else 0) for w in simple.ALL_RULE}
    return np.concatenate(list(tops.values())), counts


def main() -> None:
    from lth.evaluation import tokenizer

    tok = tokenizer()
    tmap, words = token_words(tok)
    wiki = shares(np.load(DATA / "readings" / "neutral" / "wikitext.npz")["top"][:, JH], tmap, len(words))
    generic_idx = np.argsort(-wiki, kind="stable")[:N_GENERIC]
    generic = {words[i] for i in generic_idx}
    e13 = {f: json.loads((RESULTS / "e13_capped" / f"fold{f}" / "word_lists.json").read_text())["tasks"] for f in HALF}
    for fold, half in HALF.items():
        lists, d = {}, {}
        for task in simple.TASKS:
            top, counts = pooled_top(half, task)
            s = shares(top, tmap, len(words))
            order = [i for i in np.argsort(-s, kind="stable") if s[i] > 0 and words[i] not in generic][:N_WORDS]
            old = set(e13[fold][task]["words"])
            lists[task] = {"words": [words[i] for i in order], "shares": [round(float(s[i]), 6) for i in order],
                           "n_prompts": int(top.shape[0]), "prompts_per_template": counts,
                           "shared_with_e13": len(old & {words[i] for i in order})}
            member = np.zeros(len(words) + 1, dtype=bool)  # index -1 -> last slot, False
            member[order] = True
            d[task] = member[tmap[top.astype(np.int64)]].any(-1).mean(0)
        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "word_lists.json").write_text(json.dumps(dict(
            fold=fold, choosing_half=half, templates=list(simple.ALL_RULE),
            rule="isalnum, not STOP, >=3 characters if it contains a Latin letter (E13)",
            stop=sorted(STOP), generic=[[words[i], round(float(wiki[i]), 6)] for i in generic_idx],
            tasks=lists), ensure_ascii=False, indent=1))
        np.savez_compressed(out / "d_scores.npz", **d)
        for task in simple.TASKS:
            x = lists[task]
            print(f"fold {fold} {task:17s} prompts {x['n_prompts']:5d}  same words as E13 {x['shared_with_e13']:2d}/30  "
                  f"{', '.join(x['words'][:8])}")


if __name__ == "__main__":
    main()
