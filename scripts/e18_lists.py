"""E18 step 1 (no GPU): E13's task-word lists at other lengths (20 and 40 words) and their D scores.

E13's rule (``e13_lists.py``) unchanged except the length: on the fold's choosing half, the task's
words by share of (prompt, head) pairs whose J-head top 10 holds them, the 50 commonest WikiText
words removed, then the first ``n`` words. No Yang terms, no generated words. The 30-word list is
E13's own (not rebuilt); checks: the 40-word list starts with E13's 30 words, the 20-word list is
E13's first 20.

Writes ``results/e18_list_length/w{n}/fold{f}/word_lists.json`` and ``d_scores.npz``.

    .venv/bin/python scripts/e18_lists.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS, simple  # noqa: E402

OUT = RESULTS / "e18_list_length"
LENGTHS = (20, 40)


def main() -> None:
    import e13_lists as L

    from lth.evaluation import tokenizer

    tmap, words = L.token_words(tokenizer())
    wiki = L.shares(np.load(DATA / "readings" / "neutral" / "wikitext.npz")["top"][:, L.JH], tmap, len(words))
    generic = {words[i] for i in np.argsort(-wiki, kind="stable")[:L.N_GENERIC]}
    for fold, half in L.HALF.items():
        e13 = json.loads((RESULTS / "e13_capped" / f"fold{fold}" / "word_lists.json").read_text())["tasks"]
        lists = {n: {} for n in LENGTHS}
        d = {n: {} for n in LENGTHS}
        for t in simple.TASKS:
            top = L.pooled_top(half, t)
            s = L.shares(top, tmap, len(words))
            ranked = [i for i in np.argsort(-s, kind="stable") if s[i] > 0 and words[i] not in generic]
            assert [words[i] for i in ranked[:L.N_WORDS]] == e13[t]["words"], f"fold {fold} {t}: E13 list not reproduced"
            for n in LENGTHS:
                order = ranked[:n]
                lists[n][t] = {"words": [words[i] for i in order], "shares": [round(float(s[i]), 6) for i in order],
                               "n_prompts": int(top.shape[0])}
                member = np.zeros(len(words) + 1, dtype=bool)  # index -1 -> last slot, False
                member[order] = True
                d[n][t] = member[tmap[top.astype(np.int64)]].any(-1).mean(0)
        for n in LENGTHS:
            out = OUT / f"w{n}" / f"fold{fold}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "word_lists.json").write_text(json.dumps(dict(
                fold=fold, choosing_half=half, n_words=n,
                rule=f"E13's rule (isalnum, not STOP, >=3 characters if Latin; the {L.N_GENERIC} commonest WikiText "
                     f"words removed), first {n} words", tasks=lists[n]), ensure_ascii=False, indent=1))
            np.savez_compressed(out / "d_scores.npz", **d[n])
        print(f"fold {fold}: E13 lists reproduced; 20- and 40-word lists and D scores written")


if __name__ == "__main__":
    main()
