"""Decoding features beyond the task-word list (user 2026-11-22), step A: four features of what a head writes,
read with J-head at the last token of the identification pool (``rebuild_sets.pool``, choosing half only), and
their 40-head sets. No GPU.

Letters (uppercase: what a head writes; lowercase: where the last token reads; internal keys in brackets):

- **T** task words [D]: the task's 30-word list by smooth TF-IDF against WikiText (``rebuild_sets.word_list``).
- **Y** Yang et al.'s terms: their predefined task-descriptive terms (``lth.terms``), as in the ladder rule
  Yang terms (J): share of pool prompts whose J-head top 10 holds a term token, ties by the mean number held.
- **A** answer words: the same TF-IDF rule restricted to the task's answers, i.e. the words that are the first
  token of the expected answer of some pool prompt; the 30 with the largest smooth TF-IDF against WikiText.
  A head scores by the share of pool prompts whose top 10 holds one of them (the task's kind of answer, not the
  prompt's own).
- **C** common words: one list for all tasks, the 30 words with the largest geometric mean of their share over
  the 14 tasks' pools (decoded on every task), minus the 50 most shared on WikiText; per task, the share of
  pool prompts whose top 10 holds one of them.
- n non-first [B], t task words in the prompt [T], o other words [O], s stopwords [S], p punctuation [P],
  m markers [M], q query [Q], l last token [L]: the stage-2 features (``results/rebuild/additions``).

Sets, 40 heads, per task and shared (ranks as in ``rebuild_sets``: per task the mean of the features'
within-task ranks; shared, each feature's within-task ranks averaged over tasks and ranked again, then the
mean over the combination): every non-empty combination of T, Y, A, C alone (``lex_<letters>``) and with
Selected's attention part (``lex_<letters>_ntspl`` per task, ``shared_lex_<letters>_ntm`` shared; Selected is
T.ntspl per task, T.ntm shared). Checks: T, n and the attention features are the stored stage-2 ones, and
lex_T / lex_T_ntspl / shared_lex_T_ntm equal the stored screen sets D, DBTSPL, DBTM.

Mixed sets (user 2026-11-22): k heads from the T + attention ranking and 40 - k from the C + attention ranking
(the attention part of Selected: ntspl per task, ntm shared), k in ``MIX`` (30, 20, 10): the top k of the first
ranking, then the C ranking in order, skipping heads already in, until 40 (``mix_T<k>C<40-k>``,
``shared_mix_T<k>C<40-k>``). The ends 40/0 and 0/40 are lex_T_ntspl / lex_C_ntspl (shared: _ntm).

Writes ``results/rebuild/lexical/fold<f>/``: ``heads.json``, ``features.npz`` (T Y A C), ``word_lists.json``
(A and C lists), ``sets.csv`` (median layer, overlap with the T set of the same scope).

    .venv/bin/python scripts/rebuild_lexical.py
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS, simple  # noqa: E402
from rebuild_sets import (FOLDS, JH, N_HEADS, SCREEN_N, order_of, pool, rankdata, ranks, smooth_tfidf,  # noqa: E402
                          top, word_list)

ADD = RESULTS / "rebuild" / "additions"
OUT = RESULTS / "rebuild" / "lexical"
N_WORDS, N_COMMON_CUT = 30, 50
DECODING = ("T", "Y", "A", "C")
INTERNAL = {"T": "D", "n": "B", "t": "T", "o": "O", "s": "S", "p": "P", "m": "M", "q": "Q", "l": "L"}
ATTENTION = {"per task": "ntspl", "shared": "ntm"}  # Selected's attention part (stage 2, TF-IDF lists)
COMBOS = ["".join(c) for k in range(1, 5) for c in itertools.combinations(DECODING, k)]
MIX = (30, 20, 10)


def holds(top_j: np.ndarray, tmap: np.ndarray, idx: list[int], n_words: int) -> np.ndarray:
    """Share of prompts whose top 10 holds a word of ``idx`` (per head)."""
    member = np.zeros(n_words + 1, dtype=bool)
    member[idx] = True
    return member[tmap[top_j.astype(np.int64)]].any(-1).mean(0)


def mix(first: np.ndarray, second: np.ndarray, k: int, n: int = SCREEN_N) -> list[list[int]]:
    """The top ``k`` heads of the ``first`` order, then the ``second`` order skipping heads already in, until ``n``."""
    chosen = [int(h) for h in first[:k]]
    for h in second:
        if len(chosen) == n:
            break
        if int(h) not in chosen:
            chosen.append(int(h))
    assert len(chosen) == n
    return sorted([h // N_HEADS, h % N_HEADS] for h in chosen)


def main() -> None:
    from e13_lists import shares, token_words

    from lth.evaluation import tokenizer
    from lth.terms import expand_to_token_ids, simple_term_lists

    tok = tokenizer()
    tmap, words = token_words(tok)
    W = len(words)
    wiki = shares(np.load(DATA / "readings" / "neutral" / "wikitext.npz")["top"][:, JH], tmap, W)
    wiki_top = set(np.argsort(-wiki, kind="stable")[:N_COMMON_CUT].tolist())
    first = lambda a: tok(a, add_special_tokens=False).input_ids[0]  # noqa: E731
    for fold, half in FOLDS.items():
        stored = np.load(ADD / f"fold{fold}" / "features.npz")
        lists = json.loads((ADD / f"fold{fold}" / "word_lists.json").read_text())["tasks"]
        index = {w: i for i, w in enumerate(words)}
        tops, keys, sh = {}, {}, {}
        for t in simple.TASKS:
            keys[t], top_t, _ = pool(half, t)
            tops[t] = top_t[:, JH]
            sh[t] = shares(tops[t], tmap, W)
        with np.errstate(divide="ignore"):  # a word some task never decodes gets 0
            geo = np.exp(np.log(np.stack([sh[t] for t in simple.TASKS])).mean(0))
        common = [int(i) for i in np.argsort(-geo, kind="stable") if geo[i] > 0 and int(i) not in wiki_top][:N_WORDS]
        feats, alists = {}, {}
        for t in simple.TASKS:
            f = {}
            tlist = [index[w] for w in lists[t]["words"]]
            assert tlist == word_list(sh[t], wiki, N_WORDS), (fold, t)
            f["T"] = holds(tops[t], tmap, tlist, W)
            assert np.array_equal(f["T"], stored[f"{t}/D"]), (fold, t)
            terms = np.array(sorted(set(expand_to_token_ids(tok, simple_term_lists()[t]))))
            hit = np.isin(tops[t], terms)
            f["Y"] = hit.any(-1).mean(0) + 1e-6 * hit.sum(-1).mean(0)  # share, ties by the mean number held
            answers = {w: simple.prompts(t, w)[1] for w in simple.WORDINGS["rule"]}
            cand = {int(tmap[first(answers[simple.WORDINGS["rule"][j]][i])]) for j, i in keys[t]} - {-1}
            score = smooth_tfidf(sh[t], wiki)
            alist = [i for i in np.argsort(-score, kind="stable") if int(i) in cand and np.isfinite(score[i])][:N_WORDS]
            alists[t] = [words[i] for i in alist]
            f["A"] = holds(tops[t], tmap, [int(i) for i in alist], W)
            f["C"] = holds(tops[t], tmap, common, W)
            for x in "ntospmql":
                f[x] = stored[f"{t}/{INTERNAL[x]}"]
            feats[t] = f
        single = {t: {c: ranks(feats[t][c]) for c in feats[t]} for t in simple.TASKS}
        shared_single = {c: rankdata(np.mean([single[t][c] for t in simple.TASKS], axis=0)) for c in feats[simple.TASKS[0]]}
        per_task, shared = {t: [] for t in simple.TASKS}, []
        for combo in COMBOS:
            for suffix in ("", "_" + ATTENTION["per task"]):
                letters = combo + suffix[1:]
                for t in simple.TASKS:
                    r = np.mean([single[t][c] for c in letters], axis=0)
                    per_task[t].append(["rule", f"lex_{combo}{suffix}", top(order_of(r), SCREEN_N)])
            for suffix in ("", "_" + ATTENTION["shared"]):
                letters = combo + suffix[1:]
                r = np.mean([shared_single[c] for c in letters], axis=0)
                shared.append(["rule", f"shared_lex_{combo}{suffix}", top(order_of(r), SCREEN_N)])
        for k in MIX:
            for t in simple.TASKS:
                orders = [order_of(np.mean([single[t][c] for c in x + ATTENTION["per task"]], axis=0)) for x in "TC"]
                per_task[t].append(["rule", f"mix_T{k}C{SCREEN_N - k}", mix(*orders, k)])
            orders = [order_of(np.mean([shared_single[c] for c in x + ATTENTION["shared"]], axis=0)) for x in "TC"]
            shared.append(["rule", f"shared_mix_T{k}C{SCREEN_N - k}", mix(*orders, k)])
        # The T sets must be the stored stage-2 screen sets.
        old = json.loads((ADD / f"fold{fold}" / "heads.json").read_text())["simple"]
        for t in simple.TASKS:
            o = {nm: hs for _, nm, hs in old[t]["sets"]}
            mine = {nm: hs for _, nm, hs in per_task[t] + shared}
            prev = (OUT / f"fold{fold}" / "heads.json")
            if prev.exists():  # the step-A sets, unchanged
                p = {nm: hs for _, nm, hs in json.loads(prev.read_text())["simple"][t]["sets"]}
                assert all(mine[nm] == hs for nm, hs in p.items()), (fold, t)
            assert mine["lex_T"] == o["screen_D"] and mine["lex_T_ntspl"] == o["screen_DBTSPL"], (fold, t)
            assert mine["shared_lex_T_ntm"] == o["shared_screen_DBTM"], (fold, t)
        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "heads.json").write_text(json.dumps({"simple": {t: {"sets": per_task[t] + shared} for t in simple.TASKS}}))
        np.savez_compressed(out / "features.npz", **{f"{t}/{c}": feats[t][c] for t in simple.TASKS for c in DECODING})
        (out / "word_lists.json").write_text(json.dumps(dict(
            fold=fold, choosing_half=half, common=[words[i] for i in common],
            answers={t: alists[t] for t in simple.TASKS}), ensure_ascii=False, indent=1))
        rows = []
        for t in simple.TASKS:
            base = {nm: {tuple(h) for h in hs} for _, nm, hs in per_task[t] + shared}
            for _, nm, hs in per_task[t] + (shared if t == simple.TASKS[0] else []):
                ref = base["shared_lex_T" if nm.startswith("shared") else "lex_T"]
                rows.append(dict(fold=fold, task="all" if nm.startswith("shared") else t, set=nm,
                                 layer_median=float(np.median([l for l, _ in hs])), overlap_T=len({tuple(h) for h in hs} & ref)))
        with (out / "sets.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"fold {fold}: checks passed; common words: {', '.join(words[i] for i in common)}", flush=True)
        print(f"fold {fold}: answer words, e.g. country-capital: {', '.join(alists['country-capital'][:10])}; "
              f"present-past: {', '.join(alists['present-past'][:10])}", flush=True)
    assert N_HEADS == 32


if __name__ == "__main__":
    main()
