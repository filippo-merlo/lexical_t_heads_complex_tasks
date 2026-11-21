"""Rebuilt design, word-list variant (user 2026-11-21): the task's 30 words by a smooth TF-IDF of the
task against WikiText, in place of the share ranking with the 50 WikiText generic words removed
(paper/head-sets-simple-design.tex, Sec. 5.2). Builds the shared 40-head sets of the five best
stage-2 shared combinations of each fold, for both folds. No GPU.

- **Word rule and shares** as in ``scripts/e13_lists.py``: a word's share on the task is the fraction of
  (prompt, head) pairs of the identification pool (``rebuild_sets.pool``, choosing half only) whose
  J-head top 10 holds it; its share on WikiText the same on the 1,000 WikiText-103 passages.
- **Smooth TF-IDF.** score(w) = s_task(w) * log(2 s_task(w) / (s_task(w) + s_wiki(w))): the task and
  WikiText are the two documents; a word decoded as often on WikiText as on the task scores about 0, one
  never decoded on WikiText scores s_task(w) * log 2. The task's list is its 30 words with share > 0 and
  the largest score (ties to the lower word index, as before). No generic-word cut: WikiText is the
  background.
- **Features.** D and the attention segments are recomputed with the new list exactly as
  ``rebuild_sets.build`` computes them (the task-word segment, and through its priority the other
  segments, depend on the list); B too, for completeness.
- **Sets.** Combinations: the union over both folds of the five best stage-2 shared combinations of
  that fold (``results/rebuild/additions/selected.json``, ``top10_other_fold[:5]``, ranked on the other
  fold), built in both folds so that alpha can be cross-fitted. Shared ranks as in the screen: each
  feature's within-task ranks averaged over the 14 tasks and ranked again, a combination's rank the
  mean of its features'; the top 40 (``shared_tfidf_<combo>``).
- **Check** (always run first): with the stage-2 lists, the same code reproduces every stored feature
  of ``results/rebuild/additions/fold<f>/features.npz`` and the stored ``shared_screen_<combo>`` sets.

Writes ``results/rebuild/tfidf/fold<f>/``: ``heads.json`` (``scripts/patch.py --set-list`` format),
``word_lists.json`` (words, scores, shares on task and WikiText), ``features.npz``, ``sets.csv``
(combination, overlap with the stage-2 set of the same combination, median layer).

``--screen`` (user 2026-11-21): also the whole stage-2 screen with these lists, for the two-stage screen
(``scripts/rebuild_quick_screen.sh tfidf_screen all``): every one of the 511 combinations at 40 heads, per task
(``screen_<combo>``, the task's own ranks) and shared (``shared_screen_<combo>``), built as in
``rebuild_sets.build``; checked first by rebuilding all of stage 2's screen sets from its lists. Writes
``results/rebuild/tfidf_screen/fold<f>/``: ``heads.json``, ``features.npz``, ``word_lists.json``.

    .venv/bin/python scripts/rebuild_tfidf_sets.py
    .venv/bin/python scripts/rebuild_tfidf_sets.py --screen
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

from lth import DATA, RESULTS, simple  # noqa: E402
from rebuild_sets import (COMBOS, FEATURES, FOLDS, JH, N_HEADS, SCREEN_N, order_of, pool, rankdata,  # noqa: E402
                          ranks, segment_masses, top)

ADD = RESULTS / "rebuild" / "additions"
OUT = RESULTS / "rebuild" / "tfidf"
SCREEN_OUT = RESULTS / "rebuild" / "tfidf_screen"
N_TOP_COMBOS = 5


def smooth_tfidf(s_task: np.ndarray, s_wiki: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        score = s_task * np.log(2 * s_task / (s_task + s_wiki))
    return np.where(s_task > 0, score, -np.inf)


def task_features(half: str, task: str, tok, tmap, words: list[str], list_idx: list[int]):
    """D, B and the segment masses of every head for one task and word list (as ``rebuild_sets.build``)."""
    keys, top_t, _ = pool(half, task)
    member = np.zeros(len(words) + 1, dtype=bool)
    member[list_idx] = True
    f = {"D": member[tmap[top_t[:, JH].astype(np.int64)]].any(-1).mean(0)}
    masses, counts = segment_masses(half, task, keys, [words[i] for i in list_idx], tok)
    f["B"] = 1.0 - masses["bos"]
    for code, (_, seg) in FEATURES.items():
        if seg is not None:
            f[code] = masses[seg]
    return f, counts, top_t


def shared_sets(feats: dict, combos: list[str]) -> dict[str, list[list[int]]]:
    single = {t: {c: ranks(feats[t][c]) for c in FEATURES} for t in simple.TASKS}
    shared_single = {c: rankdata(np.mean([single[t][c] for t in simple.TASKS], axis=0), method="average")
                     for c in FEATURES}
    return {c: top(order_of(np.mean([shared_single[x] for x in c], axis=0)), SCREEN_N) for c in combos}


def screen_sets(feats: dict) -> dict[str, list]:
    """Every combination's 40-head set, per task and shared, as ``rebuild_sets.build`` lists them."""
    single = {t: {c: ranks(feats[t][c]) for c in FEATURES} for t in simple.TASKS}
    shared = shared_sets(feats, COMBOS)
    return {t: [["rule", f"screen_{combo}", top(order_of(np.mean([single[t][c] for c in combo], axis=0)), SCREEN_N)]
                for combo in COMBOS] + [["rule", f"shared_screen_{combo}", shared[combo]] for combo in COMBOS]
            for t in simple.TASKS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--screen", action="store_true", help="also the whole 511-combination screen (see above)")
    args = parser.parse_args()
    from e13_lists import shares, token_words

    from lth.evaluation import tokenizer

    tok = tokenizer()
    tmap, words = token_words(tok)
    index = {w: i for i, w in enumerate(words)}
    wiki = shares(np.load(DATA / "readings" / "neutral" / "wikitext.npz")["top"][:, JH], tmap, len(words))
    stage2 = json.loads((ADD / "selected.json").read_text())
    combos = []
    for fold in FOLDS:
        for c, _ in stage2[str(fold)]["shared"]["top10_other_fold"][:N_TOP_COMBOS]:
            if c not in combos:
                combos.append(c)
    print(f"combinations ({len(combos)}): {combos}", flush=True)

    for fold, half in FOLDS.items():
        # Check: the stage-2 lists through this code give the stored features and screen sets.
        old_lists = json.loads((ADD / f"fold{fold}" / "word_lists.json").read_text())["tasks"]
        stored = np.load(ADD / f"fold{fold}" / "features.npz")
        old_heads = json.loads((ADD / f"fold{fold}" / "heads.json").read_text())["simple"][simple.TASKS[0]]["sets"]
        old_sets = {name[len("shared_screen_"):]: hs for _, name, hs in old_heads if name.startswith("shared_screen_")}
        old_feats = {}
        for t in simple.TASKS:
            old_feats[t], _, _ = task_features(half, t, tok, tmap, words, [index[w] for w in old_lists[t]["words"]])
            for c in FEATURES:
                assert np.array_equal(old_feats[t][c], stored[f"{t}/{c}"]), (fold, t, c)
        rebuilt = shared_sets(old_feats, combos)
        for c in combos:
            assert rebuilt[c] == old_sets[c], (fold, c)
        print(f"fold {fold}: check passed (features of 14 tasks, {len(combos)} shared screen sets)", flush=True)
        if args.screen:
            all_old = json.loads((ADD / f"fold{fold}" / "heads.json").read_text())["simple"]
            for t, entries in screen_sets(old_feats).items():
                stored_sets = {name: hs for _, name, hs in all_old[t]["sets"] if "screen_" in name}
                assert {name: hs for _, name, hs in entries} == stored_sets, (fold, t)
            print(f"fold {fold}: check passed (all {2 * len(COMBOS)} screen sets of each task)", flush=True)

        # New lists and features.
        feats, lists, seg_counts = {}, {}, {}
        for t in simple.TASKS:
            _, top_t, used = pool(half, t)
            s = shares(top_t[:, JH], tmap, len(words))
            score = smooth_tfidf(s, wiki)
            idx = [int(i) for i in np.argsort(-score, kind="stable") if np.isfinite(score[i])][:30]
            feats[t], seg_counts[t], _ = task_features(half, t, tok, tmap, words, idx)
            lists[t] = {"words": [words[i] for i in idx], "scores": [round(float(score[i]), 6) for i in idx],
                        "shares": [round(float(s[i]), 6) for i in idx], "wiki_shares": [round(float(wiki[i]), 6) for i in idx],
                        "n_prompts": int(top_t.shape[0]), "prompts_per_template": used,
                        "stage2_words_kept": len(set(old_lists[t]["words"]) & {words[i] for i in idx})}
        new = shared_sets(feats, combos)
        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        entries = [["rule", f"shared_tfidf_{c}", new[c]] for c in combos]
        (out / "heads.json").write_text(json.dumps({"simple": {t: {"sets": entries} for t in simple.TASKS}}))
        (out / "word_lists.json").write_text(json.dumps(dict(
            fold=fold, choosing_half=half, rule="smooth TF-IDF against WikiText, 30 words, no generic-word cut",
            tasks=lists, segment_prompts=seg_counts), ensure_ascii=False, indent=1))
        np.savez_compressed(out / "features.npz", **{f"{t}/{c}": feats[t][c] for t in simple.TASKS for c in FEATURES})
        rows = []
        for c in combos:
            hs, old = {tuple(h) for h in new[c]}, {tuple(h) for h in old_sets[c]}
            rows.append(dict(fold=fold, combo=c, top5_of_fold=",".join(str(f) for f in FOLDS if c in
                                                                     [x for x, _ in stage2[str(f)]["shared"]["top10_other_fold"][:N_TOP_COMBOS]]),
                             n_heads=len(hs), overlap_stage2=len(hs & old),
                             layer_median=float(np.median([l for l, _ in hs])),
                             layer_median_stage2=float(np.median([l for l, _ in old]))))
        with (out / "sets.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        kept = [lists[t]["stage2_words_kept"] for t in simple.TASKS]
        print(f"fold {fold}: words kept from stage-2 lists {min(kept)}-{max(kept)} (mean {np.mean(kept):.1f}); "
              + "; ".join(f"{r['combo']} {r['overlap_stage2']}/40" for r in rows), flush=True)
        if args.screen:
            sets = screen_sets(feats)
            for c in combos:  # the sets patched above are the screen's shared sets
                assert {name: hs for _, name, hs in sets[simple.TASKS[0]]}[f"shared_screen_{c}"] == new[c], (fold, c)
            sdir = SCREEN_OUT / f"fold{fold}"
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / "heads.json").write_text(json.dumps({"simple": {t: {"sets": sets[t]} for t in simple.TASKS}}))
            np.savez_compressed(sdir / "features.npz", **{f"{t}/{c}": feats[t][c] for t in simple.TASKS for c in FEATURES})
            (sdir / "word_lists.json").write_text((out / "word_lists.json").read_text())
            distinct = {s: len({frozenset(map(tuple, hs)) for _, name, hs in sets[simple.TASKS[0]] if name.startswith(p)})
                        for s, p in (("shared", "shared_screen_"),)}
            print(f"fold {fold}: screen written to {sdir} ({len(COMBOS)} combinations per scope, "
                  f"{distinct['shared']} distinct shared sets)", flush=True)
    assert N_HEADS == 32


if __name__ == "__main__":
    main()
