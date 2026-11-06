"""E4 (paper/e4_heldout.tex): the decoded-word and E1 head sets of one fold, J-head only.

Everything is chosen on the choosing half (``--choose A`` for fold 1, ``B`` for fold 2): its
identification readings (``scripts/heldout_readings.py``), its accuracies, the word shares of
every task. With ``--choose all`` the same code runs on the readings and items of the whole
study, and must give the Sec. 15 sets (``--check``). The rules are those of Sec. 15, frozen
(``scripts/e3/word_list_union.py``, ``neutral_sets.py``, ``cc_sizes.py``):

- words: a head's top-10 at the last prompt token, decoded, lowercased, alphabetic, at least
  three letters, not in ``STOP``, Zipf frequency at least 3; a word's share in a task is the
  share of (prompt, head) pairs whose top 10 holds it, over the five rule wordings' P+ pooled;
- task words: share in the task minus the mean share in the other 13 tasks, top 5; common
  words: highest smallest share over the 14 tasks, top 5; the CC list is both (union);
- ``e1``: Definition 1 (k = 10, rho = 0.10) with Yang et al.'s term list on the P+ of the
  rule template most accurate on the choosing half;
- ``cc``: Definition 1 with the CC list on the five templates' P+ pooled (``N_t`` heads);
- ``cc20``, ``cc40``: the first 20 and 40 heads of the model by pooled match rate under the
  CC list (ties: mean number of list tokens in the top 10, then layer and head);
- ``ccclean``: the same after removing the 50 words with the largest share on the 1,000
  WikiText-103 passages (``data/readings/neutral/wikitext.npz``), 5 + 5 words;
- ``ad20``, ``ad40`` (with ``--ad-scores``): the first 20 and 40 heads by the attend x decode
  score of Sec. 15 on this half (ties: attention, layer, head);
- ``random{N}_{i}``, N in 20, 40, N_t (named ``randomNt_{i}``), i = 0..4: the same number of
  heads in every layer as ``cc{N}`` (``cc`` for N_t), drawn outside it, seed i.

Writes ``--out``: ``heads.json`` (set-list layout of ``scripts/rank_heads.py``: per task
``sets`` = ``[style, name, heads]``), ``word_lists.json`` (per task: specific top 10, common
top 10, CC list, CC-clean list, the E1 wording) and ``set_sizes.csv``.

    .venv/bin/python scripts/heldout_sets.py --choose A --out results/heldout/fold1
    .venv/bin/python scripts/heldout_sets.py --choose all --out <scratch> --check
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

from lth import DATA, READOUTS, RESULTS, simple  # noqa: E402

N_WORDS, RHO, N_HEADS, K_NEUTRAL, N_RANDOM = 5, 0.10, 32, 50, 5
SIZES = (20, 40)
R = READOUTS.index("jhead")
STOP = set("the and for with that this from are was were you your has have had not but its".split())
WORD = re.compile(r"^[a-z]{3,}$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--choose", choices=("A", "B", "all"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="with --choose all: compare with the Sec. 15 sets")
    parser.add_argument("--ad-scores", type=Path, default=None,
                        help="attend x decode scores of this half (scripts/e3/attend_decode_scores.py): adds ad20, ad40")
    args = parser.parse_args()

    from wordfreq import zipf_frequency

    from lth.evaluation import tokenizer as get_tok
    from lth.store import load
    from lth.terms import expand_to_token_ids, simple_term_lists

    tok = get_tok()
    root = DATA / "readings" / ("heldout/" + args.choose if args.choose != "all" else "")
    halves = json.loads((RESULTS / "heldout" / "split.json").read_text())
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

    def lists(s):
        vocab = set.union(*(set(v) for v in s.values()))
        common = sorted((w for w in vocab if all(w in s[t] for t in simple.TASKS)),
                        key=lambda w: (-min(s[t][w] for t in simple.TASKS), w))
        spec = {}
        for t in simple.TASKS:
            o = [x for x in simple.TASKS if x != t]
            spec[t] = sorted(s[t], key=lambda w: (-(s[t][w] - np.mean([s[x].get(w, 0) for x in o])), w))
        return common, spec

    def rate_of(top, words):
        ids = np.array(sorted(set(expand_to_token_ids(tok, words))))
        hit = np.isin(top, ids)
        return hit.any(-1).mean(0), hit.sum(-1).mean(0)

    def heads_of(idx):
        return sorted([int(h) // N_HEADS, int(h) % N_HEADS] for h in idx)

    run = {t: {w: load(root / "simple" / t, w) for w in simple.WORDINGS["rule"]} for t in simple.TASKS}
    pooled = {t: np.concatenate([run[t][w]["top"][:, R] for w in simple.WORDINGS["rule"]]) for t in simple.TASKS}
    task_shares = {t: shares(pooled[t]) for t in simple.TASKS}
    common, spec = lists(task_shares)
    neutral = shares(np.load(DATA / "readings" / "neutral" / "wikitext.npz")["top"][:, R])
    removed = set(sorted(neutral, key=lambda w: (-neutral[w], w))[:K_NEUTRAL])
    common_c, spec_c = lists({t: {w: v for w, v in s.items() if w not in removed} for t, s in task_shares.items()})

    import pandas as pd

    ad = pd.read_csv(args.ad_scores) if args.ad_scores else None
    out, word_lists, sizes = {}, {}, []
    for t in simple.TASKS:
        members = None if args.choose == "all" else set(halves[t][args.choose])
        accuracy = {}
        for w in simple.WORDINGS["rule"]:
            rec = [r["correct"] for r in run[t][w]["records"] if members is None or int(r["item"]) in members]
            accuracy[w] = float(np.mean(rec))
        e1_wording = simple.identification_wording("rule", accuracy)
        sets = []
        e1_rate, _ = rate_of(run[t][e1_wording]["top"][:, R], simple_term_lists()[t])
        sets.append(["rule", "e1", heads_of(np.flatnonzero(e1_rate >= RHO))])
        cc_words = spec[t][:N_WORDS] + [w for w in common[:N_WORDS] if w not in spec[t][:N_WORDS]]
        rate, count = rate_of(pooled[t], cc_words)
        cc = heads_of(np.flatnonzero(rate >= RHO))
        sets.append(["rule", "cc", cc])
        order = np.lexsort((np.arange(len(rate)), -count, -rate))
        ranked = {n: heads_of(order[:n]) for n in SIZES}
        for n in SIZES:
            sets.append(["rule", f"cc{n}", ranked[n]])
        clean_words = spec_c[t][:N_WORDS] + [w for w in common_c[:N_WORDS] if w not in spec_c[t][:N_WORDS]]
        clean_rate, _ = rate_of(pooled[t], clean_words)
        sets.append(["rule", "ccclean", heads_of(np.flatnonzero(clean_rate >= RHO))])
        if ad is not None:
            g = ad[ad.task == t].sort_values(["score", "attn", "layer", "head"], ascending=[False, False, True, True])
            for n in SIZES:
                sets.append(["rule", f"ad{n}", sorted([int(l), int(h)] for l, h in zip(g.layer[:n], g["head"][:n]))])
        for label, base in [*((str(n), ranked[n]) for n in SIZES), ("Nt", cc)]:
            inside = {tuple(h) for h in base}
            per_layer = Counter(l for l, _ in inside)
            for i in range(N_RANDOM):
                rng = np.random.default_rng(i)
                pick = []
                for layer in sorted(per_layer):
                    pool = [(layer, x) for x in range(N_HEADS) if (layer, x) not in inside]
                    pick += [list(pool[int(j)]) for j in rng.choice(len(pool), per_layer[layer], replace=False)]
                sets.append(["rule", f"random{label}_{i}", sorted(pick)])
        out[t] = {"sets": sets}
        word_lists[t] = dict(specific=spec[t][:10], common=common[:10], cc=cc_words, ccclean=clean_words,
                             e1_wording=e1_wording, n_prompts=int(len(pooled[t])),
                             n_prompts_e1=int(len(run[t][e1_wording]["top"])))
        sizes.append(dict(task=t, e1=len(sets[0][2]), cc=len(cc), ccclean=len(sets[4][2]),
                          e1_wording=e1_wording, n_prompts=len(pooled[t]), cc_words=" ".join(cc_words)))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "heads.json").write_text(json.dumps({"simple": out}, indent=1))
    (args.out / "word_lists.json").write_text(json.dumps(dict(removed=sorted(removed), tasks=word_lists), indent=1))
    with (args.out / "set_sizes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sizes[0]))
        w.writeheader()
        w.writerows(sizes)
    for s in sizes:
        print(f"{s['task']:17s} E1 {s['e1']:3d} ({s['e1_wording']}) CC {s['cc']:3d} clean {s['ccclean']:3d} "
              f"prompts {s['n_prompts']:3d} | {s['cc_words']}")
    if args.check:
        check(out)


def check(out) -> None:
    """--choose all must give the Sec. 15 sets: E1 (results/heads.json), CC-pooled
    (word_lists_union), CC-pooled-clean (neutral_union) and the CC top 20 and 40 (cc_sizes)."""
    def named(path, name):
        return {t: {tuple(h) for h in next(hs for _, n, hs in v["sets"] if n == name)}
                for t, v in json.loads(Path(path).read_text())["simple"].items()}

    e1 = {t: {(l, h) for l, h, _ in v["rule"]["jhead"]["heads"]}
          for t, v in json.loads((RESULTS / "heads.json").read_text())["simple"].items()}
    refs = {"e1": e1, "cc": named(RESULTS / "word_lists_union/heads.json", "union_alllist_allid"),
            "ccclean": named(RESULTS / "neutral_union/heads.json", "neutral_5+5"),
            "cc20": named(RESULTS / "cc_sizes/heads.json", "cc_top20"),
            "cc40": named(RESULTS / "cc_sizes/heads.json", "cc_top40")}
    if any(n == "ad20" for _, n, _ in out[simple.TASKS[0]]["sets"]):
        refs["ad20"] = named(RESULTS / "attend_decode/heads.json", "ad_top20")
    bad = 0
    for name, ref in refs.items():
        for t in simple.TASKS:
            mine = {tuple(h) for _, n, hs in out[t]["sets"] if n == name for h in hs}
            if mine != ref[t]:
                bad += 1
                print(f"MISMATCH {name} {t}: mine {len(mine)}, Sec. 15 {len(ref[t])}, both {len(mine & ref[t])}")
    print("check:", "all sets equal to Sec. 15" if not bad else f"{bad} mismatches")


if __name__ == "__main__":
    main()
