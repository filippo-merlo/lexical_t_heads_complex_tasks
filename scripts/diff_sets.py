"""E6: head sets from the instruction-difference readings (``scripts/diff_readings.py``), J-head.

E4's CC rule (``scripts/heldout_sets.py``, frozen), applied to other readings: words decoded
from a head's top 10 (lowercased, alphabetic, at least three letters, not in ``STOP``, Zipf
frequency at least 3); a word's share in a task = share of (prompt, head) pairs whose top 10
holds it, over the templates' prompts pooled; task words = share in the task minus the mean
share in the other 13 tasks, top 5; common words = highest smallest share over the 14 tasks,
top 5; a head is in when a list token is among its top 10 on at least 10% of the prompts
(``N_t`` heads); the first 20 and 40 by that rate (ties: mean number of list tokens, then layer
and head); layer-matched random sets (5 draws per size, seeds 0-4).

Sets (names in the choosing half's ``heads.json``):
- ``dcc``, ``dcc20``, ``dcc40``: the rule on the difference readings (inst1-inst4);
- ``dccA``: ``dcc`` minus the heads whose difference holds the prompt's own answer (first token,
  leading space) on at least 10% of the prompts (as E5's removal);
- ``random20_i``, ``random40_i``, ``randomNt_i``: drawn as E4's, around ``dcc20``, ``dcc40``, ``dcc``;
- with ``--task-words-only`` (added after the first sets: the common words do not cancel in the
  difference, they are *what, https, http, mph* and one more): ``dccT``, ``dccT20``, ``dccT40``,
  ``dccTA``, the same with the 5 task words only, no random sets; appended to ``heads.json``
  (the other sets unchanged).

``--check``: the same code on E4's raw readings (``data/readings/heldout/{half}``, inst0-inst4)
must give E4's ``cc``, ``cc20``, ``cc40`` and random sets of that fold.

    .venv/bin/python scripts/diff_sets.py --choose A --out results/diff_heads/fold1
    .venv/bin/python scripts/diff_sets.py --choose A --check
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

N_WORDS, RHO, N_HEADS, N_RANDOM = 5, 0.10, 32, 5
SIZES = (20, 40)
R = READOUTS.index("jhead")
STOP = set("the and for with that this from are was were you your has have had not but its".split())
WORD = re.compile(r"^[a-z]{3,}$")
DIFF_WORDINGS = ("inst1", "inst2", "inst3", "inst4")


def build(root: Path, wordings, prefix: str, answer_removal: bool, n_common: int = N_WORDS, random_sets: bool = True):
    """Per task: the sets and the word list, from the readings under ``root``."""
    from wordfreq import zipf_frequency

    from lth.evaluation import tokenizer as get_tok
    from lth.store import load
    from lth.terms import expand_to_token_ids

    tok = get_tok()
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
        hit = np.isin(top, ids)
        return hit.any(-1).mean(0), hit.sum(-1).mean(0)

    def heads_of(idx):
        return sorted([int(h) // N_HEADS, int(h) % N_HEADS] for h in idx)

    run = {t: {w: load(root / "simple" / t, w) for w in wordings} for t in simple.TASKS}
    pooled = {t: np.concatenate([run[t][w]["top"][:, R] for w in wordings]) for t in simple.TASKS}
    task_shares = {t: shares(pooled[t]) for t in simple.TASKS}
    vocab = set.union(*(set(v) for v in task_shares.values()))
    common = sorted((w for w in vocab if all(w in task_shares[t] for t in simple.TASKS)),
                    key=lambda w: (-min(task_shares[t][w] for t in simple.TASKS), w))
    out, word_lists, sizes = {}, {}, []
    for t in simple.TASKS:
        o = [x for x in simple.TASKS if x != t]
        spec = sorted(task_shares[t], key=lambda w: (-(task_shares[t][w] - np.mean([task_shares[x].get(w, 0) for x in o])), w))
        words = spec[:N_WORDS] + [w for w in common[:n_common] if w not in spec[:N_WORDS]]
        rate, count = rate_of(pooled[t], words)
        cc = heads_of(np.flatnonzero(rate >= RHO))
        sets = [["rule", prefix, cc]]
        order = np.lexsort((np.arange(len(rate)), -count, -rate))
        ranked = {n: heads_of(order[:n]) for n in SIZES}
        sets += [["rule", f"{prefix}{n}", ranked[n]] for n in SIZES]
        removed = []
        if answer_removal:
            data = simple.dataset(t)
            items = [int(i) for w in wordings for i in run[t][w]["id_items"]]
            answers = np.array([tok(" " + data[i]["output"], add_special_tokens=False).input_ids[0] for i in items])
            answer_share = (pooled[t] == answers[:, None, None]).any(-1).mean(0)
            removed = heads_of(np.flatnonzero((rate >= RHO) & (answer_share >= RHO)))
            sets.append(["rule", f"{prefix}A", [h for h in cc if h not in removed]])
        for label, base in [*((str(n), ranked[n]) for n in SIZES), ("Nt", cc)] if random_sets else ():
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
        word_lists[t] = dict(specific=spec[:10], common=common[:10], cc=words, n_prompts=int(len(pooled[t])),
                             answer_removed=removed)
        sizes.append(dict(task=t, n=len(cc), removed=len(removed), n_prompts=len(pooled[t]), words=" ".join(words)))
    return out, word_lists, sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--choose", choices=("A", "B"), required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="run on E4's raw readings and compare with E4's sets")
    parser.add_argument("--task-words-only", action="store_true",
                        help="add the dccT sets (5 task words, no common words) to --out/heads.json")
    args = parser.parse_args()

    if args.check:
        fold = 1 if args.choose == "A" else 2
        out, _, _ = build(DATA / "readings" / "heldout" / args.choose, simple.WORDINGS["rule"], "cc", False)
        ref = json.loads((RESULTS / "heldout" / f"fold{fold}" / "heads.json").read_text())["simple"]
        bad = 0
        for t in simple.TASKS:
            mine = {n: hs for _, n, hs in out[t]["sets"]}
            theirs = {n: hs for _, n, hs in ref[t]["sets"]}
            for n, hs in mine.items():
                if theirs.get(n) != hs:
                    bad += 1
                    print(f"MISMATCH {t} {n}: mine {len(hs)}, E4 {len(theirs.get(n, []))}")
        print(f"check fold {fold}:", "every set equal to E4's" if not bad else f"{bad} mismatches")
        return

    if args.task_words_only:
        out, word_lists, sizes = build(DATA / "readings" / "diff" / args.choose, DIFF_WORDINGS, "dccT", True,
                                       n_common=0, random_sets=False)
        path = args.out / "heads.json"
        heads = json.loads(path.read_text())
        for t, v in heads["simple"].items():
            v["sets"] = [s for s in v["sets"] if not s[1].startswith("dccT")] + out[t]["sets"]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(heads, indent=1))
        tmp.replace(path)
        (args.out / "word_lists_T.json").write_text(json.dumps(word_lists, indent=1))
        for s in sizes:
            print(f"{s['task']:17s} dccT {s['n']:3d} (answer heads removed in dccTA: {s['removed']}) | {s['words']}")
        return
    out, word_lists, sizes = build(DATA / "readings" / "diff" / args.choose, DIFF_WORDINGS, "dcc", True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "heads.json").write_text(json.dumps({"simple": out}, indent=1))
    (args.out / "word_lists.json").write_text(json.dumps(word_lists, indent=1))
    with (args.out / "set_sizes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sizes[0]))
        w.writeheader()
        w.writerows(sizes)
    for s in sizes:
        print(f"{s['task']:17s} dcc {s['n']:3d} (answer heads removed in dccA: {s['removed']}) "
              f"prompts {s['n_prompts']:3d} | {s['words']}")


if __name__ == "__main__":
    main()
