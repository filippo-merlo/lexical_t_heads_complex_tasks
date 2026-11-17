"""Rebuilt design (paper/head-sets-simple-design.tex, Secs. 3-5): the head sets of the two published rules
and of our additions, per fold, from the choosing half only. No GPU.

Choosing pool (Sec. 3.3): Yang et al.'s five templates; a template with fewer than ``MIN_CORRECT``
correctly answered prompts on the half is not used; up to ``BUDGET`` prompts per task, spread evenly
over the templates used (the first ones of each, in item order, from the half's readings
``data/readings/heldout/<half>``).

- **Word list and D** (Sec. 5.2): ``scripts/e13_lists.py``'s word rule and generic words on the pool's
  J-head top 10; D = share of pool prompts whose top 10 holds a word of the 30-word list.
- **Segments** (Sec. 5.3): ``scripts/e20_bai_attention.py``'s ``categorise`` with the fold's list, on the
  stored last-token attention rows (``scripts/rebuild_attention.py``); a head's score on a segment is its
  attention mass on it averaged over the pool prompts where the segment is not empty; B (non-first) is
  one minus the mass on the first token.
- **Yang terms** (Sec. 4.1): share of pool prompts whose Logit-head (``yangL``) or J-head (``yangJ``)
  top 10 holds a token of Yang et al.'s term list (``lth.terms``); heads with share 0 are not ranked;
  ties by the mean number of term tokens. Own form: Logit-head, share >= 10% on the first 100 correctly
  answered prompts of the template most accurate on the half.
- **Causal** (Sec. 4.2): Davidson et al.'s score on the half (``results/heldout/fold<f>/fv/cie.npz``),
  averaged over the three kinds of uninformative text. Own form: top 20 by the score averaged over
  tasks. Generic heads (Sec. 5.5): two-group split of the J-head generic score (pool prompts) among the
  top 30 of the shared causal ranking; none if more than 15.
- **Rankings**: 1 = best, ties averaged, remaining ties to the lower layer then head. Shared: each
  feature's within-task ranks averaged over tasks and ranked again; a combination averages its features'
  ranks (per task: within-task ranks).
- **Screen** (Sec. 5.4): every non-empty subset of the nine features at ``SCREEN_N`` heads, per task and
  shared.
- **Random** (Sec. 3.5): for every ladder rule, own form, scope and size, ``N_RANDOM`` sets with as many
  heads in each layer, drawn from all heads of those layers.

Writes ``results/rebuild/additions/fold<f>/``: ``heads.json`` (``scripts/patch.py --set-list`` format),
``word_lists.json``, ``features.npz``, ``generic.json``, ``sets.csv`` (size, median layer, overlap with
Causal of the same scope and size) and ``screen.json`` (distinct screen sets).

    .venv/bin/python scripts/rebuild_sets.py
    .venv/bin/python scripts/rebuild_sets.py --show antonym      # segments of one pool prompt per template
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, READOUTS, RESULTS, simple  # noqa: E402

OUT = RESULTS / "rebuild" / "additions"
FOLDS = {1: "A", 2: "B"}
SIZES = (10, 20, 40, 60, 80, 100, 120)
SCREEN_N = 40
BUDGET, MIN_CORRECT = 300, 10
N_RANDOM, N_HEADS, N_ALL = 3, 32, 1024
KINDS = ("equiprobable", "wikitext", "other_task")
LH, JH = READOUTS.index("logithead"), READOUTS.index("jhead")
# Features: code -> (name, segment of e20's CATS; None for D and B)
FEATURES = {"D": ("decodes a list word", None), "B": ("non-first", None), "T": ("task words", "task"),
            "O": ("other words", "content"), "S": ("stopwords", "function"), "P": ("punctuation", "punct"),
            "M": ("markers", "template"), "Q": ("query", "query"), "L": ("last token", "self")}
COMBOS = ["".join(c) for k in range(1, len(FEATURES) + 1) for c in itertools.combinations(FEATURES, k)]
YANG_OWN_SHARE, YANG_OWN_PROMPTS, DAVIDSON_N, GENERIC_TOP, GENERIC_MAX = 0.10, 100, 20, 30, 15


def rankdata(x: np.ndarray, method: str = "average") -> np.ndarray:
    """1 = smallest, ties averaged."""
    return pd.Series(np.asarray(x, dtype=np.float64)).rank(method=method).to_numpy()


def ranks(score: np.ndarray) -> np.ndarray:
    """1 = largest, ties averaged."""
    return rankdata(-np.asarray(score, dtype=np.float64))


def order_of(rank: np.ndarray) -> np.ndarray:
    return np.argsort(rank, kind="stable")  # ties: lower index = lower layer, then head


def heads_of(idx) -> list[list[int]]:
    return sorted([int(i) // N_HEADS, int(i) % N_HEADS] for i in idx)


def top(order, n, drop=frozenset()) -> list[list[int]] | None:
    kept = [i for i in order if int(i) not in drop][:n]
    return heads_of(kept) if len(kept) == n else None


def random_sets(base: list[list[int]], key: str) -> list[list[list[int]]]:
    per_layer = Counter(l for l, _ in base)
    out = []
    for i in range(N_RANDOM):
        rng = np.random.default_rng([zlib.crc32(key.encode()), i])
        out.append(sorted([l, int(h)] for l in sorted(per_layer)
                          for h in rng.choice(N_HEADS, per_layer[l], replace=False)))
    return out


def pool(half: str, task: str):
    """The pool prompts as (template index, item) and their readings [P, 2, 1024, 10]."""
    from lth.store import load

    root = DATA / "readings" / "heldout" / half / "simple" / task
    loaded = [(j, load(root, w)) for j, w in enumerate(simple.WORDINGS["rule"])]
    used = [(j, d) for j, d in loaded if len(d["id_items"]) >= MIN_CORRECT]
    per = BUDGET // len(used)
    keys = [(j, int(i)) for j, d in used for i in d["id_items"][:per]]
    tops = np.concatenate([d["top"][:per] for _, d in used])
    return keys, tops, {simple.WORDINGS["rule"][j]: min(per, len(d["id_items"])) for j, d in used}


def segment_masses(half: str, task: str, keys, words: list[str], tok):
    """Mean attention mass of every head on each segment, over the pool prompts where it is not empty."""
    from e20_bai_attention import CATS, categorise, task_pattern

    att = np.load(DATA / "readings" / "rebuild" / f"attention_{half}" / f"{task}.npz")
    start = np.concatenate([[0], np.cumsum(att["lengths"])[:-1]])
    where = {(int(j), int(i)): (int(s), int(n)) for j, i, s, n in zip(att["wordings"], att["items"], start, att["lengths"])}
    data, pattern = simple.dataset(task), task_pattern(words)
    prompts = {w: simple.prompts(task, w)[0] for w in simple.WORDINGS["rule"]}
    total, count = np.zeros((len(CATS), N_ALL)), np.zeros(len(CATS))
    for j, item in keys:
        w = simple.WORDINGS["rule"][j]
        text = prompts[w][item]
        enc = tok(text, return_offsets_mapping=True)
        s, n = where[(j, item)]
        assert n == len(enc.input_ids), (task, w, item)
        cat, _, _ = categorise(text, simple.instruction_dict()[task][w[4:]], data[item]["input"],
                               enc.input_ids, enc.offset_mapping, pattern)
        rows = att["rows"][s:s + n].astype(np.float64)  # [T, 1024]
        for c in range(len(CATS)):
            m = cat == c
            if m.any():
                total[c] += rows[m].sum(0)
                count[c] += 1
    masses = {c: total[k] / count[k] if count[k] else np.zeros(N_ALL) for k, c in enumerate(CATS)}
    return masses, {c: int(count[k]) for k, c in enumerate(CATS)}


def generic_scores(tops: dict[str, np.ndarray], heads) -> dict[int, float]:
    def counts(rows):
        c = Counter()
        for r in rows:
            c.update(set(r.tolist()))
        return c

    def cos(a, b):
        num = sum(v * b.get(k, 0) for k, v in a.items())
        den = np.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
        return num / den if den else 0.0

    out = {}
    for s in heads:
        c = {t: counts(tops[t][:, JH, s]) for t in simple.TASKS}
        out[s] = float(np.mean([cos(c[a], c[b]) for a, b in itertools.combinations(simple.TASKS, 2)]))
    return out


def two_group_cut(values) -> int:
    v = np.asarray(values, dtype=np.float64)
    cost = [k * v[:k].var() + (len(v) - k) * v[k:].var() for k in range(1, len(v))]
    return int(np.argmin(cost)) + 1


def name(i: int) -> str:
    return f"L{i // N_HEADS}H{i % N_HEADS}"


def build(fold: int, tok, tmap, words, generic_words):
    from e13_lists import N_WORDS, shares

    from lth.store import load
    from lth.terms import expand_to_token_ids, simple_term_lists

    half = FOLDS[fold]
    out = OUT / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads((RESULTS / "heldout" / "split.json").read_text())
    feats, lists, yang, tops, used, seg_counts, yang_own = {}, {}, {}, {}, {}, {}, {}
    for t in simple.TASKS:
        keys, top_t, used[t] = pool(half, t)
        tops[t] = top_t
        jt = top_t[:, JH].astype(np.int64)
        sh = shares(top_t[:, JH], tmap, len(words))
        order = [i for i in np.argsort(-sh, kind="stable") if sh[i] > 0 and words[i] not in generic_words][:N_WORDS]
        lists[t] = {"words": [words[i] for i in order], "shares": [round(float(sh[i]), 6) for i in order],
                    "n_prompts": len(keys), "prompts_per_template": used[t]}
        member = np.zeros(len(words) + 1, dtype=bool)
        member[order] = True
        f = {"D": member[tmap[jt]].any(-1).mean(0)}
        masses, seg_counts[t] = segment_masses(half, t, keys, lists[t]["words"], tok)
        f["B"] = 1.0 - masses["bos"]
        for code, (_, seg) in FEATURES.items():
            if seg is not None:
                f[code] = masses[seg]
        feats[t] = f
        terms = np.array(sorted(set(expand_to_token_ids(tok, simple_term_lists()[t]))))
        for readout, r in (("L", LH), ("J", JH)):
            hit = np.isin(top_t[:, r], terms)
            yang[(t, readout)] = (hit.any(-1).mean(0), hit.sum(-1).mean(0))
        # Own form: Logit-head on the first 100 correct prompts of the template most accurate on the half.
        half_items = set(split[t][half])
        acc = {}
        for w in simple.WORDINGS["rule"]:
            recs = [r for r in load(DATA / "readings" / "simple" / t, w)["records"] if int(r["item"]) in half_items]
            acc[w] = sum(r["correct"] for r in recs) / len(recs)
        best = max(simple.WORDINGS["rule"], key=lambda w: (acc[w], -simple.WORDINGS["rule"].index(w)))
        own_top = load(DATA / "readings" / "heldout" / half / "simple" / t, best)["top"][:YANG_OWN_PROMPTS, LH]
        share = np.isin(own_top, terms).any(-1).mean(0)
        yang_own[t] = {"template": best, "accuracy": round(acc[best], 3), "heads": heads_of(np.flatnonzero(share >= YANG_OWN_SHARE))}

    cie = np.load(RESULTS / "heldout" / f"fold{fold}" / "fv" / "cie.npz")
    causal = {t: np.mean([cie[f"{t}/{k}"] for k in KINDS], axis=0).ravel() for t in simple.TASKS}

    # Per-task ranks of every single score; shared ranks: within-task ranks averaged, ranked again.
    single = {t: {**{c: ranks(feats[t][c]) for c in FEATURES}, "causal": ranks(causal[t])} for t in simple.TASKS}
    shared_single = {c: rankdata(np.mean([single[t][c] for t in simple.TASKS], axis=0), method="average")
                     for c in list(FEATURES) + ["causal"]}
    shared_causal_order = order_of(shared_single["causal"])

    # Generic heads among the top 30 of the shared causal ranking.
    top30 = [int(i) for i in shared_causal_order[:GENERIC_TOP]]
    g = generic_scores(tops, top30)
    by_score = sorted(top30, key=lambda x: -g[x])
    k = two_group_cut([g[x] for x in by_score])
    generic = set(by_score[:k]) if k <= GENERIC_MAX else set()

    def yang_order(t, readout):
        rate, count = yang[(t, readout)]
        eligible = np.flatnonzero(rate > 0)
        return eligible[np.lexsort((eligible, -count[eligible], -rate[eligible]))]

    def yang_shared_order(readout):
        rank = np.mean([ranks(yang[(t, readout)][0]) for t in simple.TASKS], axis=0)
        eligible = np.flatnonzero(np.max([yang[(t, readout)][0] for t in simple.TASKS], axis=0) > 0)
        return eligible[np.argsort(rank[eligible], kind="stable")]

    per_task, shared, rows = {t: [] for t in simple.TASKS}, [], []

    def add(target: list, rule: str, n, heads, key: str):
        if heads is None:  # cannot fill n heads: not run
            return
        label = f"{rule}{n}" if n != "" else rule
        target.append(["rule", label, heads])
        target += [["rule", f"random_{label}_{i}", r] for i, r in enumerate(random_sets(heads, f"{fold}/{key}/{label}"))]

    for t in simple.TASKS:
        add(per_task[t], "yang_own", "", yang_own[t]["heads"] or None, t)
        for n in SIZES:
            for readout in "LJ":
                y = yang_order(t, readout)
                add(per_task[t], f"yang{readout}", n, heads_of(y[:n]) if len(y) >= n else None, t)
            add(per_task[t], "D", n, top(order_of(single[t]["D"]), n), t)
            corder = order_of(single[t]["causal"])
            add(per_task[t], "causal", n, top(corder, n), t)
            add(per_task[t], "causalnogen", n, top(corder, n, generic), t)
        for combo in COMBOS:
            r = np.mean([single[t][c] for c in combo], axis=0)
            per_task[t].append(["rule", f"screen_{combo}", top(order_of(r), SCREEN_N)])
    davidson = order_of(ranks(np.mean([causal[t] for t in simple.TASKS], axis=0)))
    add(shared, "shared_davidson_own", "", top(davidson, DAVIDSON_N), "shared")
    for n in SIZES:
        for readout in "LJ":
            y = yang_shared_order(readout)
            add(shared, f"shared_yang{readout}", n, heads_of(y[:n]) if len(y) >= n else None, "shared")
        add(shared, "shared_D", n, top(order_of(shared_single["D"]), n), "shared")
        add(shared, "shared_causal", n, top(shared_causal_order, n), "shared")
        add(shared, "shared_causalnogen", n, top(shared_causal_order, n, generic), "shared")
    for combo in COMBOS:
        r = np.mean([shared_single[c] for c in combo], axis=0)
        shared.append(["rule", f"shared_screen_{combo}", top(order_of(r), SCREEN_N)])

    heads = {"simple": {t: {"sets": per_task[t] + shared} for t in simple.TASKS}}
    (out / "heads.json").write_text(json.dumps(heads))

    # Screen: distinct sets.
    screen = {"per_task": {t: len({frozenset(map(tuple, h)) for _, nm, h in per_task[t] if nm.startswith("screen_")})
                           for t in simple.TASKS},
              "shared": len({frozenset(map(tuple, h)) for _, nm, h in shared if nm.startswith("shared_screen_")}),
              "combos": len(COMBOS)}
    (out / "screen.json").write_text(json.dumps(screen, indent=1))

    # Report rows (no random sets).
    for t in simple.TASKS:
        sets = {nm: {tuple(h) for h in hs} for _, nm, hs in per_task[t] + shared if not nm.startswith("random_")}
        for nm, hs in sets.items():
            n = len(hs)
            ref = sets.get(f"shared_causal{n}" if nm.startswith("shared") else f"causal{n}", set())
            if nm.startswith("shared") and t != simple.TASKS[0]:
                continue
            rows.append(dict(fold=fold, task="all" if nm.startswith("shared") else t, set=nm, n_heads=n,
                             layer_median=float(np.median([l for l, _ in hs])) if hs else "",
                             generic=len(hs & {(x // N_HEADS, x % N_HEADS) for x in generic}),
                             overlap_causal=len(hs & ref) if ref else ""))
    with (out / "sets.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(out / "features.npz", **{f"{t}/{c}": feats[t][c] for t in simple.TASKS for c in FEATURES},
                        **{f"{t}/causal": causal[t] for t in simple.TASKS})
    (out / "word_lists.json").write_text(json.dumps(dict(
        fold=fold, choosing_half=half, generic_words=sorted(generic_words), tasks=lists,
        segment_prompts=seg_counts), ensure_ascii=False, indent=1))
    (out / "generic.json").write_text(json.dumps(dict(
        fold=fold, choosing_half=half, top30_by_score=[[name(x), round(g[x], 4)] for x in by_score],
        generic=sorted(name(x) for x in generic), yang_own={t: {k: v for k, v in d.items() if k != "heads"} | {"n": len(d["heads"])}
                                                          for t, d in yang_own.items()}), indent=1))
    return generic, screen


def show(task: str) -> None:
    from e20_bai_attention import CATS, categorise, task_pattern

    from lth.evaluation import tokenizer

    tok = tokenizer()
    lists = json.loads((OUT / "fold1" / "word_lists.json").read_text())["tasks"]
    pat = task_pattern(lists[task]["words"])
    keys, _, _ = pool("A", task)
    data = simple.dataset(task)
    seen = set()
    for j, item in keys:
        if j in seen:
            continue
        seen.add(j)
        w = simple.WORDINGS["rule"][j]
        text = simple.prompts(task, w)[0][item]
        enc = tok(text, return_offsets_mapping=True)
        cat, _, _ = categorise(text, simple.instruction_dict()[task][w[4:]], data[item]["input"],
                               enc.input_ids, enc.offset_mapping, pat)
        print(f"{w}: " + " ".join(f"[{tok.decode([i])!r} {CATS[c]}]" for i, c in zip(enc.input_ids, cat)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--show", default=None, help="print the segments of one pool prompt per template of a task (fold 1)")
    args = parser.parse_args()
    if args.show:
        show(args.show)
        return
    from e13_lists import N_GENERIC, shares, token_words

    from lth.evaluation import tokenizer

    tok = tokenizer()
    tmap, words = token_words(tok)
    wiki = shares(np.load(DATA / "readings" / "neutral" / "wikitext.npz")["top"][:, JH], tmap, len(words))
    generic_words = {words[i] for i in np.argsort(-wiki, kind="stable")[:N_GENERIC]}
    for fold in FOLDS:
        generic, screen = build(fold, tok, tmap, words, generic_words)
        print(f"fold {fold}: generic heads {sorted(name(x) for x in generic)}; distinct screen sets: shared "
              f"{screen['shared']}/{screen['combos']}, per task {min(screen['per_task'].values())}-"
              f"{max(screen['per_task'].values())}", flush=True)


if __name__ == "__main__":
    main()
