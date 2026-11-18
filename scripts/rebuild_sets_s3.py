"""Rebuilt design, stage 3 (paper/head-sets-simple-design.tex, Secs. 6.1-6.2): the rules re-chosen on
other prompts, per fold, from the choosing half only. No GPU.

Two choosing pools, built as ``scripts/rebuild_sets.py`` builds the stage-2 sets from Yang et al.'s five
templates (same budget of 300 correctly answered prompts per task spread evenly over the pool's templates
or formats with at least ten such prompts, same word rule and generic words, same rankings, sizes, scopes,
random sets and generic-head cut):

- ``wordings`` (Sec. 6.1, re-chosen): Yang et al.'s five templates and the task's 15 P wordings. Segments
  as in stage 2 (``e20_bai_attention.categorise``), with the label of a label wording before the query
  also a marker; causal score: ``scripts/davidson_fv.py --instructions p``.
- ``examples`` (Sec. 6.2, re-chosen): five-shot prompts of the three P formats. Features: D, non-first
  and attention to the demonstrations' inputs (X), their outputs (Y), the separators (S), the query (Q)
  and the last token (L); causal score: ``scripts/rebuild_demo_cie.py``.

Rules: Yang terms (J-head), D, Causal, Causal - generic at every size, per task and shared, with their
random sets, and the combination screen at 40 heads (Selected is then picked by ``scripts/rebuild_select.py
--pool <pool>``). Writes ``results/rebuild/<pool>/fold<f>/``: ``heads.json``, ``word_lists.json``,
``features.npz``, ``generic.json``, ``sets.csv``, ``screen.json``.

    .venv/bin/python scripts/rebuild_sets_s3.py --pool wordings
    .venv/bin/python scripts/rebuild_sets_s3.py --pool examples --show antonym   # segments, one prompt per format
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS, simple  # noqa: E402
from rebuild_sets import (BUDGET, FEATURES, FOLDS, GENERIC_MAX, GENERIC_TOP, JH, KINDS, MIN_CORRECT, N_ALL,  # noqa: E402
                          N_HEADS, SCREEN_N, SIZES, generic_scores, heads_of, name, order_of, random_sets, rankdata,
                          ranks, top, two_group_cut)

EX_CATS = ("bos", "query", "self", "outputs", "inputs", "separators")
EX_FEATURES = {"D": ("decodes a list word", None), "B": ("non-first", None), "X": ("demonstration inputs", "inputs"),
               "Y": ("demonstration outputs", "outputs"), "S": ("separators", "separators"), "Q": ("query", "query"),
               "L": ("last token", "self")}


def combos(features: dict) -> list[str]:
    return ["".join(c) for k in range(1, len(features) + 1) for c in itertools.combinations(features, k)]


def features_of(pool: str) -> dict:
    return EX_FEATURES if pool == "examples" else FEATURES


def wordings_of(pool: str, task: str) -> tuple[str, ...]:
    return (simple.WORDINGS["rule"] + simple.split_wordings(task, "P") if pool == "wordings"
            else simple.split_formats("P"))


def template_of(task: str, wording: str) -> str:
    return (simple.instruction_dict()[task][wording[4:]] if wording.startswith("inst")
            else simple.e4b_templates()[task][wording])


def categorise_instruction(task, wording, text, query, ids, off, pattern) -> np.ndarray:
    """Stage-2 segments (``e20_bai_attention.CATS``); in a label wording the label before the query is
    also a marker (when not a task word)."""
    from e20_bai_attention import CATS, categorise

    template = template_of(task, wording)
    cat, _, _ = categorise(text, template, query, ids, off, pattern)
    if simple.wording_group(wording).startswith("label"):
        label = len(template.split("{input}")[0].rstrip())
        for i, (a, _) in enumerate(off):
            if i > 0 and a < label and CATS[cat[i]] in ("content", "function", "punct"):
                cat[i] = CATS.index("template")
    return cat


def categorise_example(task, wording, item, text, off) -> np.ndarray:
    """Segment (``EX_CATS``) of every position of the five-shot prompt of ``item`` in a format."""
    data = simple.dataset(task)
    q_bos, qa, a_bos, delim = simple.EXAMPLE_FORMATS[wording[len(simple.FORMAT_PREFIX):]]
    demos = simple.demo_items(task, item, simple.item_halves(task)[item])
    inputs, outputs, pos = [], [], 0
    for j in demos:
        x, y = data[j]["input"], data[j]["output"]
        pos += len(q_bos)
        inputs.append((pos, pos + len(x)))
        pos += len(x) + len(qa) + len(a_bos)
        outputs.append((pos, pos + len(y)))
        pos += len(y) + len(delim)
    pos += len(q_bos)
    query = (pos, pos + len(data[item]["input"]))
    assert pos + len(data[item]["input"]) + len(qa) == len(text), (task, wording, item)

    def hit(a, b, spans):
        return any(a < e and b > s for s, e in spans)
    cat = np.empty(len(off), dtype=np.int8)
    for i, (a, b) in enumerate(off):
        c = ("bos" if i == 0 else "query" if hit(a, b, [query]) else "self" if i == len(off) - 1
             else "outputs" if hit(a, b, outputs) else "inputs" if hit(a, b, inputs) else "separators")
        cat[i] = EX_CATS.index(c)
    return cat


def pool(pool_name: str, half: str, task: str):
    """The pool prompts as (wording, item), their readings [P, 2, 1024, 10] and prompts per wording."""
    from lth.store import load

    root = DATA / "readings" / "heldout" / half / "simple" / task
    loaded = [(w, load(root, w)) for w in wordings_of(pool_name, task)]
    used = [(w, d) for w, d in loaded if len(d["id_items"]) >= MIN_CORRECT]
    per = BUDGET // len(used)
    keys = [(w, int(i)) for w, d in used for i in d["id_items"][:per]]
    tops = np.concatenate([d["top"][:per] for _, d in used])
    return keys, tops, {w: min(per, len(d["id_items"])) for w, d in used}


def attention(pool_name: str, half: str, task: str) -> dict:
    """(wording, item) -> last-token attention rows [T, 1024] (float16)."""
    files = ([("attention", simple.WORDINGS["rule"]), ("attention_p", None)] if pool_name == "wordings"
             else [("attention_fmt", None)])
    out = {}
    for stem, names in files:
        att = np.load(DATA / "readings" / "rebuild" / f"{stem}_{half}" / f"{task}.npz")
        names = list(names or att["names"])
        start = np.concatenate([[0], np.cumsum(att["lengths"])[:-1]])
        for j, i, s, n in zip(att["wordings"], att["items"], start, att["lengths"]):
            out[(names[int(j)], int(i))] = att["rows"][int(s):int(s) + int(n)]
    return out


def segment_masses(pool_name, half, task, keys, words, tok):
    """Mean attention mass of every head on each segment, over the pool prompts where it is not empty."""
    from e20_bai_attention import CATS, task_pattern

    cats = EX_CATS if pool_name == "examples" else CATS
    rows_of = attention(pool_name, half, task)
    data, pattern = simple.dataset(task), task_pattern(words)
    prompts = {w: simple.prompts(task, w)[0] for w in {w for w, _ in keys}}
    total, count = np.zeros((len(cats), N_ALL)), np.zeros(len(cats))
    for w, item in keys:
        text = prompts[w][item]
        enc = tok(text, return_offsets_mapping=True)
        rows = rows_of[(w, item)].astype(np.float64)
        assert len(rows) == len(enc.input_ids), (task, w, item)
        if pool_name == "examples":
            cat = categorise_example(task, w, item, text, enc.offset_mapping)
        else:
            cat = categorise_instruction(task, w, text, data[item]["input"], enc.input_ids, enc.offset_mapping, pattern)
        for c in range(len(cats)):
            m = cat == c
            if m.any():
                total[c] += rows[m].sum(0)
                count[c] += 1
    masses = {c: total[k] / count[k] if count[k] else np.zeros(N_ALL) for k, c in enumerate(cats)}
    return masses, {c: int(count[k]) for k, c in enumerate(cats)}


def causal_scores(pool_name: str, fold: int) -> dict[str, np.ndarray]:
    if pool_name == "wordings":
        cie = np.load(RESULTS / "rebuild" / "wordings" / f"fold{fold}" / "fv" / "cie.npz")
        return {t: np.mean([cie[f"{t}/{k}"] for k in KINDS], axis=0).ravel() for t in simple.TASKS}
    cie = np.load(RESULTS / "rebuild" / "examples" / f"fold{fold}" / "cie" / "cie.npz")
    return {t: cie[t].ravel() for t in simple.TASKS}


def build(pool_name: str, fold: int, tok, tmap, words, generic_words):
    from e13_lists import N_WORDS, shares

    from lth.terms import expand_to_token_ids, simple_term_lists

    half = FOLDS[fold]
    feats_def = features_of(pool_name)
    out = RESULTS / "rebuild" / pool_name / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    feats, lists, yang, tops, seg_counts = {}, {}, {}, {}, {}
    for t in simple.TASKS:
        keys, top_t, used = pool(pool_name, half, t)
        tops[t] = top_t
        jt = top_t[:, JH].astype(np.int64)
        sh = shares(top_t[:, JH], tmap, len(words))
        order = [i for i in np.argsort(-sh, kind="stable") if sh[i] > 0 and words[i] not in generic_words][:N_WORDS]
        lists[t] = {"words": [words[i] for i in order], "shares": [round(float(sh[i]), 6) for i in order],
                    "n_prompts": len(keys), "prompts_per_wording": used}
        member = np.zeros(len(words) + 1, dtype=bool)
        member[order] = True
        f = {"D": member[tmap[jt]].any(-1).mean(0)}
        masses, seg_counts[t] = segment_masses(pool_name, half, t, keys, lists[t]["words"], tok)
        f["B"] = 1.0 - masses["bos"]
        for code, (_, seg) in feats_def.items():
            if seg is not None:
                f[code] = masses[seg]
        feats[t] = f
        terms = np.array(sorted(set(expand_to_token_ids(tok, simple_term_lists()[t]))))
        hit = np.isin(top_t[:, JH], terms)
        yang[t] = (hit.any(-1).mean(0), hit.sum(-1).mean(0))
        print(f"fold {fold} {pool_name} {t}: {len(keys)} pool prompts over {len(used)} wordings", flush=True)

    causal = causal_scores(pool_name, fold)
    single = {t: {**{c: ranks(feats[t][c]) for c in feats_def}, "causal": ranks(causal[t])} for t in simple.TASKS}
    shared_single = {c: rankdata(np.mean([single[t][c] for t in simple.TASKS], axis=0))
                     for c in list(feats_def) + ["causal"]}
    shared_causal_order = order_of(shared_single["causal"])
    top30 = [int(i) for i in shared_causal_order[:GENERIC_TOP]]
    g = generic_scores(tops, top30)
    by_score = sorted(top30, key=lambda x: -g[x])
    k = two_group_cut([g[x] for x in by_score])
    generic = set(by_score[:k]) if k <= GENERIC_MAX else set()

    def yang_order(t):
        rate, count = yang[t]
        eligible = np.flatnonzero(rate > 0)
        return eligible[np.lexsort((eligible, -count[eligible], -rate[eligible]))]

    def yang_shared_order():
        rank = np.mean([ranks(yang[t][0]) for t in simple.TASKS], axis=0)
        eligible = np.flatnonzero(np.max([yang[t][0] for t in simple.TASKS], axis=0) > 0)
        return eligible[np.argsort(rank[eligible], kind="stable")]

    per_task, shared = {t: [] for t in simple.TASKS}, []

    def add(target, rule, n, heads, key):
        if heads is None:
            return
        label = f"{rule}{n}"
        target.append(["rule", label, heads])
        target += [["rule", f"random_{label}_{i}", r]
                   for i, r in enumerate(random_sets(heads, f"{pool_name}/{fold}/{key}/{label}"))]

    for t in simple.TASKS:
        for n in SIZES:
            y = yang_order(t)
            add(per_task[t], "yangJ", n, heads_of(y[:n]) if len(y) >= n else None, t)
            add(per_task[t], "D", n, top(order_of(single[t]["D"]), n), t)
            corder = order_of(single[t]["causal"])
            add(per_task[t], "causal", n, top(corder, n), t)
            add(per_task[t], "causalnogen", n, top(corder, n, generic), t)
        for combo in combos(feats_def):
            r = np.mean([single[t][c] for c in combo], axis=0)
            per_task[t].append(["rule", f"screen_{combo}", top(order_of(r), SCREEN_N)])
    for n in SIZES:
        y = yang_shared_order()
        add(shared, "shared_yangJ", n, heads_of(y[:n]) if len(y) >= n else None, "shared")
        add(shared, "shared_D", n, top(order_of(shared_single["D"]), n), "shared")
        add(shared, "shared_causal", n, top(shared_causal_order, n), "shared")
        add(shared, "shared_causalnogen", n, top(shared_causal_order, n, generic), "shared")
    for combo in combos(feats_def):
        r = np.mean([shared_single[c] for c in combo], axis=0)
        shared.append(["rule", f"shared_screen_{combo}", top(order_of(r), SCREEN_N)])

    (out / "heads.json").write_text(json.dumps({"simple": {t: {"sets": per_task[t] + shared} for t in simple.TASKS}}))
    screen = {"per_task": {t: len({frozenset(map(tuple, h)) for _, nm, h in per_task[t] if nm.startswith("screen_")})
                           for t in simple.TASKS},
              "shared": len({frozenset(map(tuple, h)) for _, nm, h in shared if nm.startswith("shared_screen_")}),
              "combos": len(combos(feats_def))}
    (out / "screen.json").write_text(json.dumps(screen, indent=1))
    rows = []
    for t in simple.TASKS:
        for _, nm, hs in per_task[t] + (shared if t == simple.TASKS[0] else []):
            if nm.startswith("random_") or "screen_" in nm:
                continue
            hs = {tuple(h) for h in hs}
            rows.append(dict(fold=fold, task="all" if nm.startswith("shared") else t, set=nm, n_heads=len(hs),
                             layer_median=float(np.median([l for l, _ in hs])),
                             generic=len(hs & {(x // N_HEADS, x % N_HEADS) for x in generic})))
    with (out / "sets.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(out / "features.npz", **{f"{t}/{c}": feats[t][c] for t in simple.TASKS for c in feats_def},
                        **{f"{t}/causal": causal[t] for t in simple.TASKS})
    (out / "word_lists.json").write_text(json.dumps(dict(
        fold=fold, choosing_half=half, pool=pool_name, generic_words=sorted(generic_words), tasks=lists,
        segment_prompts=seg_counts), ensure_ascii=False, indent=1))
    (out / "generic.json").write_text(json.dumps(dict(
        fold=fold, choosing_half=half, pool=pool_name, top30_by_score=[[name(x), round(g[x], 4)] for x in by_score],
        generic=sorted(name(x) for x in generic)), indent=1))
    return generic, screen


def show(pool_name: str, task: str) -> None:
    from e20_bai_attention import CATS, task_pattern

    from lth.evaluation import tokenizer

    tok = tokenizer()
    pattern = task_pattern(json.loads((RESULTS / "rebuild" / "additions" / "fold1" / "word_lists.json").read_text())
                           ["tasks"][task]["words"])
    keys, _, _ = pool(pool_name, "A", task)
    data, seen = simple.dataset(task), set()
    for w, item in keys:
        if w in seen:
            continue
        seen.add(w)
        text = simple.prompts(task, w)[0][item]
        enc = tok(text, return_offsets_mapping=True)
        if pool_name == "examples":
            cat, names = categorise_example(task, w, item, text, enc.offset_mapping), EX_CATS
        else:
            cat, names = categorise_instruction(task, w, text, data[item]["input"], enc.input_ids, enc.offset_mapping,
                                                pattern), CATS
        print(f"{w}: " + " ".join(f"[{tok.decode([i])!r} {names[c]}]" for i, c in zip(enc.input_ids, cat)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", choices=("wordings", "examples"), required=True)
    parser.add_argument("--show", default=None, help="print the segments of one pool prompt per wording of a task")
    args = parser.parse_args()
    if args.show:
        show(args.pool, args.show)
        return
    from e13_lists import N_GENERIC, shares, token_words

    from lth.evaluation import tokenizer

    tok = tokenizer()
    tmap, words = token_words(tok)
    wiki = shares(np.load(DATA / "readings" / "neutral" / "wikitext.npz")["top"][:, JH], tmap, len(words))
    generic_words = {words[i] for i in np.argsort(-wiki, kind="stable")[:N_GENERIC]}
    for fold in FOLDS:
        generic, screen = build(args.pool, fold, tok, tmap, words, generic_words)
        print(f"fold {fold} {args.pool}: generic heads {sorted(name(x) for x in generic)}; distinct screen sets: shared "
              f"{screen['shared']}/{screen['combos']}, per task {min(screen['per_task'].values())}-"
              f"{max(screen['per_task'].values())}", flush=True)


if __name__ == "__main__":
    main()
