"""E5 (paper/proposal-llm-heads.tex): head sets from the LLM's scores, references, review file.

For each task and fold (fold 1 = scores on half A, fold 2 = half B), readout and condition
(``all``, ``batched``), from ``data/llm_heads/scores`` (``scripts/llm_heads_select.py``):
- the heads ordered by score, ties by anonymous id (a random order, seed 0);
- after removal: heads with an answer share of 0.10 or more are removed first; the first 20, 30
  and 40 form ``{cond}{R}{N}``, e.g. ``allJ20``, ``batL40`` (R = J for J-head, L for Logit-head);
- without removal: the same from the scores alone, ``{cond}{R}{N}raw`` (identical sets are
  patched once by ``patch.py``);
- random draws for every set after removal: five, seeds 0-4, with the same number of heads in
  each layer, drawn outside it, as E4 (``scripts/heldout_sets.py``): ``random_{set}_{i}``;
- references at 30 heads: ``fv30``, the first 30 by the fold's function-vector score
  (``scripts/heldout_fv_sets.py``), and ``cc30``, the first 30 of E4's CC ranking, recomputed
  with ``scripts/heldout_sets.py`` at sizes 20, 30, 40 (its 20 and 40 must equal E4's sets);
- ``fv20``, E4's own set, copied: patched again, it must give E4's accuracies (a check that the
  patching here runs as E4's);
- larger sets (added 2026-11-07, after the first patching): the same rule at 60 and 80 heads and
  at the size of the fold's full CC set (``{cond}{R}Nt``, 99 or 97 heads on English--German), only
  where every head of the set has a score above 0 in both folds (beyond that the order is the
  random order of the heads scored 0); their random draws as above; references ``cc60``, ``cc80``
  (CC ranking) and ``fv60``, ``fv80`` (function-vector ranking). E4's ``cc`` and ``fvNt`` are the
  references at N_t.

Tasks after the pilot (country-capital, 2026-11-07) have the ``all`` condition only (user): the
batched sets and comparisons are skipped where their scores do not exist.

Writes ``results/llm_heads/fold{1,2}/heads.json`` (the set-list layout of E4; the sets of the
tasks not given are kept), the stability and overlap measures ``results/llm_heads/agreement.csv``
(rows of the other tasks kept), and the review file ``data/review/llm_heads_{task}.md``.

    .venv/bin/python scripts/llm_heads_sets.py --tasks english-german
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS  # noqa: E402

SCORES = DATA / "llm_heads" / "scores"
OUT = RESULTS / "llm_heads"
SIZES, RHO, N_RANDOM, N_HEADS = (20, 30, 40), 0.10, 5, 32
BIG = (60, 80, "Nt")  # larger sets; Nt = size of the fold's full CC set
CC_SIZES = (20, 30, 40, 60, 80)
HALF = {1: "A", 2: "B"}
R = {"jhead": "J", "logithead": "L"}
COND = {"all": "all", "batched": "bat"}


def ordered(heads: list[dict], remove: bool) -> list[dict]:
    keep = [h for h in heads if not (remove and h["answer_share"] >= RHO)]
    return sorted(keep, key=lambda h: (-h["score"], int(h["id"][1:])))


def as_set(heads: list[dict]) -> list[list[int]]:
    return sorted([h["layer"], h["head"]] for h in heads)


def random_draws(base: list[list[int]]) -> list[list[list[int]]]:
    inside = {tuple(h) for h in base}
    per_layer = Counter(l for l, _ in inside)
    draws = []
    for i in range(N_RANDOM):
        rng = np.random.default_rng(i)
        pick = []
        for layer in sorted(per_layer):
            pool = [(layer, x) for x in range(N_HEADS) if (layer, x) not in inside]
            pick += [list(pool[int(j)]) for j in rng.choice(len(pool), per_layer[layer], replace=False)]
        draws.append(sorted(pick))
    return draws


def cc_ranked(fold: int) -> dict:
    """E4's CC ranking of a fold cut at 30, 60 and 80 heads (``heldout_sets.py`` with more SIZES)."""
    import heldout_sets

    heldout_sets.SIZES = CC_SIZES
    with tempfile.TemporaryDirectory() as tmp:
        argv = sys.argv
        sys.argv = ["heldout_sets.py", "--choose", HALF[fold], "--out", tmp]
        try:
            heldout_sets.main()
        finally:
            sys.argv = argv
        sets = json.loads((Path(tmp) / "heads.json").read_text())["simple"]
    e4 = json.loads((RESULTS / "heldout" / f"fold{fold}" / "heads.json").read_text())["simple"]
    for task, v in sets.items():
        for name in ("cc20", "cc40"):
            mine = next(hs for _, n, hs in v["sets"] if n == name)
            theirs = next(hs for _, n, hs in e4[task]["sets"] if n == name)
            assert mine == theirs, f"fold {fold} {task} {name}: recomputed CC set differs from E4's"
    print(f"fold {fold}: recomputed CC-20 and CC-40 equal E4's on all tasks", flush=True)
    return {t: {n: next(hs for _, name, hs in v["sets"] if name == f"cc{n}") for n in (30, 60, 80)}
            for t, v in sets.items()}


def fv_ranked(fold: int) -> dict:
    """The first 30, 60 and 80 heads by the fold's function-vector score."""
    from heldout_fv_sets import KINDS, top

    from lth import simple

    cie = np.load(RESULTS / "heldout" / f"fold{fold}" / "fv" / "cie.npz")
    sets = {}
    for t in simple.TASKS:
        score = np.mean([cie[f"{t}/{k}"] for k in KINDS], axis=0)
        sets[t] = {n: top(score, n) for n in (30, 60, 80)}
        e4 = json.loads((RESULTS / "heldout" / f"fold{fold}" / "heads.json").read_text())["simple"][t]["sets"]
        for n in (20, 40):  # the stored FV sets must come out of the same ranking
            assert top(score, n) == next(hs for _, name, hs in e4 if name == f"fv{n}"), (fold, t, n)
    return sets


def mean_ranks(values) -> np.ndarray:
    """Ranks from 1, tied values at the mean of their ranks."""
    _, inverse, counts = np.unique(np.asarray(values, float), return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + 1 + ends) / 2)[inverse]


def spearman(a: dict, b: dict) -> float:
    """Spearman correlation of two scorings over all heads, tied scores at their mean rank."""
    ids = sorted(a)
    return float(np.corrcoef(mean_ranks([a[i] for i in ids]), mean_ranks([b[i] for i in ids]))[0, 1])


def jaccard(a, b) -> float:
    a, b = {tuple(h) for h in a}, {tuple(h) for h in b}
    return len(a & b) / len(a | b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()

    agreement, review = [], {t: [] for t in args.tasks}
    # sizes N_t, and which larger sizes hold only heads scored above 0 (per fold, with/without removal)
    n_t, fits = {}, {}
    for task in args.tasks:
        for fold in (1, 2):
            e4 = json.loads((RESULTS / "heldout" / f"fold{fold}" / "heads.json").read_text())["simple"][task]["sets"]
            n_t[(task, fold)] = len(next(hs for _, n, hs in e4 if n == "cc"))
            for readout in R:
                for condition in COND:
                    f = SCORES / f"{task}_{HALF[fold]}_{readout}_{condition}.json"
                    if not f.exists():
                        continue
                    heads = json.loads(f.read_text())["heads"]
                    for remove in (True, False):
                        pos = sum(h["score"] > 0 for h in ordered(heads, remove))
                        fits[(task, readout, condition, fold, remove)] = {
                            size: (n_t[(task, fold)] if size == "Nt" else size) <= pos for size in BIG}
    for fold in (1, 2):
        half = HALF[fold]
        cc, fv = cc_ranked(fold), fv_ranked(fold)
        path = OUT / f"fold{fold}" / "heads.json"
        out = json.loads(path.read_text())["simple"] if path.exists() else {}
        for task in args.tasks:
            sets, scored = [], {}
            for readout in R:
                for condition in COND:
                    f = SCORES / f"{task}_{half}_{readout}_{condition}.json"
                    if not f.exists():
                        print(f"missing {f.name}", flush=True)
                        continue
                    heads = json.loads(f.read_text())["heads"]
                    scored[(readout, condition)] = heads
                    removed, raw = ordered(heads, True), ordered(heads, False)
                    name = f"{COND[condition]}{R[readout]}"
                    for n in SIZES:
                        main_set = as_set(removed[:n])
                        sets.append(["rule", f"{name}{n}", main_set])
                        sets.append(["rule", f"{name}{n}raw", as_set(raw[:n])])
                        sets += [["rule", f"random_{name}{n}_{i}", d] for i, d in enumerate(random_draws(main_set))]
                    review[task].append((fold, readout, condition, heads, removed, raw))
            e4 = json.loads((RESULTS / "heldout" / f"fold{fold}" / "heads.json").read_text())["simple"][task]["sets"]
            sets += [["rule", "cc30", cc[task][30]], ["rule", "fv30", fv[task][30]],
                     ["rule", "fv20", next(hs for _, n, hs in e4 if n == "fv20")]]  # check: must give E4's rows
            # larger sets, after the first patching; existing sets above are unchanged
            for readout in R:
                for condition in COND:
                    if (readout, condition) not in scored:
                        continue
                    name = f"{COND[condition]}{R[readout]}"
                    for size in BIG:
                        if not all(fits[(task, readout, condition, f, remove)].get(size, False)
                                   for f in (1, 2) for remove in (True,)):
                            continue
                        n = n_t[(task, fold)] if size == "Nt" else size
                        removed = ordered(scored[(readout, condition)], True)
                        raw = ordered(scored[(readout, condition)], False)
                        main_set = as_set(removed[:n])
                        sets.append(["rule", f"{name}{size}", main_set])
                        if all(fits[(task, readout, condition, f, False)].get(size, False) for f in (1, 2)):
                            sets.append(["rule", f"{name}{size}raw", as_set(raw[:n])])
                        sets += [["rule", f"random_{name}{size}_{i}", d]
                                 for i, d in enumerate(random_draws(main_set))]
            sets += [["rule", f"cc{n}", cc[task][n]] for n in (60, 80)]
            sets += [["rule", f"fv{n}", fv[task][n]] for n in (60, 80)]
            out[task] = {"sets": sets}
            # agreement: repeat, readouts, conditions
            named = {s[1]: s[2] for s in sets}
            for readout in R:
                rep = SCORES / f"{task}_{half}_{readout}_all_r1.json"
                if rep.exists() and (readout, "all") in scored:
                    a = {h["id"]: h["score"] for h in scored[(readout, "all")]}
                    b_heads = json.loads(rep.read_text())["heads"]
                    b = {h["id"]: h["score"] for h in b_heads}
                    second = ordered(b_heads, True)
                    for n in SIZES:
                        agreement.append(dict(task=task, fold=fold, pair=f"repeat all {R[readout]}", n=n,
                                              jaccard=round(jaccard(named[f"all{R[readout]}{n}"],
                                                                    as_set(second[:n])), 3),
                                              spearman=round(spearman(a, b), 3)))
            pairs = [("allJ", "allL", ("jhead", "all"), ("logithead", "all")),
                     ("batJ", "batL", ("jhead", "batched"), ("logithead", "batched")),
                     ("allJ", "batJ", ("jhead", "all"), ("jhead", "batched")),
                     ("allL", "batL", ("logithead", "all"), ("logithead", "batched"))]
            for x, y, kx, ky in pairs:
                if kx in scored and ky in scored:
                    a = {h["id"]: h["score"] for h in scored[kx]}
                    b = {h["id"]: h["score"] for h in scored[ky]}
                    for n in SIZES:
                        agreement.append(dict(task=task, fold=fold, pair=f"{x} vs {y}", n=n,
                                              jaccard=round(jaccard(named[f"{x}{n}"], named[f"{y}{n}"]), 3),
                                              spearman=round(spearman(a, b), 3)))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"simple": out}, indent=1))
        print(f"wrote {path}", flush=True)
    if (OUT / "agreement.csv").exists():  # keep the rows of the tasks not given here
        with (OUT / "agreement.csv").open(newline="") as f:
            agreement = [r for r in csv.DictReader(f) if r["task"] not in args.tasks] + agreement
    if agreement:
        with (OUT / "agreement.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agreement[0]))
            w.writeheader()
            w.writerows(agreement)
    for task, entries in review.items():
        write_review(task, entries, [a for a in agreement if a["task"] == task])


def write_review(task: str, entries, agreement) -> None:
    tables = {}
    lines = [f"# E5 scores, {task}", "",
             "Heads by score (ties: anonymous id). `removed` = answer share >= 0.10, left out of the sets. "
             "Tokens: the LLM's supporting tokens (score >= 50), then the table's top 8.", ""]
    for fold, readout, condition, heads, removed, raw in entries:
        key = (task, HALF[fold], readout)
        if key not in tables:
            tables[key] = json.loads((DATA / "llm_heads" / "tables" / f"{task}_{HALF[fold]}_{readout}.json").read_text())
        rows = tables[key]["rows"]
        scores = np.array([h["score"] for h in heads])
        flag_a = sum(h["answer"] for h in heads)
        share_a = sum(h["answer_share"] >= RHO for h in heads)
        both = sum(h["answer"] and h["answer_share"] >= RHO for h in heads)
        lines += [f"## Fold {fold} (half {HALF[fold]}), {readout}, {condition}", "",
                  f"Scores: >= 80: {(scores >= 80).sum()}, 50-79: {((scores >= 50) & (scores < 80)).sum()}, "
                  f"20-49: {((scores >= 20) & (scores < 50)).sum()}, < 20: {(scores < 20).sum()}. "
                  f"Answer flag: {flag_a} heads; answer share >= 0.10: {share_a}; both: {both}. "
                  f"Removed from the first 40 by score: {sum(h['answer_share'] >= RHO for h in raw[:40])}.", "",
                  "| rank | head | id | score | answer flag | input flag | answer share | input share | LLM tokens | why | table top 8 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for i, h in enumerate(raw[:50], 1):
            mark = " (removed)" if h["answer_share"] >= RHO else ""
            top8 = ", ".join(json.dumps(s, ensure_ascii=False) + f" {v:.2f}" for _, s, v in rows[h["id"]]["tokens"][:8])
            llm = ", ".join(json.dumps(t, ensure_ascii=False) for t in (h["tokens"] or []))
            lines.append(f"| {i} | L{h['layer']}H{h['head']}{mark} | {h['id']} | {h['score']} | {h['answer']} | "
                         f"{h['input']} | {h['answer_share']:.2f} | {h['input_share']:.2f} | {llm} | "
                         f"{(h['why'] or '').replace('|', '/')} | {top8.replace('|', '/')} |")
        lines.append("")
    if agreement:
        lines += ["## Agreement", "", "| fold | pair | N | Jaccard | Spearman (scores) |", "|---|---|---|---|---|"]
        lines += [f"| {a['fold']} | {a['pair']} | {a['n']} | {a['jaccard']} | {a['spearman']} |" for a in agreement]
    path = DATA / "review" / f"llm_heads_{task}.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
