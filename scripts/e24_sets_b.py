"""E24 step B (no GPU): the filter sets built again from example prompts, and their overlap with the
rule-built sets.

For each fold, on the choosing half's correctly answered 5-shot prompts (``scripts/heldout_readings.py
--wordings shot5``, up to 100 per task; ``scripts/e24_attention.py``), the E13/E21 filter scores:

- **D**: the task-word list built by E13's rule (``scripts/e13_lists.py``: word rule, the same 50
  generic WikiText words removed, the 30 words with the largest share) from the 5-shot J-head
  readings, and each head's share of 5-shot prompts whose top 10 holds a word of that list;
- **B**: mean attention on BOS (lower ranks higher);
- **M**: mean attention on the demonstrations' ":" and ";" (Bai et al.'s template tokens);
- **S**: mean attention on punctuation and stopwords.

T and N have no counterpart without an instruction and are not built (user, 2026-11-16).
Combinations D, B, DB, DBMS, per task and shared, at N = 20 to 120, combined exactly as E13/E22
(per task: within-task ranks averaged over the combination; shared: within-task ranks averaged
over tasks, ranked again, averaged over the combination); ties to the lower layer, then head.

The rule-built sets of the same combinations are rebuilt from E13's and E20's scores with the same
code and checked against every patched set of E13 (D, B, DB), E22 (DBMS40) and E23 (DBMS at the
sizes it ran).

Overlap of the rule- and example-built set of each fold, task, scope, combination and N: the
shared heads, against two chance levels, uniform (N^2 / 1024) and layer-matched (sum over layers
of the two sets' head counts multiplied, / 32: the expected overlap of two sets drawn with these
layer counts).

Writes ``results/e24_cross_style/b/fold{f}/word_lists.json``, ``d_scores.npz``, ``heads.json``
(``scripts/patch.py`` layout: every example-built set under both styles, so that ``--same-style``
patches it with rule and with example means; see ``PATCH_SIZES``), and
``results/e24_cross_style/b/overlap.csv``, ``lists.csv``. Step C (the other half of the 2 x 2):
``results/e24_cross_style/c/fold{f}/heads.json``, the rule-built sets of the same combinations and
sizes under the example style (patched with the example mean), and per-task DBMS100 under the rule
style (the one rule-built set at these sizes with no rule-mean run in E13/E22/E23).

    .venv/bin/python scripts/e24_sets_b.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS, simple  # noqa: E402

from e13_lists import JH, N_GENERIC, N_WORDS, shares, token_words  # noqa: E402
from e13_sets import N_HEADS, N_RANDOM, SIZES, ranks, top  # noqa: E402
from e20_bai_attention import CATS  # noqa: E402
from e11_sets import layer_matched  # noqa: E402

OUT = RESULTS / "e24_cross_style" / "b"
HALF = {1: "A", 2: "B"}
COMBOS = ("D", "B", "DB", "DBMS")
PARTS = {"B": ("bos",), "M": ("template",), "S": ("punct", "function")}
SIGN = {"D": 1, "B": -1, "M": 1, "S": 1}
SCOPES = {"task": "", "shared": "shared_"}
PATCH_SIZES = (40, 100)  # the example-built sets patched (rule and example means), with random draws


def example_scores(half: str, tmap, words, generic) -> tuple[dict, dict, dict]:
    from lth.store import load

    lists, score = {}, {}
    for t in simple.TASKS:
        top10 = load(DATA / "readings" / "heldout" / half / "simple" / t, "shot5")["top"][:, JH]
        s = shares(top10, tmap, len(words))
        order = [i for i in np.argsort(-s, kind="stable") if s[i] > 0 and words[i] not in generic][:N_WORDS]
        lists[t] = {"words": [words[i] for i in order], "shares": [round(float(s[i]), 6) for i in order],
                    "n_prompts": int(top10.shape[0])}
        member = np.zeros(len(words) + 1, dtype=bool)
        member[order] = True
        att = np.load(DATA / "readings" / "heldout" / f"attention24_{half}" / f"{t}.npy").astype(np.float64)
        assert att.shape[0] == top10.shape[0], (t, att.shape, top10.shape)
        att = att.mean(0)
        score[t] = {"D": member[tmap[top10.astype(np.int64)]].any(-1).mean(0),
                    **{f: att[:, [CATS.index(c) for c in cs]].sum(1) for f, cs in PARTS.items()}}
    d = {t: score[t]["D"] for t in simple.TASKS}
    return lists, score, d


def rule_scores(fold: int, half: str) -> dict:
    dsc = np.load(RESULTS / "e13_capped" / f"fold{fold}" / "d_scores.npz")
    out = {}
    for t in simple.TASKS:
        att = np.load(DATA / "readings" / "heldout" / f"attention20_{half}" / f"{t}.npy").astype(np.float64).mean(0)
        out[t] = {"D": dsc[t], **{f: att[:, [CATS.index(c) for c in cs]].sum(1) for f, cs in PARTS.items()}}
    return out


def build(score: dict) -> dict:
    """(scope, combo, n) -> {task: heads}."""
    letters = sorted({f for c in COMBOS for f in c})
    within = {f: np.stack([ranks(SIGN[f] * score[t][f]) for t in simple.TASKS]).mean(0) for f in letters}
    combined = {f: ranks(-within[f]) for f in letters}
    sets = {}
    for c in COMBOS:
        shared_order = np.argsort(np.mean([combined[f] for f in c], axis=0), kind="stable")
        for t in simple.TASKS:
            r = {f: ranks(SIGN[f] * score[t][f]) for f in c}
            order = np.argsort(np.mean([r[f] for f in c], axis=0), kind="stable")
            for n in SIZES:
                sets.setdefault(("task", c, n), {})[t] = top(order, n)
                sets.setdefault(("shared", c, n), {})[t] = top(shared_order, n)
    return sets


def patched(fold: int) -> dict:
    out = {}
    for exp in ("e13_capped", "e22_all_combos", "e23_best_sizes"):
        for t, v in json.loads((RESULTS / exp / f"fold{fold}" / "heads.json").read_text())["simple"].items():
            for style, name, hs in v["sets"]:
                if style == "rule":
                    out.setdefault(t, {})[name] = hs
    return out


def main() -> None:
    from lth.evaluation import tokenizer

    tok = tokenizer()
    tmap, words = token_words(tok)
    wiki = shares(np.load(DATA / "readings" / "neutral" / "wikitext.npz")["top"][:, JH], tmap, len(words))
    generic = {words[i] for i in np.argsort(-wiki, kind="stable")[:N_GENERIC]}
    OUT.mkdir(parents=True, exist_ok=True)
    overlap_rows, list_rows = [], []
    for fold, half in HALF.items():
        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        lists, ex_score, d = example_scores(half, tmap, words, generic)
        (out / "word_lists.json").write_text(json.dumps(dict(fold=fold, choosing_half=half, wording="shot5",
                                                              tasks=lists), ensure_ascii=False, indent=1))
        np.savez_compressed(out / "d_scores.npz", **d)
        rule_lists = json.loads((RESULTS / "e13_capped" / f"fold{fold}" / "word_lists.json").read_text())["tasks"]
        for t in simple.TASKS:
            shared_words = set(lists[t]["words"]) & set(rule_lists[t]["words"])
            list_rows.append(dict(fold=fold, task=t, example_prompts=lists[t]["n_prompts"],
                                  shared_words=len(shared_words), example_list=" ".join(lists[t]["words"]),
                                  rule_list=" ".join(rule_lists[t]["words"])))

        rule = build(rule_scores(fold, half))
        ex = build(ex_score)
        # Check: the rule-built sets are the patched ones wherever those exist.
        known = patched(fold)
        checked = 0
        for (scope, c, n), per in rule.items():
            name = f"{SCOPES[scope]}{c}{n}"
            for t, hs in per.items():
                if name in known[t]:
                    assert hs == known[t][name], (fold, t, name)
                    checked += 1
        print(f"fold {fold}: {checked} rule-built sets equal the patched ones")

        for (scope, c, n), per in ex.items():
            for t, hs in per.items():
                rs = rule[scope, c, n][t]
                cr, ce = Counter(l for l, _ in rs), Counter(l for l, _ in hs)
                overlap_rows.append(dict(
                    fold=fold, task=t, scope=scope, combo=c, n=n,
                    shared=len({tuple(h) for h in rs} & {tuple(h) for h in hs}),
                    chance_uniform=n * n / (N_HEADS * 32), chance_layer=sum(cr[l] * ce[l] for l in cr) / N_HEADS,
                    layer_median_rule=float(np.median([l for l, _ in rs])),
                    layer_median_example=float(np.median([l for l, _ in hs]))))

        sets = {t: [] for t in simple.TASKS}
        for scope, prefix in SCOPES.items():
            for c in COMBOS:
                for n in PATCH_SIZES:
                    for t in simple.TASKS:
                        hs = ex[scope, c, n][t]
                        for style in ("rule", "example"):
                            sets[t].append([style, f"ex_{prefix}{c}{n}", hs])
            for n in PATCH_SIZES:  # random draws layer-matched to the example-built DB set
                for i in range(N_RANDOM):
                    for t in simple.TASKS:
                        base = ex[scope, "DB", n][t if scope == "task" else simple.TASKS[0]]
                        draw = layer_matched(base, seed=i if scope == "task" else 100 + i)
                        for style in ("rule", "example"):
                            sets[t].append([style, f"random_ex_{prefix}db{n}_{i}", draw])
        (out / "heads.json").write_text(json.dumps({"simple": {t: {"sets": v} for t, v in sets.items()}}, indent=1))

        c_sets = {t: [] for t in simple.TASKS}
        for scope, prefix in SCOPES.items():
            for c in COMBOS:
                for n in PATCH_SIZES:
                    for t in simple.TASKS:
                        name = f"{prefix}{c}{n}"
                        c_sets[t].append(["example", name, rule[scope, c, n][t]])
                        if name not in known[t]:
                            c_sets[t].append(["rule", name, rule[scope, c, n][t]])
        c_out = OUT.parent / "c" / f"fold{fold}"
        c_out.mkdir(parents=True, exist_ok=True)
        (c_out / "heads.json").write_text(json.dumps({"simple": {t: {"sets": v} for t, v in c_sets.items()}}, indent=1))
        print(f"fold {fold}: step C, {len(c_sets[simple.TASKS[0]])} set-styles per task "
              f"({sorted(s[1] for s in c_sets[simple.TASKS[0]] if s[0] == 'rule')} under the rule style)")

    for name, rows in (("overlap", overlap_rows), ("lists", list_rows)):
        with (OUT / f"{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
