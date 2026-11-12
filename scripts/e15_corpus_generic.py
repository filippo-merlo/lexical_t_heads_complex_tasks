"""E15 (no GPU): shared Causal and DBT minus generic heads, with generic heads defined against a corpus.

E14's generic heads (``e14_sets.py``) are the heads of the fold's Davidson top 30 whose J-head decoding is
alike across the 14 tasks. Here a head is generic when its J-head decoding on the task prompts is alike its
decoding on unrelated text:

- corpus score of head h on task t: cosine between h's J-head top-10 token counts (each token once per
  prompt) at the last token of the task's choosing-half prompts, all 35 rule templates (E14's readings),
  and at the last token of the 1,000 WikiText-103 passages (``data/readings/neutral/wikitext.npz``);
- shared corpus score: mean over the 14 tasks;
- generic heads: two-group split (least within-group sum of squares) of the fold's Davidson top 30 by the
  shared corpus score, none if more than 15 (E13/E14's rule, only the score changed).

Everything else is E14's a35 setting: Davidson et al.'s causal score on the 14 question-answer templates,
D/B/T on the 35 templates, shared rank rules. Sets (shared, N = 40, 80):

- ``a35_shared_causal{N}_nogen_corpus``, ``a35_shared_DBT{N}_nogen_corpus``: minus the corpus generic heads;
- ``a35_shared_DBT{N}_nogen``: minus E14's (across-task) generic heads; E14 did not run it;
- ``a35_shared_causal40_nogen``: E14's set, re-patched as a reproduction check.

Writes ``results/e15_corpus_generic/fold{f}/heads.json`` (``scripts/patch.py`` set-list layout) and
``generic.json`` (both scores of the top 30, both lists).

    .venv/bin/python scripts/e15_corpus_generic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e12_generic_halves as e12  # noqa: E402
from e13_sets import ATT, JH, KINDS, N_HEADS, SIGN, ranks, top, two_group_cut  # noqa: E402
from lth import DATA, RESULTS, simple  # noqa: E402
from lth.store import load  # noqa: E402

E14 = RESULTS / "e14_wordings"
OUT = RESULTS / "e15_corpus_generic"
HALF = {1: "A", 2: "B"}
SIZES = (40, 80)
TASKS = list(simple.TASKS)


def name(x: int) -> str:
    return f"L{x // N_HEADS}H{x % N_HEADS}"


def corpus_scores(half: str, heads) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
    root = DATA / "readings" / "heldout" / half / "simple"
    wiki = np.load(DATA / "readings" / "neutral" / "wikitext.npz")["top"][:, JH]  # 1000, 1024, 10
    top_ = {}
    for t in TASKS:
        parts = [load(root / t.replace("/", "__"), w)["top"] for w in simple.ALL_RULE]
        top_[t] = np.concatenate([p[:, JH] for p in parts if len(p)])
    per = {}
    for s in heads:
        w = e12.counts(wiki[:, s])
        per[s] = {t: e12.cos(e12.counts(top_[t][:, s]), w) for t in TASKS}
    return {s: float(np.mean(list(v.values()))) for s, v in per.items()}, per


def main() -> None:
    for fold, half in HALF.items():
        src = E14 / f"fold{fold}"
        dsc = np.load(src / "d_scores.npz")
        att_dir = DATA / "readings" / "heldout" / f"attention14_{half}"
        att = {t: np.load(att_dir / f"{t}.npy").astype(np.float64).mean(0) for t in TASKS}
        score = {t: {"D": dsc[t], "B": att[t][:, ATT["B"]], "T": att[t][:, ATT["T"]]} for t in TASKS}
        cie = np.load(src / "fv" / "cie.npz")
        causal = {t: np.mean([cie[f"{t}/{k}"] for k in KINDS], axis=0).ravel() for t in TASKS}
        shared_causal_order = np.argsort(-np.mean([causal[t] for t in TASKS], axis=0), kind="stable")
        within = {f: np.stack([ranks(SIGN[f] * score[t][f]) for t in TASKS]).mean(0) for f in "DBT"}
        combined = {f: ranks(-within[f]) for f in "DBT"}
        shared_dbt_order = np.argsort(np.mean([combined[f] for f in "DBT"], axis=0), kind="stable")

        fv30 = [int(i) for i in shared_causal_order[:30]]
        g14 = json.loads((src / "generic.json").read_text())
        cross_gen = {int(n[1:].split("H")[0]) * N_HEADS + int(n.split("H")[1]) for n in g14["generic"]}
        assert {name(x) for x in fv30} == {n for n, _ in g14["fv30_by_score"]}, "E14's Davidson top 30 not reproduced"

        shared, per = corpus_scores(half, fv30)
        by_score = sorted(fv30, key=lambda x: -shared[x])
        k = two_group_cut([shared[x] for x in by_score])
        corpus_gen = set(by_score[:k]) if k <= 15 else set()

        # Reproduce E14's shared sets before building new ones.
        e14_sets = {m: hs for _, m, hs in json.loads((src / "heads.json").read_text())["simple"]["antonym"]["sets"]}
        for n in SIZES:
            assert top(shared_dbt_order, n) == e14_sets[f"a35_shared_DBT{n}"], n
            assert top(shared_causal_order, n) == e14_sets[f"a35_shared_causal{n}"], n
            assert top(shared_causal_order, n, cross_gen) == e14_sets[f"a35_shared_causal{n}_nogen"], n

        sets = [["rule", "a35_shared_causal40_nogen", top(shared_causal_order, 40, cross_gen)]]
        for n in SIZES:
            sets.append(["rule", f"a35_shared_causal{n}_nogen_corpus", top(shared_causal_order, n, corpus_gen)])
            sets.append(["rule", f"a35_shared_DBT{n}_nogen_corpus", top(shared_dbt_order, n, corpus_gen)])
            sets.append(["rule", f"a35_shared_DBT{n}_nogen", top(shared_dbt_order, n, cross_gen)])
        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "heads.json").write_text(json.dumps({"simple": {t: {"sets": sets} for t in TASKS}}, indent=1))
        cross = dict(g14["fv30_by_score"])
        (out / "generic.json").write_text(json.dumps(dict(
            fold=fold, choosing_half=half, templates=list(simple.ALL_RULE),
            rule="two-group split of the fold's Davidson top 30 by the shared corpus score (mean over tasks of the "
                 "cosine of J-head top-10 token counts on the task prompts vs WikiText); none if more than 15",
            fv30_by_corpus=[[name(x), round(shared[x], 4), cross[name(x)]] for x in by_score],
            corpus_generic=sorted(name(x) for x in corpus_gen), cross_generic=sorted(g14["generic"]),
            per_task={name(x): {t: round(v, 4) for t, v in per[x].items()} for x in fv30}), indent=1))
        dbt = {n: {tuple(h) for h in top(shared_dbt_order, n)} for n in SIZES}
        cg = {(x // N_HEADS, x % N_HEADS) for x in corpus_gen}
        xg = {(x // N_HEADS, x % N_HEADS) for x in cross_gen}
        print(f"fold {fold}: corpus generic {len(corpus_gen)} {sorted(name(x) for x in corpus_gen)}")
        print(f"         cross  generic {len(cross_gen)} {sorted(g14['generic'])}")
        print(f"         both {sorted(name(x) for x in corpus_gen & cross_gen)}")
        for n in SIZES:
            print(f"         DBT{n} holds {len(dbt[n] & cg)} corpus-generic, {len(dbt[n] & xg)} cross-generic heads")


if __name__ == "__main__":
    main()
