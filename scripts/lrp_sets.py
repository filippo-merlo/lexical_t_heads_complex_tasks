"""T and LRP-T sets (Pham et al. 2026 mapped onto paper/head-sets-simple-design.tex Sec. 5.3), added to each fold's
``results/rebuild/additions/fold<f>/heads.json``. No GPU.

- **T**: the task-word feature of Sec. 5.3 alone, the attention mass of the last token on the task-word segment,
  averaged over the pool prompts where the segment is not empty (``features.npz``).
- **LRP-T**: the same with each attention weight replaced by its AttnLRP relevance for the logit of the answer's
  first token, set to zero where negative, as Pham et al. score heads: ``sum_{j in T} max(0, A[n, j] *
  dlogit/dA[n, j])`` (``scripts/rebuild_lrp.py``), averaged over the same prompts; relevance not normalised.

Both rules exactly as ``scripts/rebuild_sets.py`` ranks every rule, through its own functions: ``T<N>`` and
``lrpT<N>``, the top N of the task's ranking; ``shared_T<N>`` and ``shared_lrpT<N>``, the top N by within-task
ranks averaged over tasks; N in 10..120; three random sets per set, ``random_<name>_<i>``, with as many heads per
layer (same seeding rule).

Self-check before anything is written: the segment scores of the stored attention rows, computed by this
script's code, equal every attention feature in ``features.npz`` (B and every segment), bit for bit. Existing
T and LRP-T sets are replaced; the script asserts that no other set changes.

Also writes ``results/rebuild/additions/fold<f>/lrp_features.npz``: per task, the LRP version of every attention
feature (``<task>/<code>``; non-first = the relevance on every position but the first), the whole-row relevance
(``<task>/whole``, clamped per key; ``<task>/whole_signed``) and the target ranks (``<task>/rank``); and
``lrp_sets.json``: per set, the median layer and the heads in common with T, Causal, Causal (zero-shot) and
Selected of the same scope and size.

    .venv/bin/python scripts/lrp_sets.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS, simple  # noqa: E402
from rebuild_sets import (FEATURES, FOLDS, N_ALL, OUT, SIZES, order_of, pool, random_sets,  # noqa: E402
                          rankdata, ranks, top)

READINGS = DATA / "readings" / "rebuild"
RULES = ("T", "lrpT")
MINE = re.compile(r"^(random_)?(shared_)?(T|lrpT)\d+(_\d+)?$")
SEGMENTED = [c for c, (_, seg) in FEATURES.items() if seg is not None]


def segment_scores(half: str, task: str, keys, words: list[str], tok, source: str):
    """Per segment, the row values on it summed per prompt and averaged over the pool prompts where it is not empty.

    ``source`` ``attention``: the stored attention rows, as ``rebuild_sets.segment_masses``; ``lrp``: the AttnLRP
    rows clamped at zero per key. Also, for ``lrp``, the whole row summed (clamped and signed) averaged over all
    pool prompts, and the target ranks.
    """
    from e20_bai_attention import CATS, categorise, task_pattern

    f = np.load(READINGS / f"{source}_{half}" / f"{task}.npz")
    start = np.concatenate([[0], np.cumsum(f["lengths"])[:-1]])
    where = {(int(j), int(i)): (int(s), int(n), k)
             for k, (j, i, s, n) in enumerate(zip(f["wordings"], f["items"], start, f["lengths"]))}
    data, pattern = simple.dataset(task), task_pattern(words)
    prompts = {w: simple.prompts(task, w)[0] for w in simple.WORDINGS["rule"]}
    total, count = np.zeros((len(CATS), N_ALL)), np.zeros(len(CATS))
    whole, whole_signed, rank = np.zeros(N_ALL), np.zeros(N_ALL), []
    for j, item in keys:
        w = simple.WORDINGS["rule"][j]
        text = prompts[w][item]
        enc = tok(text, return_offsets_mapping=True)
        s, n, k = where[(j, item)]
        assert n == len(enc.input_ids), (source, task, w, item)
        cat, _, _ = categorise(text, simple.instruction_dict()[task][w[4:]], data[item]["input"],
                               enc.input_ids, enc.offset_mapping, pattern)
        rows = f["rows"][s:s + n].astype(np.float64)  # [T, 1024]
        if source == "lrp":
            whole_signed += rows.sum(0)
            rows = np.maximum(rows, 0.0)
            whole += rows.sum(0)
            rank.append(int(f["rank"][k]))
        for c in range(len(CATS)):
            m = cat == c
            if m.any():
                total[c] += rows[m].sum(0)
                count[c] += 1
    masses = {c: total[k] / count[k] if count[k] else np.zeros(N_ALL) for k, c in enumerate(CATS)}
    extra = {}
    if source == "lrp":
        extra = {"whole": whole / len(keys), "whole_signed": whole_signed / len(keys), "rank": np.array(rank)}
    return masses, extra


def features(masses, source: str) -> dict[str, np.ndarray]:
    """Feature codes of Sec. 5.3 from segment scores: B is non-first (attention: one minus the first token's
    mass, as ``rebuild_sets``; LRP: the clamped relevance on every other position, i.e. the sum of the others)."""
    from e20_bai_attention import CATS

    out = {code: masses[FEATURES[code][1]] for code in SEGMENTED}
    out["B"] = 1.0 - masses["bos"] if source == "attention" else sum(masses[c] for c in CATS if c != "bos")
    return out


def selected_sets(fold: int) -> dict[str, dict[str, set]]:
    """Selected per task (heads.json) and shared Selected at each size (its own screen where there is one)."""
    out = {}
    for n in SIZES:
        d = RESULTS / "rebuild" / f"shared_n{n}" / f"fold{fold}" / "heads.json"
        if d.exists():
            h = json.loads(d.read_text())["simple"]
            out.update({(t, nm): set(map(tuple, hs)) for t in simple.TASKS for _, nm, hs in h[t]["sets"]
                        if nm == f"shared_selected{n}"})
    return out


def main() -> None:
    from lth.evaluation import tokenizer

    tok = tokenizer()
    for fold, half in FOLDS.items():
        out = OUT / f"fold{fold}"
        lists = json.loads((out / "word_lists.json").read_text())["tasks"]
        stored_features = np.load(out / "features.npz")
        feats, lrp_feats, extras = {}, {}, {}
        for t in simple.TASKS:
            keys = pool(half, t)[0]
            att, _ = segment_scores(half, t, keys, lists[t]["words"], tok, "attention")
            feats[t] = features(att, "attention")
            for code, v in feats[t].items():  # self-check against rebuild_sets.py's features
                assert np.array_equal(v, stored_features[f"{t}/{code}"]), (fold, t, code)
            lrp, extras[t] = segment_scores(half, t, keys, lists[t]["words"], tok, "lrp")
            lrp_feats[t] = features(lrp, "lrp")
        print(f"fold {fold}: attention features of all {len(simple.TASKS)} tasks reproduced exactly "
              f"({', '.join(sorted(feats[simple.TASKS[0]]))})", flush=True)

        score = {"T": {t: feats[t]["T"] for t in simple.TASKS}, "lrpT": {t: lrp_feats[t]["T"] for t in simple.TASKS}}
        per_task, shared = {t: [] for t in simple.TASKS}, []

        def add(target: list, rule: str, n, heads, key: str):
            label = f"{rule}{n}"
            target.append(["rule", label, heads])
            target += [["rule", f"random_{label}_{i}", r] for i, r in enumerate(random_sets(heads, f"{fold}/{key}/{label}"))]

        orders = {}
        for rule in RULES:
            single = {t: ranks(score[rule][t]) for t in simple.TASKS}
            shared_order = order_of(rankdata(np.mean([single[t] for t in simple.TASKS], axis=0), method="average"))
            orders[rule] = ({t: order_of(single[t]) for t in simple.TASKS}, shared_order)
            for t in simple.TASKS:
                for n in SIZES:
                    add(per_task[t], rule, n, top(orders[rule][0][t], n), t)
            for n in SIZES:
                add(shared, f"shared_{rule}", n, top(shared_order, n), "shared")

        heads = json.loads((out / "heads.json").read_text())
        before = {t: [s for s in heads["simple"][t]["sets"] if not MINE.match(s[1])] for t in simple.TASKS}
        for t in simple.TASKS:
            heads["simple"][t]["sets"] = before[t] + per_task[t] + shared
            assert all(MINE.match(nm) for _, nm, _ in per_task[t] + shared), t
        partial = out / "heads.json.partial"  # replaced at once: patch runs may be reading it
        partial.write_text(json.dumps(heads))
        partial.replace(out / "heads.json")
        again = json.loads((out / "heads.json").read_text())["simple"]
        for t in simple.TASKS:
            assert [s for s in again[t]["sets"] if not MINE.match(s[1])] == before[t], (fold, t)

        np.savez_compressed(out / "lrp_features.npz",
                            **{f"{t}/{c}": v for t in simple.TASKS for c, v in lrp_feats[t].items()},
                            **{f"{t}/{k}": v for t in simple.TASKS for k, v in extras[t].items()})

        # Overlaps with the other rules of the same scope and size.
        stored = {(t, nm): set(map(tuple, hs)) for t in simple.TASKS for _, nm, hs in again[t]["sets"]}
        stored.update(selected_sets(fold))
        info = {"fold": fold, "choosing_half": half, "sets": []}
        for n in SIZES:
            for scope, prefix, tasks in (("per_task", "", simple.TASKS), ("shared", "shared_", simple.TASKS[:1])):
                for t in tasks:
                    mine = stored[(t, f"{prefix}lrpT{n}")]
                    row = {"scope": scope, "task": t if scope == "per_task" else "all", "n": n,
                           "layer_median_lrpT": float(np.median([l for l, _ in mine])),
                           "layer_median_T": float(np.median([l for l, _ in stored[(t, f"{prefix}T{n}")]]))}
                    for other in ("T", "causal", "causalzs", "selected"):
                        ref = stored.get((t, f"{prefix}{other}{n}"))
                        row[f"common_{other}"] = len(mine & ref) if ref is not None else None
                    info["sets"].append(row)
        info["target_rank"] = {t: {"top1": float((extras[t]["rank"] == 1).mean()),
                                   "top5": float((extras[t]["rank"] <= 5).mean())} for t in simple.TASKS}
        (out / "lrp_sets.json").write_text(json.dumps(info, indent=1))
        n_sets = sum(len(per_task[t]) for t in simple.TASKS) + len(shared)
        print(f"fold {fold}: {n_sets} T / LRP-T sets written (random sets included); no other set changed", flush=True)


if __name__ == "__main__":
    main()
