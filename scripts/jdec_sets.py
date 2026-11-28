"""J-T sets: the J decomposition on the task-word segment, shared, 20 and 40 heads (user 2026-11-28), against
LRP-T. No GPU.

LRP-T (``scripts/lrp_sets.py``) scores a head by ``sum_{j in T} max(0, a[n, j] g~^T v_j)``, ``g~`` the AttnLRP
gradient of the answer's first-token logit at the head's output. J-T keeps everything but the direction: ``g~`` is
replaced by the fitted Jacobian's pullback of the task-word list, ``J_{l,h}^T q_list`` (``scripts/rebuild_jdec.py``,
rows ``list``): the part of the head's J-head score of the list that it reads from the task-word positions. Same pool
prompts (each half's identification pool), same segment (``e20_bai_attention.categorise`` with the fold's list:
the first token and the query left out), terms set to zero where negative, summed over the segment, averaged over
the pool prompts where the segment is not empty; shared = within-task ranks averaged over the 14 tasks, top N
(``rebuild_sets`` ties). Patched (``VARIANTS``): ``jT``, with ``q_list`` as it is; ``jA`` (user, after ``jT``), the
answer target on the same segment (``J_{l,h}^T q_y``, ``q_y`` the scaled unembedding row of the answer's first token,
LRP-T's target): LRP-T with the averaged Jacobian in place of the backward pass.

Not patched, for reading (``READ_ONLY``): ``jTc``, ``q_list`` with the mean scaled unembedding row removed
(``list - mean``: a head writing along it raises every logit alike; it shares 16/20 and 37-38/40 heads with ``jT``).

The random sets are seeded by name, so adding a variant leaves the others' entries unchanged.

Checks before anything is written: (a) the segment-score code here, on the stored AttnLRP rows, reproduces every
task's LRP-T score in ``lrp_features.npz`` exactly; (b) ranks averaged over tasks give the stored
``shared_lrpT20`` and ``shared_lrpT40``.

Writes ``results/rebuild/jdec/fold<f>/heads.json`` (each set with ``N_RANDOM`` random sets of the same layer counts,
key ``jdec/<fold>/shared/<name>``, listed under every task) and ``results/rebuild/jdec/jdec_sets.json`` (heads in
common with the other shared rules of the same size, fold-to-fold, median layers, and per task the Spearman
correlation of the scores over all heads with LRP-T's). Patched by ``scripts/jdec_patch.sh``.

    .venv/bin/python scripts/jdec_sets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS, simple  # noqa: E402
from rebuild_sets import FOLDS, N_RANDOM, OUT as ADD, order_of, pool, random_sets, rankdata, ranks, top  # noqa: E402
from pham_frame_sets import segment_score  # noqa: E402

READINGS = DATA / "readings" / "rebuild"
R = RESULTS / "rebuild"
OUT = R / "jdec"
SIZES = (20, 40)
VARIANTS = ("jT", "jA")
READ_ONLY = ("jTc",)


def stored(path: Path, name: str) -> list[list[int]] | None:
    if not path.exists():
        return None
    sets = json.loads(path.read_text())["simple"][simple.TASKS[0]]["sets"]
    found = [hs for _, nm, hs in sets if nm == name]
    return found[0] if found else None


def others(fold: int, n: int) -> dict[str, list[list[int]] | None]:
    """The other shared rules of size ``n`` on this fold."""
    add = ADD / f"fold{fold}" / "heads.json"
    raw = R / ("pham_frame" if n == 20 else "lrp_raw_n") / f"fold{fold}" / "heads.json"
    return {"lrpT": stored(add, f"shared_lrpT{n}"), "lrpTraw": stored(raw, f"shared_lrpTraw{n}"),
            "T": stored(add, f"shared_T{n}"), "D": stored(add, f"shared_D{n}"), "causal": stored(add, f"shared_causal{n}"),
            "combined": stored(add if n == 40 else R / f"shared_n{n}" / f"fold{fold}" / "heads.json", f"shared_selected{n}")}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def main() -> None:
    from e20_bai_attention import CATS, categorise, task_pattern
    from lth.evaluation import tokenizer

    tok = tokenizer()
    instr = simple.instruction_dict()
    TASK = CATS.index("task")
    info, all_sets = {"folds": {}}, {}
    for fold, half in FOLDS.items():
        lists = json.loads((ADD / f"fold{fold}" / "word_lists.json").read_text())["tasks"]
        lrp_stored = np.load(ADD / f"fold{fold}" / "lrp_features.npz")
        score = {k: {} for k in ("lrpT",) + VARIANTS + READ_ONLY}
        for t in simple.TASKS:
            f_lrp = np.load(READINGS / f"lrp_{half}" / f"{t}.npz")
            f_j = np.load(READINGS / f"jdec_{half}" / f"{t}.npz")
            data, pattern = simple.dataset(t), task_pattern(lists[t]["words"])
            prompts = {w: simple.prompts(t, w)[0] for w in simple.WORDINGS["rule"]}
            keys = pool(half, t)[0]

            def index(f):
                start = np.concatenate([[0], np.cumsum(f["lengths"])[:-1]])
                return {(int(j), int(i)): (int(s), int(n)) for j, i, s, n in zip(f["wordings"], f["items"], start, f["lengths"])}

            w_lrp, w_j = index(f_lrp), index(f_j)
            assert set(w_j) == set(keys) and set(w_lrp) >= set(keys), (fold, t)
            pos = {}
            for j, item in keys:
                w = simple.WORDINGS["rule"][j]
                text = prompts[w][item]
                enc = tok(text, return_offsets_mapping=True)
                assert w_lrp[(j, item)][1] == w_j[(j, item)][1] == len(enc.input_ids), (fold, t, w, item)
                cat, _, _ = categorise(text, instr[t][w[4:]], data[item]["input"], enc.input_ids, enc.offset_mapping, pattern)
                pos[(j, item)] = [i for i in range(len(cat)) if cat[i] == TASK]
            rows_lrp, rows_l, rows_m, rows_a = f_lrp["rows"], f_j["rows_list"], f_j["rows_mean"], f_j["rows_answer"]

            def rows_of(rows, where):
                def get(key):
                    s, n = where[key]
                    return rows[s:s + n].astype(np.float64)
                return get

            score["lrpT"][t] = segment_score(rows_of(rows_lrp, w_lrp), pos.get, keys)
            assert np.array_equal(score["lrpT"][t], lrp_stored[f"{t}/T"]), f"(a) LRP-T not reproduced: fold {fold} {t}"
            get_l, get_m = rows_of(rows_l, w_j), rows_of(rows_m, w_j)
            score["jT"][t] = segment_score(get_l, pos.get, keys)
            score["jTc"][t] = segment_score(lambda key: get_l(key) - get_m(key), pos.get, keys)
            score["jA"][t] = segment_score(rows_of(rows_a, w_j), pos.get, keys)
        print(f"fold {fold}: (a) LRP-T of all 14 tasks reproduced exactly", flush=True)

        def rank_order(sc):
            return order_of(rankdata(np.mean([ranks(sc[t]) for t in simple.TASKS], axis=0), method="average"))

        orders = {k: rank_order(v) for k, v in score.items()}
        ref = {n: others(fold, n) for n in SIZES}
        for n in SIZES:
            assert top(orders["lrpT"], n) == ref[n]["lrpT"], f"(b) shared_lrpT{n} not reproduced: fold {fold}"
        print(f"fold {fold}: (b) shared_lrpT{', '.join(map(str, SIZES))} reproduced exactly", flush=True)

        sets = {f"shared_{k}{n}": top(orders[k], n) for k in VARIANTS + READ_ONLY for n in SIZES}
        all_sets[fold] = sets
        entries = []
        for k in VARIANTS:
            for n in SIZES:
                name = f"shared_{k}{n}"
                entries += [["rule", name, sets[name]]] + [["rule", f"random_{name}_{i}", r]
                                                          for i, r in enumerate(random_sets(sets[name], f"jdec/{fold}/shared/{name}"))]
        assert len(entries) == len(VARIANTS) * len(SIZES) * (N_RANDOM + 1)
        d = OUT / f"fold{fold}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "heads.json").write_text(json.dumps({"simple": {t: {"sets": entries} for t in simple.TASKS}}))

        fi = {"common": {}, "layer_median": {}, "spearman_vs_lrpT": {}}
        for n in SIZES:
            for k in VARIANTS + READ_ONLY:
                mine = set(map(tuple, sets[f"shared_{k}{n}"]))
                cmp = {o: len(mine & set(map(tuple, hs))) for o, hs in ref[n].items() if hs is not None}
                cmp.update({v: len(mine & set(map(tuple, sets[f"shared_{v}{n}"]))) for v in VARIANTS + READ_ONLY if v != k})
                fi["common"][f"shared_{k}{n}"] = cmp
                fi["layer_median"][f"shared_{k}{n}"] = float(np.median([l for l, _ in mine]))
            fi["layer_median"][f"shared_lrpT{n}"] = float(np.median([l for l, _ in ref[n]["lrpT"]]))
        for k in VARIANTS + READ_ONLY:
            fi["spearman_vs_lrpT"][k] = {t: spearman(score[k][t], score["lrpT"][t]) for t in simple.TASKS}
        fi["spearman_jT_vs_jTc"] = {t: spearman(score["jT"][t], score["jTc"][t]) for t in simple.TASKS}
        info["folds"][f"fold{fold}"] = fi
        print(f"fold {fold}: heads in common", flush=True)
        for name, cmp in fi["common"].items():
            print(f"  {name:14s} " + ", ".join(f"{o} {c}" for o, c in cmp.items()), flush=True)
        for k, by_task in fi["spearman_vs_lrpT"].items():
            v = list(by_task.values())
            print(f"  Spearman {k} vs LRP-T over 1024 heads, per task: median {np.median(v):.2f} "
                  f"(range {min(v):.2f} to {max(v):.2f})", flush=True)
        v = list(fi["spearman_jT_vs_jTc"].values())
        print(f"  Spearman jT vs jTc: median {np.median(v):.2f} (range {min(v):.2f} to {max(v):.2f})", flush=True)
    info["fold_to_fold"] = {name: len(set(map(tuple, all_sets[1][name])) & set(map(tuple, all_sets[2][name])))
                            for name in all_sets[1]}
    info["fold_to_fold"].update({f"shared_lrpT{n}": len(set(map(tuple, others(1, n)["lrpT"])) & set(map(tuple, others(2, n)["lrpT"])))
                                 for n in SIZES})
    info["sets"] = {f"fold{f}": s for f, s in all_sets.items()}
    print("fold 1 vs fold 2, heads in common: " + ", ".join(f"{k} {v}" for k, v in info["fold_to_fold"].items()), flush=True)
    (OUT / "jdec_sets.json").write_text(json.dumps(info, indent=1))
    print("written:", OUT, flush=True)


if __name__ == "__main__":
    main()
