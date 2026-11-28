"""Pham et al.'s own form (shared, 20 heads) with their intensional frame vs LRP-T's task-word segment, in a 2x2 of
frame x way of combining tasks (user 2026-11-28). No GPU.

Pham et al. (2026, "Fast & Faithful Function Vectors", Secs. 3-4) score a head by the positive AttnLRP relevance of
the last position's attention to the tokens of the *intensional frame*, identified by hand per instruction, average
it over a task's prompts, average the raw scores over tasks and keep the top 20 (code: ``extract_heads_lrp.py``,
``eval/utils.py`` ``load_prompt_aie``). Their frames for Llama-3.1-8B-Instruct are the strings of
``formatted_tasks[task]["llama8b"]``, found as case-insensitive substrings of the prompt; a token is in the frame
when it overlaps an occurrence. On Yang et al.'s five templates those strings match 64 of 70 templates; the six gaps
are filled with the one word that defines the task there (user's choice (b)): "antonym" (antonym templates 0, 1, 2,
4; in their own Llama-3.2-3B list), "nation" (park-country template 2), "previous" (prev_item template 3). As for
T, tokens inside the query and the first token are left out.

Everything else is LRP-T's (``scripts/lrp_sets.py``): the stored AttnLRP rows of each half's identification pool
(``data/readings/rebuild/lrp_<half>``, target = the answer's first token), relevance set to zero where negative,
summed over the segment, averaged over the pool prompts where the segment is not empty, not normalised.

Sets (per fold, on its choosing half):
- ``shared_lrpFraw20``: Pham frame, raw scores averaged over the 14 tasks (their form);
- ``shared_lrpFrank20``: Pham frame, within-task ranks averaged (our shared rule);
- ``shared_lrpTraw20``: task-word segment, raw scores averaged;
- the fourth corner, ``shared_lrpT20`` (task-word segment, ranks averaged), is the stored set, already patched.
Each with ``N_RANDOM`` random sets of the same layer counts (``rebuild_sets.random_sets``, key
``pham_frame/<fold>/shared/<name>``), listed under every task.

Checks before anything is written: (a) the segment-score code here, given T's positions, reproduces every task's
LRP-T score in ``lrp_features.npz`` exactly; (b) ranks averaged over tasks give the stored ``shared_lrpT20``;
(c) the frame is non-empty on every pool prompt, and each gap word adds tokens only in its gap templates
(elsewhere Pham et al.'s strings are non-empty and already cover it).

Writes ``results/rebuild/pham_frame/fold<f>/heads.json`` and ``results/rebuild/pham_frame/frames.json`` (per task
and template, the frame tokens and T's tokens of the first pool prompt; per set, heads in common with the others).
Patched by ``scripts/pham_frame_patch.sh``, reported by ``scripts/rebuild_report.py --dir pham_frame --stage pham``.

    .venv/bin/python scripts/pham_frame_sets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS, simple  # noqa: E402
from rebuild_sets import FOLDS, N_ALL, N_RANDOM, OUT as ADD, order_of, pool, random_sets, rankdata, ranks, top  # noqa: E402

READINGS = DATA / "readings" / "rebuild"
OUT = RESULTS / "rebuild" / "pham_frame"
N = 20
# Pham et al.'s code, formatted_tasks[task]["llama8b"], for the 14 tasks (all in their task list).
PHAM = {"antonym": ["opposite", "opposing", "contrasting"],
        "synonym": ["synonym", "same meaning", "same idea", "similar", "meaning"],
        "singular-plural": ["plural"], "present-past": ["past tense", "past form"],
        "country-capital": ["capital"], "country-currency": ["currency", "currencies"],
        "landmark-country": ["country"], "park-country": ["country", "countries"],
        "product-company": ["company", "developer"], "next_item": ["Next", "next position", "Next step"],
        "prev_item": ["before", "back one step", " previous item"],
        "english-french": ["French"], "english-german": ["German"], "english-spanish": ["Spanish"]}
# Gaps on Yang et al.'s templates, filled with the word that defines the task there: word -> templates it fills.
GAPS = {"antonym": {"antonym": {"inst0", "inst1", "inst2", "inst4"}},
        "park-country": {"nation": {"inst2"}},
        "prev_item": {"previous": {"inst3"}}}


def frame_strings(task: str) -> list[str]:
    return PHAM[task] + list(GAPS.get(task, {}))


def matched(text: str, strings: list[str], qspan: tuple[int, int]) -> dict[str, list[tuple[int, int]]]:
    """Occurrences of each string in ``text`` (case-insensitive substrings, as ``get_substring_indices``),
    outside the query."""
    low, out = text.lower(), {}
    for s in strings:
        sl, st, spans = s.lower(), 0, []
        while (st := low.find(sl, st)) != -1:
            e = st + len(s)
            if not (st < qspan[1] and e > qspan[0]):
                spans.append((st, e))
            st += 1
        out[s] = spans
    return out


def frame_positions(text: str, off, strings: list[str], qspan: tuple[int, int]) -> list[int]:
    spans = [sp for v in matched(text, strings, qspan).values() for sp in v]
    return sorted({i for i, (a, b) in enumerate(off) if i > 0 and a != b and any(a < e and b > s for s, e in spans)})


def query_span(text: str, query: str) -> tuple[int, int]:
    qs = text.rfind(query)  # as e20_bai_attention.categorise
    return qs, qs + len(query)


def segment_score(rows_of, positions_of, keys) -> np.ndarray:
    """Clamped relevance summed over each prompt's positions, averaged over the prompts where they are not empty."""
    total, count = np.zeros(N_ALL), 0
    for key in keys:
        pos = positions_of(key)
        if pos:
            total += np.maximum(rows_of(key)[pos], 0.0).sum(0)
            count += 1
    return total / count if count else np.zeros(N_ALL)


def main() -> None:
    from e20_bai_attention import CATS, categorise, task_pattern
    from lth.evaluation import tokenizer

    tok = tokenizer()
    instr = simple.instruction_dict()
    TASK = CATS.index("task")
    frames_info, overlap, all_sets = {}, {}, {}
    for fold, half in FOLDS.items():
        lists = json.loads((ADD / f"fold{fold}" / "word_lists.json").read_text())["tasks"]
        stored = np.load(ADD / f"fold{fold}" / "lrp_features.npz")
        score = {"F": {}, "T": {}}
        for t in simple.TASKS:
            f = np.load(READINGS / f"lrp_{half}" / f"{t}.npz")
            start = np.concatenate([[0], np.cumsum(f["lengths"])[:-1]])
            where = {(int(j), int(i)): (int(s), int(n)) for j, i, s, n in zip(f["wordings"], f["items"], start, f["lengths"])}
            data, pattern = simple.dataset(t), task_pattern(lists[t]["words"])
            prompts = {w: simple.prompts(t, w)[0] for w in simple.WORDINGS["rule"]}
            keys = pool(half, t)[0]
            pos_T, pos_F = {}, {}
            for j, item in keys:
                w = simple.WORDINGS["rule"][j]
                text = prompts[w][item]
                enc = tok(text, return_offsets_mapping=True)
                assert where[(j, item)][1] == len(enc.input_ids), (fold, t, w, item)
                cat, _, _ = categorise(text, instr[t][w[4:]], data[item]["input"], enc.input_ids, enc.offset_mapping, pattern)
                pos_T[(j, item)] = [i for i in range(len(cat)) if cat[i] == TASK]
                qspan = query_span(text, data[item]["input"])
                pos_F[(j, item)] = frame_positions(text, enc.offset_mapping, frame_strings(t), qspan)
                assert pos_F[(j, item)], f"(c) empty frame: fold {fold} {t} {w} item {item}"
                # (c) a gap word adds tokens only in its gap templates (elsewhere Pham's strings already cover it)
                own = frame_positions(text, enc.offset_mapping, PHAM[t], qspan)
                gap = any(w in g for g in GAPS.get(t, {}).values())
                assert (own != pos_F[(j, item)]) == gap and (gap or own), (fold, t, w, item)
                if w not in frames_info.setdefault(t, {}).setdefault(f"fold{fold}", {}):
                    dec = lambda ix: [tok.decode([enc.input_ids[i]]) for i in ix]  # noqa: E731
                    frames_info[t][f"fold{fold}"][w] = {"text": text, "frame": dec(pos_F[(j, item)]), "T": dec(pos_T[(j, item)])}

            def rows_of(key):
                s, n = where[key]
                return f["rows"][s:s + n].astype(np.float64)

            score["T"][t] = segment_score(rows_of, pos_T.get, keys)
            assert np.array_equal(score["T"][t], stored[f"{t}/T"]), f"(a) LRP-T not reproduced: fold {fold} {t}"
            score["F"][t] = segment_score(rows_of, pos_F.get, keys)
        print(f"fold {fold}: (a) LRP-T of all 14 tasks reproduced exactly; (c) frame non-empty on every pool prompt, "
              f"gap words add tokens only in their gap templates", flush=True)

        def raw_order(sc):
            return order_of(ranks(np.mean([sc[t] for t in simple.TASKS], axis=0)))

        def rank_order(sc):
            return order_of(rankdata(np.mean([ranks(sc[t]) for t in simple.TASKS], axis=0), method="average"))

        sets = {"shared_lrpFraw20": top(raw_order(score["F"]), N), "shared_lrpFrank20": top(rank_order(score["F"]), N),
                "shared_lrpTraw20": top(raw_order(score["T"]), N)}
        ref = json.loads((ADD / f"fold{fold}" / "heads.json").read_text())["simple"][simple.TASKS[0]]["sets"]
        (stored_T,) = [hs for _, nm, hs in ref if nm == f"shared_lrpT{N}"]
        assert top(rank_order(score["T"]), N) == stored_T, f"(b) shared_lrpT{N} not reproduced: fold {fold}"
        print(f"fold {fold}: (b) shared_lrpT{N} reproduced exactly", flush=True)
        sets[f"shared_lrpT{N}"] = stored_T  # reference only, not written (already patched in additions)
        all_sets[fold] = sets

        entries = []
        for name in ("shared_lrpFraw20", "shared_lrpFrank20", "shared_lrpTraw20"):
            heads = sets[name]
            entries += [["rule", name, heads]] + [["rule", f"random_{name}_{i}", r]
                                                  for i, r in enumerate(random_sets(heads, f"pham_frame/{fold}/shared/{name}"))]
        assert len(entries) == 3 * (N_RANDOM + 1)
        d = OUT / f"fold{fold}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "heads.json").write_text(json.dumps({"simple": {t: {"sets": entries} for t in simple.TASKS}}))
        names = list(sets)
        as_set = {k: set(map(tuple, v)) for k, v in sets.items()}
        overlap[f"fold{fold}"] = {a: {b: len(as_set[a] & as_set[b]) for b in names} for a in names}
        overlap[f"fold{fold}"]["layers"] = {k: sorted(l for l, _ in v) for k, v in sets.items()}
        print(f"fold {fold}: heads in common (of {N}):", flush=True)
        for a in names:
            print(f"  {a:18s} " + " ".join(f"{overlap[f'fold{fold}'][a][b]:3d}" for b in names), flush=True)
    # fold-to-fold: heads in common between the same rule's sets identified on the two halves
    stability = {name: len(set(map(tuple, all_sets[1][name])) & set(map(tuple, all_sets[2][name]))) for name in all_sets[1]}
    print("fold 1 vs fold 2, heads in common (of 20): " + ", ".join(f"{k} {v}" for k, v in stability.items()), flush=True)
    overlap["fold_to_fold"] = stability
    overlap["sets"] = {f"fold{f}": s for f, s in all_sets.items()}
    (OUT / "frames.json").write_text(json.dumps({"frames": frames_info, "overlap": overlap,
                                                 "pham_llama8b": PHAM, "gaps": {t: {w: sorted(v) for w, v in g.items()}
                                                                                for t, g in GAPS.items()}}, indent=1))
    print("written:", OUT, flush=True)


if __name__ == "__main__":
    main()
