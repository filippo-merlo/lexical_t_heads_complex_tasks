"""AttnLRP relevance rows of the identification pool (Pham et al. 2026, mapped onto paper/head-sets-simple-design.tex
Sec. 5.3), for the rule LRP-T (``scripts/lrp_sets.py``).

Prompts: each half's identification pool, exactly the prompts the features of ``scripts/rebuild_sets.py`` use
(``rebuild_sets.pool``: Yang et al.'s five templates, up to 300 correctly answered prompts per task spread over
the templates used), run alone, eager attention. Target: the first token of " " + answer, as for Causal
(``scripts/causal_context_cie.py``). One forward and one AttnLRP backward pass from the target's logit at the
last token (``lth.lrp.relevance_row``); the row kept is, for every head, ``A[n, j] * dlogit/dA[n, j]`` of the last
position ``n`` over all positions ``j``.

Writes ``data/readings/rebuild/lrp_<half>/<task>.npz``: ``rows`` [sum of prompt lengths, n_layers * n_heads]
(float32, prompts one after another, positions in order), ``lengths``, ``wordings`` (template index into
``lth.simple.WORDINGS["rule"]``), ``items``, ``target``, ``rank`` (the target's rank, 1 = top), ``logit``,
``greedy`` (the top token) and ``head_rel`` [prompts, n_layers * n_heads] (``z . dlogit/dz`` at ``o_proj``'s
input). A task already there is skipped.

Checks on the first prompt of every task, before its file is written (the run stops at a failure):
(a) the last-token logits under AttnLRP's rules equal those without them, bit for bit; (b) the attention row
equals the one ``scripts/rebuild_attention.py`` stored, bit for bit; (c) for every head, the row summed over
positions equals ``head_rel`` within 1% of the largest ``|head_rel|``, under the rules and with plain gradients;
(d) the rules change the gradient at the input embeddings (by more than 0.1% of its largest value).

``--fp32-check TASK``: 20 pool prompts of TASK on half A, evenly spaced over the pool, in bfloat16 and again with the model in
float32; prints the overlap of the top 40 heads by LRP-T (fold 1's word list) and by the whole-row score.

    .venv/bin/python scripts/rebuild_lrp.py --gpu 0 --halves A --tasks antonym --check-only
    .venv/bin/python scripts/rebuild_lrp.py --gpu 0 --fp32-check antonym
    .venv/bin/python scripts/rebuild_lrp.py --gpu 0 --halves A
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS, simple  # noqa: E402

OUT = DATA / "readings" / "rebuild"
FOLD_OF = {"A": 1, "B": 2}
FP32_PROMPTS, FP32_TOP = 20, 40


def stored_rows(half: str, task: str):
    """(template index, item) -> that prompt's stored attention rows [T, n_layers * n_heads]."""
    att = np.load(OUT / f"attention_{half}" / f"{task}.npz")
    start = np.concatenate([[0], np.cumsum(att["lengths"])[:-1]])
    rows = att["rows"]
    return {(int(j), int(i)): rows[s:s + n] for j, i, s, n in zip(att["wordings"], att["items"], start, att["lengths"])}


def check(lm, ids, target, r, stored, label: str) -> None:
    import torch

    from lth.lrp import relevance_row

    plain = relevance_row(lm, ids, target, lrp=False, extras=True)
    a = torch.equal(r["logits"], plain["logits"])
    b = r["att"].shape == stored.shape and np.array_equal(r["att"], stored)

    def identity(x):
        return float(np.abs(x["rel"].sum(0) - x["head_rel"]).max() / np.abs(x["head_rel"]).max())

    c_lrp, c_plain = identity(r), identity(plain)
    g = plain["input_grad"]
    d = float((r["input_grad"] - g).abs().max() / g.abs().max())
    ok = a and b and c_lrp <= 1e-2 and c_plain <= 1e-2 and d > 1e-3
    print(f"check {label}: (a) logits equal {a}; (b) attention row equal {b}; (c) row sum vs head relevance "
          f"{c_lrp:.1e} (LRP), {c_plain:.1e} (plain); (d) input-gradient change {d:.2f}; target rank {r['rank']}"
          f" -> {'ok' if ok else 'FAILED'}", flush=True)
    if not ok:
        raise SystemExit(f"check failed on {label}")


def task_positions(lm, task: str, keys, words: list[str]):
    """Per pool prompt, the positions of the task-word segment (``e20_bai_attention.categorise``)."""
    from e20_bai_attention import CATS, categorise, task_pattern

    tok, data, pattern = lm.tokenizer, simple.dataset(task), task_pattern(words)
    prompts = {w: simple.prompts(task, w)[0] for w in simple.WORDINGS["rule"]}
    out = []
    for j, item in keys:
        w = simple.WORDINGS["rule"][j]
        text = prompts[w][item]
        enc = tok(text, return_offsets_mapping=True)
        cat, _, _ = categorise(text, simple.instruction_dict()[task][w[4:]], data[item]["input"],
                               enc.input_ids, enc.offset_mapping, pattern)
        out.append(cat == CATS.index("task"))
    return out


def fp32_check(lm, task: str) -> None:
    import torch

    from rebuild_sets import pool

    from lth.lrp import clamped, relevance_row

    keys = pool("A", task)[0]
    keys = [keys[int(i)] for i in np.linspace(0, len(keys) - 1, FP32_PROMPTS)]  # spread over the templates
    words = json.loads((RESULTS / "rebuild" / "additions" / "fold1" / "word_lists.json").read_text())["tasks"][task]["words"]
    masks = task_positions(lm, task, keys, words)
    if sum(m.any() for m in masks) < FP32_PROMPTS // 2:
        raise SystemExit(f"{task}: the task-word segment is empty on most of the {FP32_PROMPTS} prompts")
    data = simple.dataset(task)
    prompts = {w: simple.prompts(task, w)[0] for w in simple.WORDINGS["rule"]}
    inputs = [(lm.tokenizer(prompts[simple.WORDINGS["rule"][j]][item]).input_ids,
               lm.first_token(" " + data[item]["output"])) for j, item in keys]

    def scores():
        t_sum, t_n, whole = 0.0, 0, 0.0
        for (ids, y), m in zip(inputs, masks):
            rel = clamped(relevance_row(lm, ids, y)["rel"].astype(np.float64))
            whole = whole + rel.sum(0)
            if m.any():
                t_sum, t_n = t_sum + rel[m].sum(0), t_n + 1
        return t_sum / max(t_n, 1), whole / len(inputs)

    bf16 = scores()
    lm.model.hf_model.float()
    torch.cuda.empty_cache()
    fp32 = scores()
    for name, x, y in (("LRP-T", bf16[0], fp32[0]), ("whole row", bf16[1], fp32[1])):
        top_x, top_y = set(np.argsort(-x)[:FP32_TOP]), set(np.argsort(-y)[:FP32_TOP])
        rank_corr = np.corrcoef(np.argsort(np.argsort(-x)), np.argsort(np.argsort(-y)))[0, 1]
        print(f"fp32 check {task}, {len(inputs)} prompts ({sum(m.any() for m in masks)} with task words), {name}: "
              f"top-{FP32_TOP} overlap bf16 vs fp32 {len(top_x & top_y)}/{FP32_TOP}; Spearman {rank_corr:.4f}",
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--halves", nargs="+", choices=("A", "B"), default=["A", "B"])
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS))
    parser.add_argument("--check-only", action="store_true", help="run the checks on each task's first prompt only")
    parser.add_argument("--fp32-check", default=None, metavar="TASK", help="compare bfloat16 with float32 on TASK")
    args = parser.parse_args()

    from lth import gpu
    gpu.pin(args.gpu)

    from rebuild_sets import pool

    from lth.lrp import relevance_row
    from lth.model import LM

    lm = LM.load()
    if args.fp32_check:
        fp32_check(lm, args.fp32_check)
        return
    tok = lm.tokenizer
    for half in args.halves:
        out = OUT / f"lrp_{half}"
        out.mkdir(parents=True, exist_ok=True)
        for task in args.tasks:
            path = out / f"{task}.npz"
            if path.exists() and not args.check_only:
                print(f"{half} {task}: done before, skipped", flush=True)
                continue
            start = time.time()
            keys = pool(half, task)[0]
            data = simple.dataset(task)
            prompts = {w: simple.prompts(task, w)[0] for w in simple.WORDINGS["rule"]}
            stored = stored_rows(half, task)
            kept = {k: [] for k in ("rows", "lengths", "wordings", "items", "target", "rank", "logit", "greedy",
                                    "head_rel")}
            for k, (j, item) in enumerate(keys):
                ids = tok(prompts[simple.WORDINGS["rule"][j]][item]).input_ids
                target = lm.first_token(" " + data[item]["output"])
                r = relevance_row(lm, ids, target, extras=k == 0)
                if k == 0:
                    check(lm, ids, target, r, stored[(j, item)], f"{half} {task} {simple.WORDINGS['rule'][j]} item {item}")
                    if args.check_only:
                        break
                for name, value in (("rows", r["rel"]), ("lengths", len(ids)), ("wordings", j), ("items", item),
                                    ("target", target), ("rank", r["rank"]), ("logit", r["logit"]),
                                    ("greedy", r["greedy"]), ("head_rel", r["head_rel"])):
                    kept[name].append(value)
            if args.check_only:
                continue
            partial = out / f"{task}.partial.npz"
            np.savez(partial, rows=np.concatenate(kept["rows"]).astype(np.float32),
                     lengths=np.array(kept["lengths"]), wordings=np.array(kept["wordings"], dtype=np.int8),
                     items=np.array(kept["items"]), target=np.array(kept["target"]), rank=np.array(kept["rank"]),
                     logit=np.array(kept["logit"], dtype=np.float32), greedy=np.array(kept["greedy"]),
                     head_rel=np.stack(kept["head_rel"]).astype(np.float32))
            partial.replace(path)
            rank = np.array(kept["rank"])
            print(f"{half} {task}: {len(rank)} prompts ({time.time() - start:.0f} s); target top-1 on "
                  f"{(rank == 1).mean():.2f}, top-5 on {(rank <= 5).mean():.2f}, median rank {np.median(rank):.0f}",
                  flush=True)


if __name__ == "__main__":
    main()
