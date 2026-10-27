"""The causal check: mean head outputs patched into prompts without task information.

As in Yang et al. (2026, App. G; ``causal_recover_corrupted_prompts.py``): every head of a
lexical-task-head set is set, at the last token, to alpha times its mean ``z``, for alpha
in their scaling factors. The mean is taken over the style's mean set (``lth.store``),
pooled over its wordings. The patched prompts are the prompts without task information
of the fixing items: the first 100 answered correctly 1-shot (simple tasks) or under the
original rules prompt (complex tasks); the baseline is their unpatched accuracy. Head sets:
Logit-head, J-head, and the heads only J-head finds. The control patches the same heads
with the mean of another task, same style.

Simple tasks are scored by the first token; complex tasks are generated greedily from the
patched prompt and scored by Fu et al.'s runner, as in E0.

Writes ``results/patching_{family}.csv`` (with ``--levels``, the levels are added to the
name): one row per task, style, head set, mean source and alpha, plus a baseline row
(``source`` = ``none``) per task. In reasoning mode the first ``REASONING_ITEMS`` fixing
items are patched, over ``REASONING_ALPHAS``.

    .venv/bin/python scripts/patch.py --gpu 0 --family simple
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import RESULTS, gpu, simple  # noqa: E402

ALPHAS = [-10, -6, -4, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Yang et al., causal_fix_incorrect.py
# Reasoning mode (proposal Sec. 5.3): a fixed sample of patched items and a coarser grid,
# to bound the cost of long generations (user decision, 2026-10-27).
REASONING_ITEMS = 10
REASONING_ALPHAS = [0, 1, 4, 10]
# Controls for the complex tasks: another task with the same answer format, same level.
COMPLEX_CONTROL = {
    **{f"noun_class_agreement/{l}": f"lexical_category_inference/{l if l == 'easy' else l + '-or'}"
       for l in ("easy", "medium", "hard")},
    **{f"lexical_category_inference/{k}": f"noun_class_agreement/{k.split('-')[0]}"
       for k in ("easy", "medium-or", "medium-and", "hard-or", "hard-and")},
    **{f"set_game/{l}": f"operator_func/{l}" for l in ("easy", "medium", "hard")},
    **{f"operator_func/{l}": f"set_game/{l}" for l in ("easy", "medium", "hard")},
    **{f"tapatan/{l}": f"set_game/{l}" for l in ("easy", "medium", "hard")},
}


def style_mean(directory: Path, wordings):
    """Mean ``z`` over the style's mean set, pooled over its wordings; None if it is empty."""
    import torch

    from lth.store import done, load

    total, count = None, 0
    for wording in wordings:
        if done(directory, wording):
            run = load(directory, wording)
            total = run["z_sum"] if total is None else total + run["z_sum"]
            count += len(run["mean_items"])
    return torch.from_numpy(total / count) if count else None


def head_sets(heads: dict) -> dict[str, list[tuple[int, int]]]:
    logit = {(l, h) for l, h, _ in heads["logithead"]["heads"]}
    j = {(l, h) for l, h, _ in heads["jhead"]["heads"]}
    return {"logithead": sorted(logit), "jhead": sorted(j), "jhead_only": sorted(j - logit)}


def conditions(heads_by_style, styles, directory, control_directory):
    """``(style, head set name, heads, source, mean)`` for every patch to run."""
    for style, wordings in styles.items():
        if style not in heads_by_style:
            continue
        means = {"task": style_mean(directory, wordings), "control": style_mean(control_directory, wordings)}
        for name, heads in head_sets(heads_by_style[style]).items():
            for source, mean in means.items():
                if heads and mean is not None:
                    yield style, name, heads, source, mean


def run_simple(lm, heads_json, rows):
    from lth.store import done, load, run_dir

    for task in simple.TASKS:
        directory = run_dir("simple", task)
        if task not in heads_json.get("simple", {}) or not done(directory, simple.ZERO):
            continue
        fixed = load(directory, simple.FIXING)["fixed"].tolist()
        prompts, answers = simple.prompts(task, simple.ZERO)
        zero = {r["item"]: r["correct"] for r in load(directory, simple.ZERO)["records"]}
        rows.append(dict(family="simple", task=task, style="", heads="", source="none", n_heads=0,
                         n_prompts=len(fixed), alpha="", accuracy=sum(zero[i] for i in fixed) / len(fixed)))
        targets = [(lm.encode_raw(prompts[int(i)]), lm.first_token(answers[int(i)])) for i in fixed]
        for style, name, heads, source, mean in conditions(
                heads_json["simple"][task], simple.WORDINGS, directory, run_dir("simple", simple.CONTROL[task])):
            edits = lm.mean_patch(heads, mean, ALPHAS)
            hits = [0] * len(ALPHAS)
            for ids, answer in targets:
                for i, token in enumerate(lm.next_tokens(ids, len(ALPHAS), edits=edits)):
                    hits[i] += token == answer
            rows += [dict(family="simple", task=task, style=style, heads=name, source=source,
                          n_heads=len(heads), n_prompts=len(targets), alpha=a, accuracy=h / len(targets))
                     for a, h in zip(ALPHAS, hits)]
        print(f"simple {task} done", flush=True)


def run_complex(lm, heads_json, rows, levels):
    from lth import fu
    from lth.store import done, load, run_dir

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from identify_heads import complex_styles

    styles = complex_styles()
    for key in levels or fu.STUDY_LEVELS:
        level = fu.TASK_LEVELS[key]
        directory = run_dir("complex", key)
        if key not in heads_json.get("complex", {}) or not done(directory, "orig"):
            continue
        fixed = load(directory, "orig")["fixed"].tolist()
        if not fixed:
            print(f"complex {key}: no fixing items", flush=True)
            continue
        original = fu.ORIGINALS[level.original]
        bare = fu.input_block(fu.original_template(original), original)
        prompts = fu.recorded_prompts(directory / "fu_out" / "bare")
        baseline = {r["item"]: r["correct"] for r in load(directory, "bare")["records"]}
        rows.append(dict(family="complex", task=key, style="", heads="", source="none", n_heads=0,
                         n_prompts=len(fixed), alpha="", accuracy=sum(baseline[i] for i in fixed) / len(fixed)))
        fixed_set = set(fixed)
        for style, name, heads, source, mean in conditions(
                heads_json["complex"][key], styles, directory, run_dir("complex", COMPLEX_CONTROL[key])):
            start = time.time()
            edits = lm.mean_patch(heads, mean, ALPHAS)
            replies = {}
            for item in fixed:
                out, _ = lm.generate(lm.encode_chat(prompts[item]), fu.MAX_TOKENS[level.task],
                                     edits=edits, rows=len(ALPHAS))
                for alpha, reply in zip(ALPHAS, out):
                    replies[(prompts[item], alpha)] = reply
            for alpha in ALPHAS:
                with tempfile.TemporaryDirectory() as scratch:
                    results = fu.run(level, "rules", lambda ps, a=alpha: [replies.get((p, a), "") for p in ps],
                                     Path(scratch), rules=bare)
                correct = [fu.score(level, r)[0] for r in results if fu.item_id(r) in fixed_set]
                rows.append(dict(family="complex", task=key, style=style, heads=name, source=source,
                                 n_heads=len(heads), n_prompts=len(correct), alpha=alpha,
                                 accuracy=sum(correct) / len(correct)))
            print(f"complex {key} {style} {name} {source} ({time.time() - start:.0f} s)", flush=True)


def run_reasoning(lm, heads_json, rows, family, levels):
    """Reasoning mode (Sec. 5.3): the patch stays on at every generated token, and the final
    answer is scored; ``family`` says whose heads and means are patched (last prompt token,
    or pre-answer position)."""
    from lth import fu
    from lth.store import done, load, run_dir

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from identify_heads import complex_styles
    from run_reasoning import N_ITEMS, PRE

    suffix = PRE if family == "reasoning-pre" else ""
    styles = {s: tuple(w + suffix for w in ws) for s, ws in complex_styles().items()}
    for key in levels or fu.STUDY_LEVELS:
        level = fu.TASK_LEVELS[key]
        directory = run_dir("reasoning", key)
        if key not in heads_json.get(family, {}) or not done(directory, "orig"):
            continue
        fixed = load(directory, "orig")["fixed"].tolist()[:REASONING_ITEMS]
        if not fixed:
            print(f"{family} {key}: no fixing items", flush=True)
            continue
        original = fu.ORIGINALS[level.original]
        bare = fu.input_block(fu.original_template(original), original)
        prompts = fu.recorded_prompts(directory / "fu_out" / "bare")
        baseline = {r["item"]: r["correct"] for r in load(directory, "bare")["records"]}
        rows.append(dict(family=family, task=key, style="", heads="", source="none", n_heads=0,
                         n_prompts=len(fixed), alpha="", accuracy=sum(baseline[i] for i in fixed) / len(fixed)))
        fixed_set = set(fixed)
        for style, name, heads, source, mean in conditions(
                heads_json[family][key], styles, directory, run_dir("reasoning", COMPLEX_CONTROL[key])):
            start = time.time()
            grid = [(item, alpha) for item in fixed for alpha in REASONING_ALPHAS]
            ids = [lm.encode_chat(prompts[item]) for item, _ in grid]
            replies = {}
            for batch in lm.token_batches([len(i) for i in ids], fu.REASONING_MAX_TOKENS):
                edits = lm.mean_patch(heads, mean, [grid[r][1] for r in batch])
                out = lm.generate_batch([ids[r] for r in batch], fu.REASONING_MAX_TOKENS, edits=edits,
                                        stop=lambda text: fu.answer_complete(level, text))
                for r, reply in zip(batch, out):
                    replies[(prompts[grid[r][0]], grid[r][1])] = lm.decode(reply)
            for alpha in REASONING_ALPHAS:
                with tempfile.TemporaryDirectory() as scratch:
                    results = fu.run(level, "rules", lambda ps, a=alpha: [replies.get((p, a), "") for p in ps],
                                     Path(scratch), rules=bare, n_test=N_ITEMS, batch=N_ITEMS, reasoning=True)
                correct = [fu.score(level, r, reasoning=True)[0] for r in results if fu.item_id(r) in fixed_set]
                rows.append(dict(family=family, task=key, style=style, heads=name, source=source,
                                 n_heads=len(heads), n_prompts=len(correct), alpha=alpha,
                                 accuracy=sum(correct) / len(correct)))
            print(f"{family} {key} {style} {name} {source} ({time.time() - start:.0f} s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--family", choices=("simple", "complex", "reasoning", "reasoning-pre"), required=True)
    parser.add_argument("--levels", nargs="+", default=None,
                        help="complex task levels to patch (default: all in the study); "
                             "the output file then names them")
    args = parser.parse_args()
    gpu.pin(args.gpu)

    from lth.model import LM

    lm = LM.load()
    heads_json = json.loads((RESULTS / "heads.json").read_text())
    rows: list[dict] = []
    if args.family in ("reasoning", "reasoning-pre"):
        run_reasoning(lm, heads_json, rows, args.family, args.levels)
    elif args.family == "complex":
        run_complex(lm, heads_json, rows, args.levels)
    else:
        run_simple(lm, heads_json, rows)
    name = args.family + "".join("_" + level.replace("/", "-") for level in args.levels or [])
    with (RESULTS / f"patching_{name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
