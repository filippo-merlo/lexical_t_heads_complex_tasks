"""The causal check: mean head outputs patched into prompts without task information.

As in Yang et al. (2026, App. G; ``causal_recover_corrupted_prompts.py``): every head of a
lexical-task-head set is set, at the last token, to alpha times its mean ``z``, for alpha
in their scaling factors. The patched prompts are the prompts without task information
of the fixing items: the first 100 answered correctly 1-shot (simple tasks) or under the
original rules prompt (complex tasks); the baseline is their unpatched accuracy. Head sets:
Logit-head, J-head, and the heads only J-head finds.

Simple tasks follow Yang et al.'s code exactly (``run_simple``): the heads of each style
are patched with the mean of each source wording of each style separately (the first
100 correct prompts of that wording that are not patched), run in batches of 20 with the
padding masked (``lth.simple``); the control mean is the control task's 1-shot prompts at
all the items it answers correctly under the mean style's wording 1. Complex tasks: the
mean is taken over the style's mean set (``lth.store``), pooled over its wordings, and the
control patches the same heads with the mean of another task, same style.

Family ``simple-wrong`` is the second simple-task check, Yang et al.'s released
``causal_fix_incorrect.py``: the patched prompts are the 1-shot prompts of the first 100
items answered wrongly 1-shot (baseline 0), means and heads as above, no control. It is
the setting that reproduces their Figs. 57 and 62; App. G.1 describes the first.

Simple tasks are scored by the first token; complex tasks are generated greedily from the
patched prompt and scored by Fu et al.'s runner, as in E0.

Writes ``results/patching_{family}.csv`` (with ``--levels``, the levels are added to the
name): one row per task, style, head set, mean source and alpha (for the simple tasks also
per mean style and source wording), plus a baseline row (``source`` = ``none``) per task. In reasoning mode the first ``REASONING_ITEMS`` fixing
items are patched, over ``REASONING_ALPHAS``.

Every answer behind those accuracies is kept in ``results/answers/patching_{...}.jsonl.gz``
(same name as the table): one line per patched prompt and alpha, and per baseline prompt,
with the condition, item, expected and predicted answer, whether it is correct, and for
generated answers the reply.

    .venv/bin/python scripts/patch.py --gpu 0 --family simple
"""

from __future__ import annotations

import argparse
import csv
import gzip
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


def fu_answers(level, results, fixed_set, *, reasoning=False, **condition) -> list[dict]:
    """One answer line per fixing item in the results of Fu et al.'s runner."""
    from lth import fu

    lines = []
    for r in results:
        if fu.item_id(r) in fixed_set:
            expected, predicted = fu.answer(level, r, reasoning=reasoning)
            lines.append(dict(**condition, item=fu.item_id(r), expected=expected, predicted=predicted,
                              correct=fu.score(level, r, reasoning=reasoning)[0], reply=r["raw_response"]))
    return lines


def simple_mean(lm, ids: list[list[int]], items: list[int]):
    """Mean ``z`` at the last token over ``items``, run in batches and averaged in float32, as Yang et al."""
    total = None
    for b in range(0, len(items), simple.BATCH):
        _, z = lm.next_tokens_batch([ids[i] for i in items[b:b + simple.BATCH]])
        part = z.float().sum(0).cpu()
        total = part if total is None else total + part
    return total / len(items) if items else None


def run_simple(lm, heads_json, rows, log, tasks=simple.TASKS, family="simple"):
    """Yang et al.'s ``fix_corrupted_prompts_vary_prompt_instances_average_new``, with each
    style's heads (``intervention_component_prompt_type``) patched with the means of each
    style (``prompt_type_fix``), one source wording at a time, and their control.

    With ``family="simple-wrong"``, their ``fix_incorrect_prompts_vary_prompt_instances_average_new``
    instead: the patched prompts are the 1-shot prompts of the first 100 items answered
    wrongly 1-shot, and there is no control (commented out in their code)."""
    from lth.store import done, load, run_dir

    def records(task, wording):
        return load(run_dir("simple", task), wording)["records"]

    wrong = family == "simple-wrong"
    target = simple.FIXING if wrong else simple.ZERO
    for task in tasks:
        directory = run_dir("simple", task)
        if task not in heads_json.get("simple", {}) or not done(directory, simple.ZERO):
            continue
        one_shot = records(task, simple.FIXING)
        if wrong:
            fixed = [int(r["item"]) for r in one_shot if not r["correct"]][: simple.CAP]
        else:
            fixed = simple.correct_items(one_shot)[: simple.CAP]
        prompts, answers = simple.prompts(task, target)
        corrupt = simple.token_ids(lm.tokenizer, prompts)
        unpatched = {int(r["item"]): r for r in records(task, target)}
        rows.append(dict(family=family, task=task, style="", heads="", source="none", mean_style="",
                         mean_wording="", n_heads=0, n_prompts=len(fixed), n_mean=0, alpha="",
                         accuracy=sum(unpatched[i]["correct"] for i in fixed) / len(fixed)))
        log += [dict(family=family, task=task, style="", heads="", source="none", mean_style="",
                     mean_wording="", alpha="", item=str(i), expected=lm.tokenizer.decode([lm.first_token(answers[i])]),
                     predicted=lm.tokenizer.decode([unpatched[i]["prediction"]]), correct=unpatched[i]["correct"])
                for i in fixed]
        # Means: each source wording's first 100 correct items that are not patched; the
        # control: the control task's 1-shot prompts at all its items correct under the
        # mean style's wording 1.
        means = []  # (mean style, source, wording, items, mean)
        for mean_style, wordings in simple.WORDINGS.items():
            for wording in wordings:
                items = sorted(set(simple.correct_items(records(task, wording))) - set(fixed))[: simple.CAP]
                ids = simple.token_ids(lm.tokenizer, simple.prompts(task, wording)[0])
                means.append((mean_style, "task", wording, items, simple_mean(lm, ids, items)))
            if wrong:
                continue
            control = simple.CONTROL[task]
            items = simple.correct_items(records(control, simple.CONTROL_ITEMS[mean_style]))
            ids = simple.token_ids(lm.tokenizer, simple.prompts(control, simple.CONTROL_PROMPTS)[0])
            means.append((mean_style, "control", simple.CONTROL_PROMPTS, items, simple_mean(lm, ids, items)))
        for style in simple.WORDINGS:
            if style not in heads_json["simple"][task]:
                continue
            for name, heads in head_sets(heads_json["simple"][task][style]).items():
                if not heads:
                    continue
                for mean_style, source, wording, items, mean in means:
                    if mean is None:
                        continue
                    condition = dict(family=family, task=task, style=style, heads=name, source=source,
                                     mean_style=mean_style, mean_wording=wording)
                    for alpha in ALPHAS:
                        edits = lm.mean_patch(heads, mean, [alpha] * simple.BATCH)
                        hits = 0
                        for b in range(0, len(fixed), simple.BATCH):
                            batch = fixed[b:b + simple.BATCH]
                            if len(batch) < simple.BATCH:
                                edits = lm.mean_patch(heads, mean, [alpha] * len(batch))
                            tokens, _ = lm.next_tokens_batch([corrupt[i] for i in batch], edits=edits)
                            for i, token in zip(batch, tokens):
                                answer = lm.first_token(answers[i])
                                hits += token == answer
                                log.append(dict(**condition, alpha=alpha, item=str(i),
                                                expected=lm.tokenizer.decode([answer]),
                                                predicted=lm.tokenizer.decode([token]), correct=token == answer))
                        rows.append(dict(**condition, n_heads=len(heads), n_prompts=len(fixed), n_mean=len(items),
                                         alpha=alpha, accuracy=hits / len(fixed)))
        print(f"{family} {task} done", flush=True)


def run_complex(lm, heads_json, rows, log, levels):
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
        log += fu_answers(level, fu.recorded_results(directory / "fu_out" / "bare"), fixed_set,
                          family="complex", task=key, style="", heads="", source="none", alpha="")
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
                lines = fu_answers(level, results, fixed_set, family="complex", task=key, style=style,
                                   heads=name, source=source, alpha=alpha)
                log += lines
                correct = [line["correct"] for line in lines]
                rows.append(dict(family="complex", task=key, style=style, heads=name, source=source,
                                 n_heads=len(heads), n_prompts=len(correct), alpha=alpha,
                                 accuracy=sum(correct) / len(correct)))
            print(f"complex {key} {style} {name} {source} ({time.time() - start:.0f} s)", flush=True)


def run_reasoning(lm, heads_json, rows, log, family, levels):
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
        log += fu_answers(level, fu.recorded_results(directory / "fu_out" / "bare"), fixed_set, reasoning=True,
                          family=family, task=key, style="", heads="", source="none", alpha="")
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
                lines = fu_answers(level, results, fixed_set, reasoning=True, family=family, task=key,
                                   style=style, heads=name, source=source, alpha=alpha)
                log += lines
                correct = [line["correct"] for line in lines]
                rows.append(dict(family=family, task=key, style=style, heads=name, source=source,
                                 n_heads=len(heads), n_prompts=len(correct), alpha=alpha,
                                 accuracy=sum(correct) / len(correct)))
            print(f"{family} {key} {style} {name} {source} ({time.time() - start:.0f} s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--family", choices=("simple", "simple-wrong", "complex", "reasoning", "reasoning-pre"),
                        required=True)
    parser.add_argument("--levels", nargs="+", default=None,
                        help="complex task levels to patch (default: all in the study); "
                             "the output file then names them")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="simple tasks to patch (default: all); the output file then ends in --part")
    parser.add_argument("--part", default=None, help="name of this share of the simple tasks")
    args = parser.parse_args()
    gpu.pin(args.gpu)

    from lth.model import LM

    lm = LM.load()
    heads_json = json.loads((RESULTS / "heads.json").read_text())
    rows: list[dict] = []
    log: list[dict] = []
    if args.family in ("reasoning", "reasoning-pre"):
        run_reasoning(lm, heads_json, rows, log, args.family, args.levels)
    elif args.family == "complex":
        run_complex(lm, heads_json, rows, log, args.levels)
    else:
        run_simple(lm, heads_json, rows, log, args.tasks or simple.TASKS, args.family)
    name = args.family + "".join("_" + level.replace("/", "-") for level in args.levels or [])
    if args.tasks:
        name += f".part-{args.part}"
    with (RESULTS / f"patching_{name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "answers").mkdir(exist_ok=True)
    with gzip.open(RESULTS / "answers" / f"patching_{name}.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(line, ensure_ascii=False) + "\n" for line in log)


if __name__ == "__main__":
    main()
