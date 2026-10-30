"""The causal check: mean head outputs patched into prompts without task information.

As in Yang et al. (2026, App. G; ``causal_recover_corrupted_prompts.py``): every head of a
lexical-task-head set is set, at the last prompt token, to alpha times its mean ``z``, for
alpha in their scaling factors; the reply is then generated greedily from the patched
prompt, the reply's own tokens unpatched. The patched prompts are the prompts without task
information of the fixing items: the first 100 answered correctly 1-shot (simple tasks) or
under the original rules prompt (complex tasks); the baseline is their unpatched accuracy.
Head sets: Logit-head, J-head, and the heads only J-head finds. "Correct" is the flexible
score throughout (proposal Sec. 5.5): for the fixing items, the mean sets and the control,
read from ``scripts/build_readings.py``, and for the patched replies.

Simple tasks follow Yang et al.'s code (``run_simple``): the heads of each style are
patched with the mean of each source wording of each style separately (the first 100
correct prompts of that wording that are not patched); the control mean is the control
task's 1-shot prompts at all the items it answers correctly under the mean style's wording
1. Unlike their code, which reads the first token, the reply is generated as the saved
outputs are (``scripts/run_outputs.py``: up to the delimiter ``;``, a newline once begun,
EOS, or 32 tokens) and scored by the flexible score; the first-token score is kept beside it.

Family ``simple-wrong`` is the second simple-task check, Yang et al.'s released
``causal_fix_incorrect.py``: the patched prompts are the 1-shot prompts of the first 100
items answered wrongly 1-shot (baseline 0), means and heads as above, no control. It is
the setting that reproduces their Figs. 57 and 62; App. G.1 describes the first.

Complex tasks (``run_fu``), all five alike, rule style only: the rule heads are patched with
the mean of each rule wording separately (its mean set, ``lth.store``); the control mean is
the control task's mean under its original rules prompt. Family ``complex-wrong`` is the
second check adapted: the original rules prompt of the first 100 items answered wrongly
under it, no control. The reply runs to EOS or ``COMPLEX_MAX_TOKENS`` tokens, over
``COMPLEX_ALPHAS``, and is scored by the flexible score (``lth.evaluation.flexible_complex``).

Prompts are run in batches (padding masked, at most ``PATCH_TOKENS`` prompt-plus-reply
tokens), every patched prompt under every alpha.

Writes ``results/patching_{family}.csv`` (with ``--levels``, the levels are added to the
name): one row per task, style, head set, mean source, mean wording and alpha (for the
simple tasks also per mean style), plus a baseline row (``source`` = ``none``) per task.

Every answer behind those accuracies is kept in ``results/answers/patching_{...}.jsonl.gz``
(same name as the table): one line per patched prompt and alpha, and per baseline prompt,
with the condition, item, expected and predicted answer, whether it is correct, and the reply.

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
# Complex tasks: the even alphas of Yang et al.'s grid and 1, the same for all five tasks,
# to bound the cost of 1,024-token replies (user decision, 2026-10-30).
COMPLEX_ALPHAS = [-10, -6, -4, -2, 0, 1, 2, 4, 6, 8, 10]
# The complex tasks' check runs on the rule style only (user decision, 2026-10-28); check 2
# adds WRONG to the family name.
STYLE = "rule"
WRONG = "-wrong"
COMPLEX_FAMILIES = ("complex", "complex" + WRONG)
# Rows per generation batch: at most this many, and padded prompt-plus-reply tokens within
# ``lth.model.TOKEN_BUDGET``.
PATCH_ROWS = {"simple": 256, "complex": 256}
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


def wording_mean(directory: Path, wording: str):
    """Mean ``z`` over a wording's mean set (``lth.store``) and its size; None if empty."""
    import torch

    from lth.store import done, load

    if not done(directory, wording):
        return None, 0
    run = load(directory, wording)
    n = len(run["mean_items"])
    return (torch.from_numpy(run["z_sum"] / n), n) if n else (None, 0)


def head_sets(heads: dict) -> dict[str, list[tuple[int, int]]]:
    logit = {(l, h) for l, h, _ in heads["logithead"]["heads"]}
    j = {(l, h) for l, h, _ in heads["jhead"]["heads"]}
    return {"logithead": sorted(logit), "jhead": sorted(j), "jhead_only": sorted(j - logit)}


def generate_patched(lm, prompts: list[list[int]], heads, mean, alphas, max_new_tokens: int, max_rows: int,
                     stop=None) -> dict[tuple[int, float], list[int]]:
    """``(prompt index, alpha)`` -> the greedy reply (token ids) to that prompt with the heads
    patched at its last token; every prompt under every alpha, shortest prompts first."""
    pairs = sorted(((p, a) for p in range(len(prompts)) for a in alphas), key=lambda pa: len(prompts[pa[0]]))
    replies = {}
    for batch in lm.token_batches([len(prompts[p]) for p, _ in pairs], max_new_tokens, max_rows):
        chosen = [pairs[r] for r in batch]
        edits = lm.mean_patch(heads, mean, [a for _, a in chosen], prompt_only=True)
        replies.update(zip(chosen, lm.generate_batch([prompts[p] for p, _ in chosen], max_new_tokens,
                                                     edits=edits, stop=stop)))
    return replies


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
    from run_outputs import SIMPLE_MAX_TOKENS, simple_answer

    from lth.evaluation import flexible_simple
    from lth.store import done, load, readings_dir

    def records(task, wording):
        return load(readings_dir("simple", task), wording)["records"]

    def stop(text):
        return simple_answer(text) != text

    wrong = family == "simple-wrong"
    target = simple.FIXING if wrong else simple.ZERO
    for task in tasks:
        directory = readings_dir("simple", task)
        if task not in heads_json.get("simple", {}) or not done(directory, target):
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
                         accuracy=sum(unpatched[i]["correct"] for i in fixed) / len(fixed),
                         first_token=sum(unpatched[i]["first_token"] for i in fixed) / len(fixed)))
        log += [dict(family=family, task=task, style="", heads="", source="none", mean_style="",
                     mean_wording="", alpha="", item=str(i), expected=answers[i], predicted=unpatched[i]["answer"],
                     correct=unpatched[i]["correct"], first_token=unpatched[i]["first_token"]) for i in fixed]
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
        patched = [corrupt[i] for i in fixed]
        first = [lm.first_token(answers[i]) for i in fixed]
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
                    replies = generate_patched(lm, patched, heads, mean, ALPHAS, SIMPLE_MAX_TOKENS,
                                               PATCH_ROWS["simple"], stop=stop)
                    for alpha in ALPHAS:
                        hits = tokens = 0
                        for p, i in enumerate(fixed):
                            reply = replies[p, alpha]
                            text = lm.tokenizer.decode(reply, skip_special_tokens=True)
                            answer = simple_answer(text)
                            correct = flexible_simple({"answer_text": answer, "reference": answers[i]})
                            first_token = reply[:1] == [first[p]]
                            hits += correct
                            tokens += first_token
                            log.append(dict(**condition, alpha=alpha, item=str(i), expected=answers[i],
                                            predicted=answer, correct=correct, first_token=first_token, reply=text))
                        rows.append(dict(**condition, n_heads=len(heads), n_prompts=len(fixed), n_mean=len(items),
                                         alpha=alpha, accuracy=hits / len(fixed), first_token=tokens / len(fixed)))
        print(f"{family} {task} done", flush=True)


def score_complex(level, mode: str, rules: str | None, replies: dict[str, str], items: set[str]) -> list[dict]:
    """Flexible scores of the replies to the patched prompts (prompt -> reply), by Fu et
    al.'s runner over the answers :func:`lth.evaluation.flexible_answer` reads, as
    :func:`lth.evaluation.flexible_complex`; one line per item in ``items``."""
    from lth import fu
    from lth.evaluation import flexible_answer

    answers = {p: flexible_answer(level.task, reply) for p, reply in replies.items()}
    with tempfile.TemporaryDirectory() as scratch:
        results = fu.run(level, mode, lambda prompts: [answers.get(p) or "" for p in prompts], Path(scratch),
                         rules=rules)
    return [dict(item=fu.item_id(r), expected=str(fu.answer(level, r)[0]), predicted=answers[r["prompt"]],
                 correct=fu.score(level, r, reasoning=True)[0], reply=replies[r["prompt"]])
            for r in results if fu.item_id(r) in items]


def run_fu(lm, heads_json, rows, log, family, levels):
    """The complex tasks, rule style only: each rule wording's own mean patched into the rule
    heads, one wording at a time, as for the simple tasks.

    Check 1 (``complex``) patches the prompts without task information of the fixing items,
    with a control: the control task's mean under its original rules prompt. Check 2
    (``complex-wrong``) patches the original rules prompt of the first 100 items answered
    wrongly under it, without a control. The patch is at the last prompt token only."""
    from lth import fu
    from lth.store import CAP, done, load, load_outputs, outputs_path, readings_dir

    from identify_heads import complex_styles
    from run_complex import BARE, FIXING, variants
    from run_outputs import COMPLEX_MAX_TOKENS

    wrong = family.endswith(WRONG)
    for key in levels or fu.STUDY_LEVELS:
        level = fu.TASK_LEVELS[key]
        directory = readings_dir("complex", key)
        if key not in heads_json.get("complex", {}) or not done(directory, FIXING):
            continue
        original = load(directory, FIXING)
        if wrong:
            items = [r["item"] for r in original["records"] if not r["correct"]][:CAP]
            target = FIXING
        else:
            items = original["fixed"].tolist()
            target = BARE
        if not items:
            print(f"{family} {key}: no items to patch", flush=True)
            continue
        mode, rules = {name: (m, r) for name, m, r in variants(level)}[target]
        outputs = {r["item"]: r for r in load_outputs(outputs_path("complex", key, target))}
        unpatched = {r["item"]: r for r in load(directory, target)["records"]}
        rows.append(dict(family=family, task=key, style="", heads="", source="none", mean_wording="", n_heads=0,
                         n_prompts=len(items), n_mean=0, alpha="",
                         accuracy=sum(unpatched[i]["correct"] for i in items) / len(items)))
        log += [dict(family=family, task=key, style="", heads="", source="none", mean_wording="", alpha="",
                     item=i, expected=str(outputs[i]["reference"]), predicted=unpatched[i]["answer"],
                     correct=unpatched[i]["correct"], reply=outputs[i]["reply_text"]) for i in items]
        means = [("task", w, *wording_mean(directory, w)) for w in complex_styles()[STYLE]]
        if not wrong:
            means.append(("control", FIXING, *wording_mean(readings_dir("complex", COMPLEX_CONTROL[key]), FIXING)))
        # One reply per distinct prompt (Tapatan's sample repeats boards).
        prompts = list(dict.fromkeys(outputs[i]["prompt"] for i in items))
        ids = [lm.encode_chat(p) for p in prompts]
        item_set = set(items)
        for name, heads in head_sets(heads_json["complex"][key][STYLE]).items():
            for source, wording, mean, n_mean in means:
                if not heads or mean is None:
                    continue
                start = time.time()
                replies = generate_patched(lm, ids, heads, mean, COMPLEX_ALPHAS, COMPLEX_MAX_TOKENS,
                                           PATCH_ROWS["complex"])
                condition = dict(family=family, task=key, style=STYLE, heads=name, source=source,
                                 mean_wording=wording)
                for alpha in COMPLEX_ALPHAS:
                    texts = {prompts[p]: lm.decode(replies[p, alpha]) for p in range(len(prompts))}
                    lines = [dict(**condition, alpha=alpha, **line)
                             for line in score_complex(level, mode, rules, texts, item_set)]
                    log += lines
                    correct = [line["correct"] for line in lines]
                    rows.append(dict(**condition, n_heads=len(heads), n_prompts=len(correct), n_mean=n_mean,
                                     alpha=alpha, accuracy=sum(correct) / len(correct)))
                print(f"{family} {key} {name} {source} {wording} ({time.time() - start:.0f} s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--family", choices=("simple", "simple-wrong", *COMPLEX_FAMILIES), required=True)
    parser.add_argument("--levels", nargs="+", default=None,
                        help="complex task levels to patch (default: all in the study); "
                             "the output file then names them")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="simple tasks to patch (default: all); the output file then ends in --part")
    parser.add_argument("--part", default=None, help="name of this share of the simple tasks")
    args = parser.parse_args()
    gpu.pin(args.gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from lth.model import LM

    lm = LM.load()
    heads_json = json.loads((RESULTS / "heads.json").read_text())
    rows: list[dict] = []
    log: list[dict] = []
    if args.family in COMPLEX_FAMILIES:
        run_fu(lm, heads_json, rows, log, args.family, args.levels)
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