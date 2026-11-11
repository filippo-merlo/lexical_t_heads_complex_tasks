"""E13: Davidson et al.'s own intervention (summed function vector) for the shared causal sets of the
capped grid, check 1 only.

As ``scripts/e11_fv_residual.py`` (the sum of the heads' task-mean outputs through W_O, added to the
residual stream after layer 11 at the last prompt token; the task mean over the fold's mean items,
25 correct prompts of each of inst1-inst4 of the choosing half; the control mean over the control
task's 1-shot prompts at its choosing-half items correct under inst1), with the sets of
``results/e13_capped/fold{f}/fv_sets.json``: the top N heads by Davidson et al.'s score averaged over
the 14 tasks (``fvsum{N}``) and the same ranking without the generic heads (``fvsum{N}_nogen``),
N = 20, 40, 60, 80. Check 1: the fold's patched items, zero-shot prompts. Alpha 1 and 2; replies of
at most 16 tokens, flexible score; every reply saved.

Writes ``results/e13_capped/fold{f}/fv_residual.part-<tasks>.csv`` and ``answers/``.

    CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python scripts/e13_fv_residual.py --gpu 0 --fold 1 --tasks antonym
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import RESULTS, gpu, simple  # noqa: E402

ALPHAS = (1.0, 2.0)
MAX_NEW_TOKENS = 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS))
    args = parser.parse_args()
    gpu.pin(args.gpu)

    from davidson_fv import LAYER, function_vector, generate_fv
    from davidson_fv_diff import task_mean
    from patch import PATCH_ROWS, simple_mean
    from run_outputs import simple_answer

    from lth.evaluation import flexible_simple
    from lth.model import LM
    from lth.store import load, readings_dir

    fv_dir = RESULTS / "heldout" / f"fold{args.fold}" / "fv"
    plan = json.loads((fv_dir / "items.json").read_text())
    assert json.loads((fv_dir / "heads.json").read_text())["layer"] == LAYER
    out = RESULTS / "e13_capped" / f"fold{args.fold}"
    sets = {k: [tuple(h) for h in v] for k, v in json.loads((out / "fv_sets.json").read_text())["sets"].items()}
    half = "A" if args.fold == 1 else "B"
    split = json.loads((RESULTS / "heldout" / "split.json").read_text())
    (out / "answers").mkdir(parents=True, exist_ok=True)

    def records(task, wording):
        return load(readings_dir("simple", task), wording)["records"]

    def stop(text):
        return simple_answer(text) != text

    lm = LM.load()
    tok = lm.tokenizer
    rows, log = [], []
    for task in args.tasks:
        chosen = set(split[task][half])
        means = {"task": task_mean(lm, task, plan)}
        control = simple.CONTROL[task]
        items = [i for i in simple.correct_items(records(control, simple.CONTROL_ITEMS["rule"]))
                 if i in set(split[control][half])]
        ids = simple.token_ids(tok, simple.prompts(control, simple.CONTROL_PROMPTS)[0])
        means["control"] = simple_mean(lm, ids, items)
        one_shot = records(task, simple.FIXING)
        fixed = plan[task]["fixed"]
        assert fixed == [i for i in simple.correct_items(one_shot) if i not in chosen][: simple.CAP]
        prompts, answers = simple.prompts(task, simple.ZERO)
        patched = [simple.token_ids(tok, prompts)[i] for i in fixed]
        unpatched = {int(r["item"]): r for r in records(task, simple.ZERO)}
        rows.append(dict(family="simple", task=task, set="", source="none", n_heads=0, alpha="",
                         n_prompts=len(fixed), accuracy=sum(unpatched[i]["correct"] for i in fixed) / len(fixed)))
        for name, heads in sets.items():
            for source, mean in means.items():
                fv = function_vector(lm, heads, mean)
                replies = generate_fv(lm, patched, fv, list(ALPHAS), MAX_NEW_TOKENS, PATCH_ROWS["simple"], stop)
                for alpha in ALPHAS:
                    hits = 0
                    for p, i in enumerate(fixed):
                        text = tok.decode(replies[p, alpha], skip_special_tokens=True)
                        answer = simple_answer(text)
                        correct = flexible_simple({"answer_text": answer, "reference": answers[i]})
                        hits += correct
                        log.append(dict(family="simple", task=task, set=name, source=source, alpha=alpha,
                                        item=str(i), expected=answers[i], predicted=answer, correct=correct,
                                        reply=text))
                    rows.append(dict(family="simple", task=task, set=name, source=source, n_heads=len(heads),
                                     alpha=alpha, n_prompts=len(fixed), accuracy=hits / len(fixed)))
        print(f"{task} done", flush=True)
    name = "fv_residual.part-" + "-".join(args.tasks)
    with (out / f"{name}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["family", "task", "set", "source", "n_heads", "alpha", "n_prompts",
                                           "accuracy"])
        w.writeheader()
        w.writerows(rows)
    with gzip.open(out / "answers" / f"{name}.jsonl.gz", "wt") as fh:
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in log)


if __name__ == "__main__":
    main()
