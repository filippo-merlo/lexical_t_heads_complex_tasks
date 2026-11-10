"""E11: Davidson et al.'s own intervention on E4's held-out items, in both checks.

The function vector of ``scripts/davidson_fv.py`` (the sum of the heads' task-mean outputs through
W_O, added to the residual stream after layer 11 at the last prompt token), with E4's fold files
(``results/heldout/fold{f}/fv/``: heads chosen and means taken on the choosing half):

- ``fv30``: the fold's 30 FV heads (E4's "Davidson FV-30"; check 1 already run there at 15 alphas);
- ``fv30_nogen``: the same without the FV generic heads it holds (``e10_sets.GENERIC6``), the
  held-out counterpart of the FV split's ``task24``.

Means as ``davidson_fv.py``: the task mean over the fold's mean items (25 correct prompts of each
of inst1-inst4, choosing half); the control mean over the control task's 1-shot prompts at its
choosing-half items correct under inst1. Check 1 (``simple``): the fold's patched items (first 100
of the testing half answered correctly 1-shot), zero-shot prompts. Check 2 (``simple-wrong``): the
first 100 of the testing half answered wrongly 1-shot, their 1-shot prompts (``scripts/patch.py``'s
choice). Alpha 1 and 2; replies of at most 16 tokens, flexible score; every reply saved.

Writes ``results/e11_complete/fold{f}/fv_residual.csv`` and ``answers/fv_residual.jsonl.gz``.

    .venv/bin/python scripts/e11_fv_residual.py --gpu 0 --fold 1
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
    parser.add_argument("--out", type=Path, default=None, help="output folder (default results/e11_complete/fold<f>)")
    args = parser.parse_args()
    gpu.pin(args.gpu)

    from davidson_fv import LAYER, function_vector, generate_fv
    from davidson_fv_diff import task_mean
    from e10_sets import GENERIC6
    from patch import PATCH_ROWS, simple_mean
    from run_outputs import simple_answer

    from lth.evaluation import flexible_simple
    from lth.model import LM
    from lth.store import load, readings_dir

    fv_dir = RESULTS / "heldout" / f"fold{args.fold}" / "fv"
    plan = json.loads((fv_dir / "items.json").read_text())
    info = json.loads((fv_dir / "heads.json").read_text())
    assert info["layer"] == LAYER
    fv30 = [(l, h) for l, h, _ in info["heads"]]
    sets = {"fv30": fv30, "fv30_nogen": [x for x in fv30 if x not in GENERIC6]}
    half = "A" if args.fold == 1 else "B"
    split = json.loads((RESULTS / "heldout" / "split.json").read_text())
    out = args.out or RESULTS / "e11_complete" / f"fold{args.fold}"
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
                 if i in set(split[control][half])]  # the control task's own choosing half
        ids = simple.token_ids(tok, simple.prompts(control, simple.CONTROL_PROMPTS)[0])
        means["control"] = simple_mean(lm, ids, items)
        one_shot = records(task, simple.FIXING)
        checks = {
            "simple": (simple.ZERO, plan[task]["fixed"]),
            "simple-wrong": (simple.FIXING, [i for i in (int(r["item"]) for r in one_shot if not r["correct"])
                                             if i not in chosen][: simple.CAP]),
        }
        assert plan[task]["fixed"] == [i for i in simple.correct_items(one_shot) if i not in chosen][: simple.CAP]
        for family, (target, fixed) in checks.items():
            prompts, answers = simple.prompts(task, target)
            patched = [simple.token_ids(tok, prompts)[i] for i in fixed]
            unpatched = {int(r["item"]): r for r in records(task, target)}
            rows.append(dict(family=family, task=task, set="", source="none", n_heads=0, alpha="",
                             n_prompts=len(fixed), accuracy=sum(unpatched[i]["correct"] for i in fixed) / len(fixed)))
            first = [lm.first_token(answers[i]) for i in fixed]
            for name, heads in sets.items():
                for source, mean in means.items():
                    fv = function_vector(lm, heads, mean)
                    replies = generate_fv(lm, patched, fv, list(ALPHAS), MAX_NEW_TOKENS, PATCH_ROWS["simple"], stop)
                    for alpha in ALPHAS:
                        hits = tokens = 0
                        for p, i in enumerate(fixed):
                            reply = replies[p, alpha]
                            text = tok.decode(reply, skip_special_tokens=True)
                            answer = simple_answer(text)
                            correct = flexible_simple({"answer_text": answer, "reference": answers[i]})
                            hits += correct
                            tokens += reply[:1] == [first[p]]
                            log.append(dict(family=family, task=task, set=name, source=source, alpha=alpha,
                                            item=str(i), expected=answers[i], predicted=answer, correct=correct,
                                            reply=text))
                        rows.append(dict(family=family, task=task, set=name, source=source, n_heads=len(heads),
                                         alpha=alpha, n_prompts=len(fixed), accuracy=hits / len(fixed),
                                         first_token=tokens / len(fixed)))
        print(f"{task} done", flush=True)
    name = "fv_residual" + ("" if args.tasks == list(simple.TASKS) else ".part-" + "-".join(args.tasks))
    with (out / f"{name}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["family", "task", "set", "source", "n_heads", "alpha", "n_prompts",
                                           "accuracy", "first_token"])
        w.writeheader()
        w.writerows(rows)
    with gzip.open(out / "answers" / f"{name}.jsonl.gz", "wt") as fh:
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in log)


if __name__ == "__main__":
    main()
