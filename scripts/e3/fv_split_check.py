"""Davidson et al.'s FV split into its generic heads and its task heads, in check 1.

The 30-head FV of ``results/davidson_fv_all14_30`` and parts of it, each a function vector (the
sum of the heads' task-mean outputs through W_O, ``davidson_fv.function_vector``) added to the
residual stream after layer 11 at the last prompt token, exactly as ``scripts/davidson_fv.py``'s
check 1 (same zero-shot prompts, items, replies of at most 32 tokens, flexible score):

- ``all30``: every FV head (its rows must equal the saved run's at the same alphas);
- ``generic6``: the six FV heads whose J-head tokens are alike across tasks
  (``scripts/e3/fv_generic_heads.py``): L13H31, L14H11, L14H29, L15H5, L18H22, L31H14;
- ``task24``: the other 24;
- ``drop6_r{0,1,2}``: the 30 minus six of the 24 non-generic heads, drawn at random (seeds 0-2).

Alphas 0, 1, 2, 3, 5, 8 (1 = Davidson et al.'s; 0 = unpatched). Task means for every set;
the control mean (``davidson_fv.py``'s) for all30 and generic6. Besides accuracy, the shape of
each reply: words in the answer (``run_outputs.simple_answer``), and whether the reply stops at
a line break within 32 tokens.

Writes ``results/davidson_fv_split/patching_simple.part-<part>.csv`` and
``answers/patching_simple.part-<part>.jsonl.gz`` (every reply).

    .venv/bin/python scripts/e3/fv_split_check.py --gpu 2 --tasks antonym synonym --part a
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from lth import RESULTS, gpu, simple  # noqa: E402

GENERIC = [(13, 31), (14, 11), (14, 29), (15, 5), (18, 22), (31, 14)]
ALPHAS = [0, 1, 2, 3, 5, 8]
RUN = RESULTS / "davidson_fv_all14_30"
OUT = RESULTS / "davidson_fv_split"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--part", required=True)
    args = parser.parse_args()
    gpu.pin(args.gpu)
    import numpy as np

    from davidson_fv import INSTRUCTIONS, N_MEAN, function_vector, generate_fv
    from davidson_fv_diff import task_mean
    from patch import PATCH_ROWS, simple_mean
    from run_outputs import SIMPLE_MAX_TOKENS, simple_answer

    from lth.evaluation import flexible_simple
    from lth.model import LM
    from lth.store import load, readings_dir

    plan = json.loads((RUN / "items.json").read_text())
    fv_heads = [(l, h) for l, h, _ in json.loads((RUN / "heads.json").read_text())["heads"]]
    assert set(GENERIC) <= set(fv_heads)
    rest = [x for x in fv_heads if x not in GENERIC]
    sets = {"all30": fv_heads, "generic6": GENERIC, "task24": rest}
    for i in range(3):
        drop = {rest[j] for j in np.random.default_rng(i).choice(len(rest), 6, replace=False)}
        sets[f"drop6_r{i}"] = [x for x in fv_heads if x not in drop]
    with_control = ("all30", "generic6")

    lm = LM.load()
    tok = lm.tokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "answers").mkdir(exist_ok=True)
    (OUT / "sets.json").write_text(json.dumps({k: [list(x) for x in v] for k, v in sets.items()}, indent=1))

    def records(task, wording):
        return load(readings_dir("simple", task), wording)["records"]

    def stop(text):
        return simple_answer(text) != text

    rows, log = [], []
    for task in args.tasks:
        fixed = plan[task]["fixed"]
        prompts, answers = simple.prompts(task, simple.ZERO)
        patched = [simple.token_ids(tok, prompts)[i] for i in fixed]
        means = {"task": task_mean(lm, task, plan)}
        control = simple.CONTROL[task]
        items = simple.correct_items(records(control, simple.CONTROL_ITEMS["rule"]))
        ids = simple.token_ids(tok, simple.prompts(control, simple.CONTROL_PROMPTS)[0])
        means["control"] = simple_mean(lm, ids, items)
        for name, heads in sets.items():
            for source in ("task", "control") if name in with_control else ("task",):
                fv = function_vector(lm, heads, means[source])
                replies = generate_fv(lm, patched, fv, ALPHAS, SIMPLE_MAX_TOKENS, PATCH_ROWS["simple"], stop)
                for alpha in ALPHAS:
                    hits = words = stops = 0
                    for p, i in enumerate(fixed):
                        text = tok.decode(replies[p, alpha], skip_special_tokens=True)
                        answer = simple_answer(text)
                        correct = flexible_simple({"answer_text": answer, "reference": answers[i]})
                        n_words = len(answer.split())
                        stopped = answer != text
                        hits += correct; words += n_words; stops += stopped
                        log.append(dict(task=task, set=name, source=source, alpha=alpha, item=str(i),
                                        expected=answers[i], predicted=answer, correct=correct,
                                        words=n_words, stopped=stopped, reply=text))
                    n = len(fixed)
                    rows.append(dict(task=task, set=name, source=source, n_heads=len(heads), alpha=alpha,
                                     n_prompts=n, accuracy=hits / n, mean_words=words / n, stopped=stops / n))
        print(f"{task} done", flush=True)
    with (OUT / f"patching_simple.part-{args.part}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with gzip.open(OUT / "answers" / f"patching_simple.part-{args.part}.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in log)
    del INSTRUCTIONS, N_MEAN


if __name__ == "__main__":
    main()
