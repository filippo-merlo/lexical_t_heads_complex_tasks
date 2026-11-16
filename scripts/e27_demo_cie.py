"""E27: Davidson et al.'s demonstration causal score (Todd et al.'s CIE) on our 14 simple tasks.

Davidson, Gureckis, Lake and Williams (2025, arXiv 2505.12075, Sec. 2, 2.2, 3.3) compare the
20 heads with the largest causal indirect effect (CIE) when the task is given by instructions
with those found when it is given by demonstrations, and find few shared (4 of 20 on
Llama-3.1-8B and -Instruct). The instruction CIE is ``scripts/davidson_fv.py``; this script
computes the demonstration CIE as they describe it, following Todd et al. (2024):

- prompts: 10 demonstrations and a query in Todd et al.'s format,
  ``"Q: x1\\nA: y1\\n\\n ... Q: x10\\nA: y10\\n\\nQ: xq\\nA:"`` (BOS first), the demonstrations
  drawn at random among the other items, the answer's first token that of ``" " + yq``;
- correct prompts: the model's next token is that first token (Todd et al.'s criterion);
- mean: each head's ``z`` at the last token over the first 100 correct prompts;
- CIE: 25 further correct prompts with the demonstrations' answers shuffled among them; a
  head's CIE is the gain in the probability of the query's answer when that head alone is set
  to its mean (``davidson_fv.cie``). With fewer than 125 correct prompts the CIE takes the
  last 25 and the mean the rest, as in ``davidson_fv.py``.

The items are those the instruction CIE could use: the check-1 items (the first 100 answered
correctly 1-shot) are left out as queries and demonstrations, and with ``--half`` only the
choosing half's items are used (``results/heldout/split.json``). Writes
``results/e27_demo_cie/{all,A,B}/``: ``cie.npz`` (one ``[n_layers, n_heads]`` array per task),
``items.json`` (queries, demonstrations and shuffled answers of every mean and CIE prompt),
``accuracy.json`` (10-shot first-token accuracy of the prompts tried), and
``answers.jsonl.gz`` (every clean prompt tried: expected and predicted first token).

    .venv/bin/python scripts/e27_demo_cie.py --gpu 0 [--half A]
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import RESULTS, gpu, simple  # noqa: E402

N_SHOT = 10  # Davidson et al., Sec. 2.2: shuffled 10-shot
N_MEAN = 100
N_CIE = 25
CIE_CHUNK = 5  # CIE prompts per pass (x 32 heads per layer)
SEED = 0
OUT = RESULTS / "e27_demo_cie"


def prompt(data: list[dict], demos: list[int], outputs: list[str], query: int) -> str:
    shots = "".join(f"Q: {data[d]['input']}\nA: {o}\n\n" for d, o in zip(demos, outputs))
    return shots + f"Q: {data[query]['input']}\nA:"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--half", choices=("A", "B"), default=None, help="choosing half (default: all items)")
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS))
    args = parser.parse_args()
    gpu.pin(args.gpu)

    import numpy as np

    from davidson_fv import cie
    from patch import simple_mean

    from lth.model import LM
    from lth.store import load, readings_dir

    halves = json.loads((RESULTS / "heldout" / "split.json").read_text()) if args.half else None
    instruction_run = {None: RESULTS / "davidson_fv_all14_30", "A": RESULTS / "heldout" / "fold1" / "fv",
                       "B": RESULTS / "heldout" / "fold2" / "fv"}[args.half]
    fixed_of = {t: v["fixed"] for t, v in json.loads((instruction_run / "items.json").read_text()).items()}
    out = OUT / (args.half or "all")
    out.mkdir(parents=True, exist_ok=True)
    lm = LM.load()
    bos = lm.tokenizer.bos_token_id

    def ids(text: str) -> list[int]:
        return [bos] + lm.tokenizer(text, add_special_tokens=False).input_ids

    scores, plan, accuracy, log = {}, {}, {}, []
    for task in args.tasks:
        data = simple.dataset(task)
        records = load(readings_dir("simple", task), simple.FIXING)["records"]
        fixed = simple.correct_items(records)
        if halves is not None:
            fixed = [i for i in fixed if i not in set(halves[task][args.half])]
        fixed = fixed[: simple.CAP]
        assert fixed == fixed_of[task], task  # the instruction run's check-1 items
        pool = [i for i in range(len(data)) if i not in set(fixed)
                and (halves is None or i in set(halves[task][args.half]))]
        rng = random.Random(f"{SEED}-{task}-{args.half}")
        queries = rng.sample(pool, len(pool))
        prompts = []  # (query, demos)
        for q in queries:
            prompts.append((q, rng.sample([i for i in pool if i != q], N_SHOT)))

        # Clean 10-shot prompts in batches until N_MEAN + N_CIE are correct.
        correct, tried = [], 0
        for b in range(0, len(prompts), simple.BATCH):
            chunk = prompts[b:b + simple.BATCH]
            rows = [ids(prompt(data, d, [data[i]["output"] for i in d], q)) for q, d in chunk]
            tokens, _ = lm.next_tokens_batch(rows)
            for (q, d), t in zip(chunk, tokens):
                target = lm.first_token(" " + data[q]["output"])
                ok = t == target
                log.append(dict(task=task, item=q, demos=d, expected=lm.tokenizer.decode([target]),
                                predicted=lm.tokenizer.decode([t]), correct=ok))
                correct += [(q, d)] if ok else []
                tried += 1
            if len(correct) >= N_MEAN + N_CIE:
                break
        correct = correct[: N_MEAN + N_CIE]
        if len(correct) >= N_MEAN + N_CIE:
            mean_p, cie_p = correct[:N_MEAN], correct[N_MEAN:]
        else:
            mean_p, cie_p = correct[:-N_CIE], correct[-N_CIE:]
            print(f"{task}: only {len(correct)} correct 10-shot prompts of {tried}", flush=True)
        accuracy[task] = dict(tried=tried, correct=sum(r["correct"] for r in log if r["task"] == task),
                              n_mean=len(mean_p), n_cie=len(cie_p))
        accuracy[task]["accuracy"] = accuracy[task]["correct"] / tried

        mean_rows = [ids(prompt(data, d, [data[i]["output"] for i in d], q)) for q, d in mean_p]
        mean_z = simple_mean(lm, mean_rows, list(range(len(mean_rows))))
        shuffled = []
        for q, d in cie_p:
            outs = [data[i]["output"] for i in d]
            rng.shuffle(outs)
            shuffled.append((q, d, outs))
        rows = [ids(prompt(data, d, outs, q)) for q, d, outs in shuffled]
        targets = [lm.first_token(" " + data[q]["output"]) for q, _, _ in shuffled]
        total = 0
        for b in range(0, len(rows), CIE_CHUNK):
            part = cie(lm, rows[b:b + CIE_CHUNK], targets[b:b + CIE_CHUNK], mean_z).numpy()
            total = total + part * len(rows[b:b + CIE_CHUNK])
        scores[task] = total / len(rows)
        plan[task] = dict(fixed=fixed, mean=[dict(query=q, demos=d) for q, d in mean_p],
                          cie=[dict(query=q, demos=d, shuffled_outputs=o) for q, d, o in shuffled])
        print(f"{task}: 10-shot accuracy {accuracy[task]['accuracy']:.2f} ({tried} tried), "
              f"mean {len(mean_p)}, CIE {len(cie_p)} prompts, done", flush=True)

    np.savez(out / "cie.npz", **scores)
    (out / "items.json").write_text(json.dumps(plan, indent=1))
    (out / "accuracy.json").write_text(json.dumps(accuracy, indent=1))
    with gzip.open(out / "answers.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in log)


if __name__ == "__main__":
    main()
