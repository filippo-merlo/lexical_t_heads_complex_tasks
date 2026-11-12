"""E14 step 1 (GPU): E0 on the 30 new rule templates of every simple task (``lth.simple.E4B_WORDINGS``).

As ``scripts/run_outputs.py`` + ``score_outputs.py`` + ``build_readings.py`` for Yang et al.'s
templates (greedy, batches of 20 with the padding masked, a reply ends at ``;``, a newline once
begun, EOS or 32 tokens; flexible and strict scores), on fewer items: the first ``N_PER_HALF`` = 500
items (in file order) of each half of ``results/heldout/split.json``. Every later step takes at most
100 correct prompts of a half, the first in file order, so the cap changes nothing where a template
has 100 correct among its first 500.

Writes per task and wording ``data/outputs/simple/<task>/<wording>.jsonl.gz`` (replies),
``<wording>.scored.jsonl.gz`` (scores) and the records ``data/readings/simple/<task>/<wording>.{json,npz}``
(no identification readings: the half readings are ``scripts/heldout_readings.py --wordings e4b``).
A wording already there is skipped.

    .venv/bin/python scripts/e14_outputs.py --gpu 0 --tasks antonym synonym
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import RESULTS, gpu, simple  # noqa: E402

N_PER_HALF = 500


def items_of(task: str) -> list[int]:
    split = json.loads((RESULTS / "heldout" / "split.json").read_text())[task]
    return sorted(i for half in ("A", "B") for i in sorted(split[half])[:N_PER_HALF])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS))
    args = parser.parse_args()
    gpu.pin(args.gpu)

    from run_outputs import SIMPLE_MAX_TOKENS, simple_answer, stop_reason

    from lth import evaluation
    from lth.model import LM
    from lth.store import done, outputs_path, readings_dir, save, save_outputs

    lm = LM.load()
    for task in args.tasks:
        data = simple.dataset(task)
        items = items_of(task)
        for wording in simple.E4B_WORDINGS:
            path = outputs_path("simple", task, wording)
            directory = readings_dir("simple", task)
            if path.exists() and done(directory, wording):
                continue
            start = time.time()
            prompts, answers = simple.prompts(task, wording)
            ids = simple.token_ids(lm.tokenizer, [prompts[i] for i in items])
            rows = []
            for b in range(0, len(ids), simple.BATCH):
                replies = lm.generate_batch(ids[b:b + simple.BATCH], SIMPLE_MAX_TOKENS,
                                            stop=lambda text: simple_answer(text) != text)
                for i, reply in zip(items[b:b + simple.BATCH], replies):
                    text = lm.tokenizer.decode(reply, skip_special_tokens=True)
                    answer = simple_answer(text)
                    rows.append({"item": str(i), "prompt": prompts[i], "input": data[i]["input"],
                                 "reference": answers[i], "reply_ids": reply, "reply_text": text,
                                 "answer_text": answer, "n_tokens": len(reply),
                                 "stop": stop_reason(reply, SIMPLE_MAX_TOKENS, answer != text)})
            save_outputs(path, rows)
            scored = [{"item": r["item"], **evaluation.strict_simple(r), "flexible": evaluation.flexible_simple(r)}
                      for r in rows]
            for s in scored:
                s["study_strict"] = s["strict"]
            save_outputs(path.with_name(f"{wording}.scored.jsonl.gz"), scored)
            records = [{"item": r["item"], "correct": bool(s["flexible"]), "first_token": bool(s["strict"]),
                        "answer": r["answer_text"]} for r, s in zip(rows, scored)]
            save(directory, wording, records, [], [], fixed=[])
            accuracy = sum(r["correct"] for r in records) / len(records)
            print(f"simple {task} {wording}: {len(rows)} replies, flexible {accuracy:.3f}, "
                  f"limit {sum(r['stop'] == 'limit' for r in rows)} ({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
