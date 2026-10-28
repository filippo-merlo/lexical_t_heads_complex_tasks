"""E0 and the head readings for the simple tasks of Yang et al. (2026), as their code runs them.

Every item of every task is run under each wording (instruction templates 0-4, 1/5/10/20/30
shots) and without task information (``" query:"``), and scored by its greedy first token
(their ``behavior_variance.py``), in their batches of 20 but with the padding masked
(``lth.simple``). The first 100 correct prompts of each
wording are then run again, in batches of 20, and every head's top-10 tokens are kept
under both readouts (their ``get_save_MAPS_scores_across_tasks_nnsight``, plus J-head).
For the 1-shot wording these 100 items are also the patched items of the causal check
(``fixed``). A wording already on disk is skipped.

    .venv/bin/python scripts/run_simple.py --gpu 0 --tasks antonym synonym
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import gpu, simple  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS))
    parser.add_argument("--limit", type=int, default=None, help="items per task (smoke runs)")
    args = parser.parse_args()
    gpu.pin(args.gpu)

    from lth.model import LM
    from lth.store import done, run_dir, save

    lm = LM.load()
    for task in args.tasks:
        directory = run_dir("simple", task)
        for wording in [w for ws in simple.WORDINGS.values() for w in ws] + [simple.ZERO]:
            if done(directory, wording):
                continue
            start = time.time()
            prompts, answers = simple.prompts(task, wording)
            ids = simple.token_ids(lm.tokenizer, prompts)[: args.limit]
            records = []
            for b in range(0, len(ids), simple.BATCH):
                tokens, _ = lm.next_tokens_batch(ids[b:b + simple.BATCH])
                for i, token in enumerate(tokens, start=b):
                    records.append({"item": str(i), "correct": token == lm.first_token(answers[i]),
                                    "straight": True, "prediction": token})
            id_items, top = [], []
            if wording != simple.ZERO:
                id_items = simple.correct_items(records)[: simple.CAP]
                for b in range(0, len(id_items), simple.BATCH):
                    _, z = lm.next_tokens_batch([ids[i] for i in id_items[b:b + simple.BATCH]])
                    top += [lm.top_tokens(row) for row in z]
            save(directory, wording, records, [str(i) for i in id_items], top,
                 fixed=[str(i) for i in id_items] if wording == simple.FIXING else [])
            accuracy = sum(r["correct"] for r in records) / len(records)
            print(f"{task} {wording}: accuracy {accuracy:.3f} over {len(records)}, P+ {len(id_items)} "
                  f"({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
