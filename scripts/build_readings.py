"""Head readings and means of the prompts answered correctly under the flexible score.

The head identification and the causal checks (proposal Secs. 4.4, 5.4) take their prompts
from the saved outputs (``scripts/run_outputs.py``) and their scores from
``scripts/score_outputs.py``: a prompt is in P+ when its reply is right under the flexible
score (no condition that the answer comes first). For every task and wording this writes,
in the format of ``lth.store``, to ``data/readings/{family}/{task}``:
- every item's behaviour (``records``: item, flexible score, strict score, answer read);
- the identification prompts: the first ``CAP`` prompts in P+, with every head's top-10
  tokens under both readouts at the last prompt token;
- simple tasks: for the 1-shot wording these items are the patched items of check 1
  (``fixed``); the means are taken in the causal check, as Yang et al. do;
- complex tasks: the patched items of check 1 are the first ``CAP`` prompts in P+ under the
  original rules prompt (``fixed``); each wording's mean set is its first ``CAP`` prompts in
  P+ whose item is not among them, with the sum of their ``z``.
Simple prompts are run in Yang et al.'s batches of 20 with the padding masked; complex
prompts alone, under the chat template, as their replies were generated. The prompts
without task information (``zero``, ``bare``) keep only their records. A wording already
on disk is skipped.

    .venv/bin/python scripts/build_readings.py --gpu 0 --family simple
    .venv/bin/python scripts/build_readings.py --gpu 2 --family complex --levels tapatan/easy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import gpu, simple  # noqa: E402
from lth.store import CAP, Collector, done, load_outputs, outputs_path, readings_dir, save  # noqa: E402


def scored(family: str, task: str, wording: str) -> tuple[list[dict], dict[str, dict]]:
    """A wording's saved outputs, in file order, and their scores by item."""
    path = outputs_path(family, task, wording)
    scores = load_outputs(path.with_name(f"{wording}.scored.jsonl.gz"))
    return load_outputs(path), {s["item"]: s for s in scores}


def build_simple(lm, tasks) -> None:
    for task in tasks:
        directory = readings_dir("simple", task)
        for wording in [w for ws in simple.WORDINGS.values() for w in ws] + [simple.ZERO]:
            if done(directory, wording):
                continue
            start = time.time()
            rows, scores = scored("simple", task, wording)
            records = [{"item": r["item"], "correct": bool(scores[r["item"]]["flexible"]),
                        "first_token": bool(scores[r["item"]]["strict"]), "answer": r["answer_text"]}
                       for r in rows]
            id_items, top = [], []
            if wording != simple.ZERO:
                id_items = simple.correct_items(records)[:CAP]
                ids = simple.token_ids(lm.tokenizer, [rows[i]["prompt"] for i in id_items])
                for b in range(0, len(ids), simple.BATCH):
                    _, z = lm.next_tokens_batch(ids[b:b + simple.BATCH])
                    top += [lm.top_tokens(row) for row in z]
            save(directory, wording, records, [str(i) for i in id_items], top,
                 fixed=[str(i) for i in id_items] if wording == simple.FIXING else [])
            accuracy = sum(r["correct"] for r in records) / len(records)
            print(f"simple {task} {wording}: flexible {accuracy:.3f}, P+ {len(id_items)} "
                  f"({time.time() - start:.0f} s)", flush=True)


def build_complex(lm, levels, variant_names) -> None:
    from run_complex import BARE, FIXING, variants

    from lth import fu

    for key in levels:
        directory = readings_dir("complex", key)
        _, original = scored("complex", key, FIXING)
        fixed = [item for item, s in original.items() if s["flexible"]][:CAP]
        for name, _, _ in variants(fu.TASK_LEVELS[key]):
            if (variant_names and name not in variant_names) or done(directory, name):
                continue
            start = time.time()
            rows, scores = scored("complex", key, name)
            records = [{"item": r["item"], "correct": bool(scores[r["item"]]["flexible"]),
                        "strict": bool(scores[r["item"]]["study_strict"]),
                        "answer": scores[r["item"]]["flexible_answer"]} for r in rows]
            collector = Collector(lm, fixed=set(fixed), fixing=name == FIXING, require_straight=False)
            if name == BARE:
                collector.records = records
            else:
                for row, record in zip(rows, records):
                    item, correct = record.pop("item"), record.pop("correct")
                    z = lm.next_token(lm.encode_chat(row["prompt"]))[1] if collector.wants(item, correct=correct) \
                        else None
                    collector.add(item, z, correct=correct, **record)
            collector.save(directory, name)
            accuracy = sum(r["correct"] for r in collector.records) / len(collector.records)
            print(f"complex {key} {name}: flexible {accuracy:.3f}, P+ read {len(collector.id_items)}, "
                  f"mean set {len(collector.mean_items)}, fixed {len(collector.fixed)} "
                  f"({time.time() - start:.0f} s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--family", choices=("simple", "complex"), required=True)
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS), help="simple tasks")
    parser.add_argument("--levels", nargs="+", default=None, help="complex task levels")
    parser.add_argument("--variants", nargs="+", default=None, help="complex prompt variants only")
    args = parser.parse_args()
    gpu.pin(args.gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from lth import fu
    from lth.model import LM

    lm = LM.load()
    if args.family == "simple":
        build_simple(lm, args.tasks)
    else:
        build_complex(lm, args.levels or list(fu.STUDY_LEVELS), args.variants)


if __name__ == "__main__":
    main()