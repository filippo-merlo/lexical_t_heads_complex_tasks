"""Reasoning mode (proposal Sec. 5.3): E0 and the head readings for the complex tasks on
which the model does not answer straight away (``lth.fu.REASONING_TASKS``).

Each task level runs through Fu's runner as in ``run_complex.py``, under the same prompt
variants, but the model may reason: greedy decoding under the chat template up to 6,000
new tokens, replies generated in batches. Every variant uses the same fixed sample of
``--n-items`` items (Fu's sampler and seed). A prompt is in P+ when its final answer is
correct, whatever precedes it. For a P+ prompt, one unpadded forward pass over the prompt
and the model's own reply gives ``z`` at two positions, kept as described in
``lth.store``:
- the last prompt token (files ``{variant}``);
- the pre-answer position, the token after which the answer's first token is written
  (files ``{variant}@pre``; ``lth.fu.answer_start`` says where the answer begins).
The first ``--n-fix`` items in P+ under the original rules prompt are the ones patched in
the causal check. A variant already on disk is skipped.

    .venv/bin/python scripts/run_reasoning.py --gpu 3 --levels operator_func/easy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import gpu  # noqa: E402
from run_complex import BARE, FIXING, variants  # noqa: E402

N_ITEMS = 100
N_FIX = 25
BATCH = 32
PRE = "@pre"


def first_answer_token(tokenizer, reply: list[int], offset: int) -> int:
    """Index of the reply token that holds character ``offset`` of the decoded reply."""
    low, high = 1, len(reply)
    while low < high:  # smallest k whose decoded prefix reaches past offset
        mid = (low + high) // 2
        if len(tokenizer.decode(reply[:mid], skip_special_tokens=True)) > offset:
            high = mid
        else:
            low = mid + 1
    return low - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--levels", nargs="+", default=None)
    parser.add_argument("--n-items", type=int, default=N_ITEMS)
    parser.add_argument("--n-fix", type=int, default=N_FIX)
    args = parser.parse_args()
    gpu.pin(args.gpu)

    from lth import fu
    from lth.model import LM
    from lth.store import Collector, done, fixed_items, run_dir

    lm = LM.load()
    levels = args.levels or [k for k, l in fu.TASK_LEVELS.items() if l.task in fu.REASONING_TASKS]
    for key in levels:
        level = fu.TASK_LEVELS[key]
        directory = run_dir("reasoning", key)
        fixed = fixed_items(directory, FIXING) if done(directory, FIXING) else None
        for name, mode, rules in variants(level):
            if done(directory, name) and (name == BARE or done(directory, name + PRE)):
                continue
            start = time.time()
            stash = {}

            def answer(prompts: list[str]) -> list[str]:
                ids = [lm.encode_chat(p) for p in prompts]
                replies = lm.generate_batch(ids, fu.REASONING_MAX_TOKENS)
                stash.update((p, (i, r)) for p, i, r in zip(prompts, ids, replies))
                return [lm.decode(r) for r in replies]

            results = fu.run(level, mode, answer, directory / "fu_out" / name, rules=rules,
                             n_test=args.n_items, batch=BATCH, reasoning=True)
            options = dict(fixed=fixed, fixing=name == FIXING, require_straight=False, fix_cap=args.n_fix)
            last, pre = Collector(lm, **options), Collector(lm, **options)
            for result in results:
                item = fu.item_id(result)
                correct, straight = fu.score(level, result, reasoning=True)
                prompt_ids, reply = stash[result["prompt"]]
                record = {"correct": correct, "straight": straight, "response": result["raw_response"],
                          "reply_tokens": len(reply)}
                if name == BARE:
                    last.records.append({"item": item, **record})
                    continue
                if not (last.wants(item, correct=correct) or pre.wants(item, correct=correct)):
                    last.add(item, None, **record)
                    pre.add(item, None, **record)
                    continue
                text = lm.tokenizer.decode(reply, skip_special_tokens=True)
                k = first_answer_token(lm.tokenizer, reply, fu.answer_start(level, text))
                sequence = prompt_ids + reply[:k]
                z_last, z_pre = lm.states(sequence, [len(prompt_ids) - 1, len(sequence) - 1])
                last.add(item, z_last, **record)
                pre.add(item, z_pre, pre_text=lm.tokenizer.decode(reply[:k], skip_special_tokens=True), **record)
            last.save(directory, name)
            if name != BARE:
                pre.save(directory, name + PRE)
            if name == FIXING:
                fixed = set(last.fixed)
            records = last.records
            tokens = sorted(r["reply_tokens"] for r in records)
            print(f"{key} {name}: accuracy {sum(r['correct'] for r in records) / len(records):.3f}, "
                  f"straight {sum(r['straight'] for r in records) / len(records):.3f} over {len(records)}, "
                  f"reply tokens median {tokens[len(tokens) // 2]} max {tokens[-1]}, "
                  f"P+ {len(last.id_items)}, mean set {len(last.mean_items)} ({time.time() - start:.0f} s)",
                  flush=True)


if __name__ == "__main__":
    main()
