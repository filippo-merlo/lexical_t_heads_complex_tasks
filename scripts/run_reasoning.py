"""Reasoning mode (proposal Sec. 5.3): E0 and the head readings for the complex tasks on
which the model does not answer straight away (``lth.fu.REASONING_TASKS``).

Each task level runs through Fu's runner as in ``run_complex.py``, under the same prompt
variants, but the model may reason: greedy decoding under the chat template up to 6,000
new tokens, replies generated in batches. A reply stops once the answer the scoring reads
is complete (``lth.fu.answer_complete``), which leaves every score unchanged. Every variant uses the same fixed sample of
``--n-items`` items (Fu's sampler and seed). A prompt is in P+ when its final answer is
correct, whatever precedes it. For a P+ prompt, one unpadded forward pass over the prompt
and the model's own reply gives ``z`` at two positions, kept as described in
``lth.store``:
- the last prompt token (files ``{variant}``);
- the pre-answer position, the token after which the answer's first token is written
  (files ``{variant}@pre``; ``lth.fu.answer_start`` says where the answer begins).
The first ``--n-fix`` items in P+ under the original rules prompt are the ones patched in
the causal check. A variant already on disk is skipped.

With ``--reuse DIR`` (an earlier runs folder of this family, e.g. a smaller sample), a
prompt answered there gets that reply instead of a new generation (its text re-tokenized),
and the patched items are taken from its original rules prompt until this run has its own,
so the variants of a level can run in parallel. The 300-item sample starts with the
earlier 100 items, with the same prompts.

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

N_ITEMS = 300  # 100 until 2026-10-28; raised to the size of the other tasks
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
    parser.add_argument("--variants", nargs="+", default=None,
                        help="run only these prompt variants (to split a level over GPUs)")
    parser.add_argument("--reuse", type=Path, default=None,
                        help="earlier runs folder whose replies are reused for the same prompts")
    args = parser.parse_args()
    gpu.pin(args.gpu)

    from lth import fu
    from lth.model import LM
    from lth.store import Collector, done, fixed_items, run_dir

    lm = LM.load()
    levels = args.levels or [k for k in fu.STUDY_LEVELS if fu.TASK_LEVELS[k].task in fu.REASONING_TASKS]
    for key in levels:
        level = fu.TASK_LEVELS[key]
        directory = run_dir("reasoning", key)
        earlier = args.reuse / directory.name if args.reuse else None
        fixed = fixed_items(directory, FIXING) if done(directory, FIXING) else None
        if fixed is None and earlier is not None:
            fixed = fixed_items(earlier, FIXING)
        for name, mode, rules in variants(level):
            if args.variants and name not in args.variants:
                continue
            if done(directory, name) and (name == BARE or done(directory, name + PRE)):
                continue
            start = time.time()
            stash = {}
            reused = ({r["prompt"]: r["raw_response"] for r in fu.recorded_results(earlier / "fu_out" / name)}
                      if earlier is not None and (earlier / "fu_out" / name).exists() else {})

            def answer(prompts: list[str]) -> list[str]:
                # One reply per distinct prompt: Tapatan's sample repeats boards, and a batched
                # reply can differ with the batch it is generated in.
                new = [p for p in dict.fromkeys(prompts) if p not in stash and p not in reused]
                ids = {p: lm.encode_chat(p) for p in new}
                for rows in lm.token_batches([len(ids[p]) for p in new], fu.REASONING_MAX_TOKENS):
                    out = lm.generate_batch([ids[new[r]] for r in rows], fu.REASONING_MAX_TOKENS,
                                            stop=lambda text: fu.answer_complete(level, text))
                    for r, reply in zip(rows, out):
                        stash[new[r]] = (ids[new[r]], reply)
                for p in prompts:
                    if p not in stash:
                        stash[p] = (lm.encode_chat(p), lm.tokenizer(reused[p], add_special_tokens=False).input_ids)
                return [reused[p] if p in reused else lm.decode(stash[p][1]) for p in prompts]

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
                if fixed is not None and set(last.fixed) != fixed:
                    raise RuntimeError(f"{key}: the patched items differ from the earlier run's")
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
