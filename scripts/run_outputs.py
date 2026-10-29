"""Answer evaluation (proposal Sec. 5.5): every E0 prompt answered once, with longer limits, and saved.

Greedy decoding, one scheme per group of tasks, with limits past the source limits so that
longer answers can still be scored; the strict score is later read from the reply cut to the
source limit (``scripts/score_outputs.py``). No readings are taken and ``data/runs`` is not
touched.
- simple: Yang et al.'s raw-text prompts in their batches of 20 (padding masked); a reply
  ends at the shot delimiter ``;``, a newline once the reply has begun, EOS, or
  ``SIMPLE_MAX_TOKENS`` new tokens;
- complex: Fu's runner under every prompt variant, chat template; a reply ends at EOS or
  ``COMPLEX_MAX_TOKENS`` new tokens (no stop at the answer). One reply per distinct prompt,
  in batches for the reasoning tasks, one prompt at a time for the others, as in E0.
Each wording is written to ``data/outputs/{family}/{task}/{wording}.jsonl.gz``; a
wording already there is skipped.

    .venv/bin/python scripts/run_outputs.py --gpu 4 --family simple --tasks antonym
    .venv/bin/python scripts/run_outputs.py --gpu 4 --family complex --levels tapatan/easy
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import gpu, simple  # noqa: E402

SIMPLE_MAX_TOKENS = 32
COMPLEX_MAX_TOKENS = 1024
BATCH = 32  # Fu's runner: prompts per call


def simple_answer(text: str) -> str:
    """The reply up to its first stop: the delimiter ``;``, or a newline once the reply has
    begun (a leading newline is kept, so that it does not leave an empty answer)."""
    begun = len(text) - len(text.lstrip())
    cuts = [i for i in (text.find(";"), text.find("\n", begun)) if i >= 0]
    return text[:min(cuts, default=len(text))]


def stop_reason(reply: list[int], limit: int, stopped: bool) -> str:
    return "stop" if stopped else ("limit" if len(reply) >= limit else "eos")


def run_simple(lm, tasks, limit_items):
    from lth.store import outputs_path, save_outputs

    for task in tasks:
        data = simple.dataset(task)
        for wording in [w for ws in simple.WORDINGS.values() for w in ws] + [simple.ZERO]:
            path = outputs_path("simple", task, wording)
            if path.exists():
                continue
            start = time.time()
            prompts, answers = simple.prompts(task, wording)
            ids = simple.token_ids(lm.tokenizer, prompts)[:limit_items]
            rows = []
            for b in range(0, len(ids), simple.BATCH):
                replies = lm.generate_batch(ids[b:b + simple.BATCH], SIMPLE_MAX_TOKENS,
                                            stop=lambda text: simple_answer(text) != text)
                for i, reply in enumerate(replies, start=b):
                    text = lm.tokenizer.decode(reply, skip_special_tokens=True)
                    answer = simple_answer(text)
                    rows.append({"item": str(i), "prompt": prompts[i], "input": data[i]["input"],
                                 "reference": answers[i], "reply_ids": reply, "reply_text": text,
                                 "answer_text": answer, "n_tokens": len(reply),
                                 "stop": stop_reason(reply, SIMPLE_MAX_TOKENS, answer != text)})
            save_outputs(path, rows)
            print(f"simple {task} {wording}: {len(rows)} replies, "
                  f"limit {sum(r['stop'] == 'limit' for r in rows)} ({time.time() - start:.0f} s)", flush=True)


def run_complex(lm, levels, variant_names, n_test):
    from lth import fu
    from lth.store import outputs_path, save_outputs
    from run_complex import variants

    for key in levels:
        level = fu.TASK_LEVELS[key]
        reasoning = level.task in fu.REASONING_TASKS
        limit = COMPLEX_MAX_TOKENS
        for name, mode, rules in variants(level):
            if variant_names and name not in variant_names:
                continue
            path = outputs_path("complex", key, name)
            if path.exists():
                continue
            start = time.time()
            stash: dict[str, list[int]] = {}

            def answer(prompts: list[str]) -> list[str]:
                # One reply per distinct prompt (Tapatan's sample repeats boards). Generated as
                # in E0, so that the reply cut to the source limit is E0's reply up to batch
                # numerics: the reasoning tasks in batches, the others one prompt at a time.
                new = [p for p in dict.fromkeys(prompts) if p not in stash]
                ids = {p: lm.encode_chat(p) for p in new}
                if reasoning:
                    for rows_ in lm.token_batches([len(ids[p]) for p in new], limit):
                        for r, reply in zip(rows_, lm.generate_batch([ids[new[r]] for r in rows_], limit)):
                            stash[new[r]] = reply
                else:
                    for p in new:
                        stash[p] = lm.generate(ids[p], limit, token_ids=True)[0][0]
                return [lm.decode(stash[p]) for p in prompts]

            with tempfile.TemporaryDirectory() as scratch:
                results = fu.run(level, mode, answer, Path(scratch), rules=rules, n_test=n_test,
                                 batch=BATCH, reasoning=reasoning)
            rows = []
            for result in results:
                reply = stash[result["prompt"]]
                source = {k: v for k, v in result.items() if k not in ("prompt", "raw_response")}
                rows.append({"item": fu.item_id(result), "prompt": result["prompt"], "mode": mode,
                             "reference": fu.answer(level, result)[0], "reply_ids": reply,
                             "reply_text": result["raw_response"], "n_tokens": len(reply),
                             "stop": stop_reason(reply, limit, False), "source": source})
            save_outputs(path, rows)
            tokens = sorted(r["n_tokens"] for r in rows)
            print(f"complex {key} {name}: {len(rows)} replies, tokens median {tokens[len(tokens) // 2]} "
                  f"max {tokens[-1]}, limit {sum(r['stop'] == 'limit' for r in rows)} "
                  f"({time.time() - start:.0f} s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--family", choices=("simple", "complex"), required=True)
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS), help="simple tasks")
    parser.add_argument("--levels", nargs="+", default=None, help="complex task levels")
    parser.add_argument("--variants", nargs="+", default=None, help="complex prompt variants only")
    parser.add_argument("--limit", type=int, default=None, help="items per wording (smoke runs)")
    args = parser.parse_args()
    gpu.pin(args.gpu)

    from lth import fu
    from lth.model import LM

    lm = LM.load()
    if args.family == "simple":
        run_simple(lm, args.tasks, args.limit)
    else:
        run_complex(lm, args.levels or list(fu.STUDY_LEVELS), args.variants, args.limit)


if __name__ == "__main__":
    main()
