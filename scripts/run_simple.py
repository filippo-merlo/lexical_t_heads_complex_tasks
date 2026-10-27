"""E0 and the head readings for the simple tasks of Yang et al. (2026).

Every item of every task is run alone through the model under each wording (six
instruction templates, 1/5/10/20/30 shots) and without task information (``" query:"``),
and scored by its greedy first token. The 1-shot wording runs first: its first 100
correct items are the ones patched in the causal check. Readings and mean sums are kept
as described in ``lth.store``. A wording already on disk is skipped.

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
    from lth.store import Collector, done, fixed_items, run_dir

    lm = LM.load()
    order = [simple.FIXING] + [w for ws in simple.WORDINGS.values() for w in ws if w != simple.FIXING]
    for task in args.tasks:
        directory = run_dir("simple", task)
        fixed = fixed_items(directory, simple.FIXING) if done(directory, simple.FIXING) else None
        for wording in order + [simple.ZERO]:
            if done(directory, wording):
                continue
            start = time.time()
            prompts, answers = simple.prompts(task, wording)
            collector = Collector(lm, fixed=fixed, fixing=wording == simple.FIXING)
            for i, (prompt, answer) in enumerate(list(zip(prompts, answers))[: args.limit]):
                token, z = lm.next_token(lm.encode_raw(prompt))
                prediction, target = token.item(), lm.first_token(answer)
                if wording == simple.ZERO:
                    collector.records.append({"item": str(i), "correct": prediction == target,
                                              "straight": True, "prediction": prediction})
                else:
                    collector.add(str(i), z, correct=prediction == target, prediction=prediction)
            collector.save(directory, wording)
            if wording == simple.FIXING:
                fixed = set(collector.fixed)
            accuracy = sum(r["correct"] for r in collector.records) / len(collector.records)
            print(f"{task} {wording}: accuracy {accuracy:.3f} over {len(collector.records)}, "
                  f"P+ {len(collector.id_items)}, mean set {len(collector.mean_items)} "
                  f"({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
