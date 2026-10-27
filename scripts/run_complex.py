"""E0 and the head readings for the complex tasks of Fu et al. (2026).

Each task level runs through Fu's own runner (``lth.fu``) with greedy decoding under
every prompt variant:
- ``orig``: their rules-only prompt; it runs first, and its first 100 items answered
  correctly and straight away are the ones patched in the causal check;
- ``w1``..``w6``: our checked rule wordings (``materials/complex/wordings.json``), once
  they exist;
- ``examples`` and ``combined``: their example-based and combined prompts;
- ``bare``: the input block alone, the prompt without task information.
``z`` is read at the last prompt token during the prompt's forward pass. Readings and mean
sums are kept as described in ``lth.store``. A variant already on disk is skipped.

    .venv/bin/python scripts/run_complex.py --gpu 3 --levels noun_class_agreement/easy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import MATERIALS, gpu  # noqa: E402

FIXING = "orig"
BARE = "bare"


def variants(level) -> list[tuple[str, str, str | None]]:
    """``(name, mode, rule template)`` for every prompt variant available for ``level``."""
    from lth import fu

    original = fu.ORIGINALS[level.original]
    template = fu.original_template(original)
    out = [(FIXING, "rules", None)]
    path = MATERIALS / "complex" / "wordings.json"
    if path.exists():
        wordings = json.loads(path.read_text())[level.original]
        out += [(f"w{i}", "rules", fu.template_from_display(text, original.placeholders))
                for i, text in enumerate(wordings, 1)]
    out += [("examples", "examples", None), ("combined", "combined", None),
            (BARE, "rules", fu.input_block(template, original))]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--levels", nargs="+", default=None)
    parser.add_argument("--n-test", type=int, default=None, help="items per level (smoke runs)")
    args = parser.parse_args()
    gpu.pin(args.gpu)

    from lth import fu
    from lth.model import LM
    from lth.store import Collector, done, fixed_items, run_dir

    lm = LM.load()
    for key in args.levels or [k for k in fu.STUDY_LEVELS if fu.TASK_LEVELS[k].task not in fu.REASONING_TASKS]:
        level = fu.TASK_LEVELS[key]
        directory = run_dir("complex", key)
        fixed = fixed_items(directory, FIXING) if done(directory, FIXING) else None
        for name, mode, rules in variants(level):
            if done(directory, name):
                continue
            start = time.time()
            stash = {}

            def answer(prompts: list[str]) -> list[str]:
                out = []
                for prompt in prompts:
                    replies, z = lm.generate(lm.encode_chat(prompt), fu.MAX_TOKENS[level.task])
                    stash[prompt] = z[0].cpu()
                    out.append(replies[0])
                return out

            results = fu.run(level, mode, answer, directory / "fu_out" / name, rules=rules,
                             n_test=args.n_test)
            collector = Collector(lm, fixed=fixed, fixing=name == FIXING)
            for result in results:
                correct, straight = fu.score(level, result)
                record = {"correct": correct, "straight": straight, "response": result["raw_response"]}
                if name == BARE:
                    collector.records.append({"item": fu.item_id(result), **record})
                else:
                    collector.add(fu.item_id(result), stash[result["prompt"]].to(lm.device), **record)
            collector.save(directory, name)
            if name == FIXING:
                fixed = set(collector.fixed)
            records = collector.records
            print(f"{key} {name}: accuracy {sum(r['correct'] for r in records) / len(records):.3f}, "
                  f"straight {sum(r['straight'] for r in records) / len(records):.3f} over {len(records)}, "
                  f"P+ {len(collector.id_items)}, mean set {len(collector.mean_items)} "
                  f"({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
