"""E4 (paper/e4_heldout.tex): identification readings of each half of the item split.

For every simple task, half (``results/heldout/split.json``) and rule wording: the first
``CAP`` items of the half answered correctly (flexible score, records of
``data/readings/simple``), with every head's top-10 tokens under both readouts at the last
prompt token, run as ``scripts/build_readings.py`` does (saved prompts, Yang et al.'s batches
of 20, padding masked). Written in the ``lth.store`` format to
``data/readings/heldout/{half}/simple/{task}/{wording}`` with the wording's full records, so
that scripts reading ``readings_dir`` work on a half with ``LTH_READINGS=data/readings/heldout/{half}``.
E5 (paper/proposal-llm-heads.tex) reads the same prompts with each head's top 20 tokens
(``--k 20``) into another folder (``--out``).

    .venv/bin/python scripts/heldout_readings.py --gpu 2
    .venv/bin/python scripts/heldout_readings.py --gpu 0 --k 20 --out data/readings/heldout_top20 --tasks english-german
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, RESULTS, gpu, simple  # noqa: E402
from lth.store import CAP, done, load, load_outputs, outputs_path, readings_dir, save  # noqa: E402

OUT = DATA / "readings" / "heldout"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS))
    parser.add_argument("--k", type=int, default=10, help="top tokens kept per head (E4: 10)")
    parser.add_argument("--out", type=Path, default=OUT, help="readings folder (E4: data/readings/heldout)")
    args = parser.parse_args()
    gpu.pin(args.gpu)
    from lth.model import LM

    lm = LM.load()
    halves = json.loads((RESULTS / "heldout" / "split.json").read_text())
    for task in args.tasks:
        for wording in simple.WORDINGS["rule"]:
            records = load(readings_dir("simple", task), wording)["records"]
            prompts = {int(r["item"]): r["prompt"] for r in load_outputs(outputs_path("simple", task, wording))}
            for half in ("A", "B"):
                directory = args.out / half / "simple" / task
                if done(directory, wording):
                    continue
                start = time.time()
                members = set(halves[task][half])
                id_items = [int(r["item"]) for r in records if r["correct"] and int(r["item"]) in members][:CAP]
                ids = simple.token_ids(lm.tokenizer, [prompts[i] for i in id_items])
                top = []
                for b in range(0, len(ids), simple.BATCH):
                    _, z = lm.next_tokens_batch(ids[b:b + simple.BATCH])
                    top += [lm.top_tokens(row, k=args.k) for row in z]
                save(directory, wording, records, [str(i) for i in id_items], top)
                print(f"{half} {task} {wording}: P+ {len(id_items)} ({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
