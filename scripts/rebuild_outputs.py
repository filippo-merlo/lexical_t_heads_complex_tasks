"""Rebuilt design (paper/head-sets-simple-design.tex, App. D): unpatched replies to the six example formats.

Every format of ``lth.simple.EXAMPLE_FORMATS`` at five shots, each item's demonstrations drawn from its
own half (``lth.simple.demo_items``), on the first 500 items of each half, as the 30 further wordings
were run (``scripts/e14_outputs.py``: greedy, batches of 20 with the padding masked, a reply ends at
``;``, a newline once begun, EOS or 32 tokens; flexible and strict scores). Writes the replies to
``data/outputs/simple/<task>/fmt-F<k>.jsonl.gz`` and the records to
``data/readings/simple/<task>/fmt-F<k>.{json,npz}``; a format already there is skipped.

    .venv/bin/python scripts/rebuild_outputs.py --gpu 0 --tasks antonym synonym
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import gpu, simple  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS))
    args = parser.parse_args()
    gpu.pin(args.gpu)

    from e14_outputs import run_wordings

    from lth.model import LM

    lm = LM.load()
    for task in args.tasks:
        run_wordings(lm, task, simple.FORMAT_WORDINGS)


if __name__ == "__main__":
    main()
