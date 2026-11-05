"""A hand-picked J-head set for one simple task, to patch with ``scripts/patch.py`` (no GPU).

Writes ``<out>/heads.json`` in the layout of ``results/heads.json`` (family ``simple``, the
styles of ``--styles``, default ``rule``) with the given heads as the J-head set and an empty
Logit-head set, so that

    LTH_RESULTS=<out> .venv/bin/python scripts/patch.py --gpu 0 --family simple \\
        --tasks <task> --part <name> --sets jhead [--same-style]

patches only them. Used for the present-past heads of the k = 1 union that no other task's set
contains (``scripts/shared_heads.py --unshared``), split into the retrieval head among them and
the rest, and for L15H1 alone, the head that the J-head example set of size 5 adds to the
threshold set (both styles, each with its own means).

    .venv/bin/python scripts/head_subset.py --task present-past --heads L21H9 --out results/pp_L21H9
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import simple  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=simple.TASKS, required=True)
    parser.add_argument("--heads", nargs="+", required=True, help="heads as L<layer>H<head>")
    parser.add_argument("--styles", nargs="+", choices=tuple(simple.WORDINGS), default=["rule"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    heads = [[int(l), int(h), None] for l, h in (re.fullmatch(r"L(\d+)H(\d+)", x).groups() for x in args.heads)]
    sets = {"jhead": {"rule": "hand-picked: " + " ".join(args.heads), "heads": heads},
            "logithead": {"heads": []}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "heads.json").write_text(json.dumps({"simple": {args.task: {style: sets for style in args.styles}}}, indent=1))
    print(f"wrote {args.out}/heads.json: {args.task}, {len(heads)} heads")


if __name__ == "__main__":
    main()
