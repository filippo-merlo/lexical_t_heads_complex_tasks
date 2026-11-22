"""Top J-head tokens of selected heads on selected tasks (paper/head-roles-design.tex, Table 3).

Same readings as scripts/roles_readings.py (``data/readings/heldout/<half>``, five instruction
templates, last token, top 10). For each (head, task): the eight tokens most often among the
head's top 10, with the share of prompts. Writes ``results/roles/readings/examples.csv``.

    .venv/bin/python scripts/roles_examples.py
"""

from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import RESULTS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roles_readings import load_half, tokenizer  # noqa: E402

HALF = "A"
EXAMPLES = (
    ("L14H13", ("country-capital", "country-currency", "english-french", "next_item")),
    ("L25H23", ("country-capital", "country-currency", "antonym")),
    ("L31H1", ("country-capital", "english-german")),
    ("L24H3", ("country-capital", "landmark-country", "english-french")),
    ("L26H27", ("country-capital", "product-company")),
    ("L28H24", ("next_item", "prev_item")),
    ("L24H1", ("english-french", "english-german")),
)


def main() -> None:
    tok = tokenizer()
    rows = []
    for head, tasks in EXAMPLES:
        layer, h = head[1:].split("H")
        index = int(layer) * 32 + int(h)
        for task in tasks:
            top, _ = load_half(task, HALF)
            c = collections.Counter(int(t) for p in range(len(top)) for t in top[p, index])
            rows.append({"head": head, "task": task, "half": HALF, "prompts": len(top),
                         "tokens": " | ".join(f"{tok.decode([t])!r} {n / len(top):.2f}" for t, n in c.most_common(8))})
            print(rows[-1])
    out = RESULTS / "roles" / "readings" / "examples.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
