"""E24 step A (no GPU): the best rule-chosen sets of the capped grid, to be patched with example means.

Yang et al.'s main claim is that the heads carrying a task are shared between example-based and
instruction-based prompts. Every set of E13/E23 is chosen on rule prompts and patched with rule
means. Step A keeps the heads and swaps the means: the same sets, patched in check 1 with the mean
of the 5-shot prompts of the choosing half (Yang et al.'s example-style identification prompts),
under E13's protocol otherwise (``scripts/e24_patch_a.sh``).

Sets (user, 2026-11-16: the best sets of both scopes; paper/head-sets-simple.tex, Best approaches):
per task DBTMS100, DBMN60 (E23), DBT40 and causal120 (E13); shared shared_causal100_nogen,
shared_causal80_nogen (E13), shared_DBTMS120 and shared_DBMN80 (E23); and E13's five layer-matched
random draws to DBT100 of each scope (task means only, as in E13). Heads are copied per fold from
the runs' own ``heads.json``, unchanged; the style tag becomes ``example`` so that
``scripts/patch.py --same-style`` gives them the example means.

Writes ``results/e24_cross_style/a/fold{f}/heads.json``.

    .venv/bin/python scripts/e24_sets_a.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import RESULTS, simple  # noqa: E402

OUT = RESULTS / "e24_cross_style" / "a"
SOURCES = {
    "e13_capped": ["DBT40", "causal120", "shared_causal100_nogen", "shared_causal80_nogen",
                   *[f"random_dbt100_{i}" for i in range(5)], *[f"random_shared_dbt100_{i}" for i in range(5)]],
    "e23_best_sizes": ["DBTMS100", "DBMN60", "shared_DBTMS120", "shared_DBMN80"],
}


def main() -> None:
    for fold in (1, 2):
        sets = {t: [] for t in simple.TASKS}
        for run, names in SOURCES.items():
            heads = json.loads((RESULTS / run / f"fold{fold}" / "heads.json").read_text())["simple"]
            for t in simple.TASKS:
                listed = {name: hs for style, name, hs in heads[t]["sets"] if style == "rule"}
                for name in names:
                    sets[t].append(["example", name, listed[name]])
        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "heads.json").write_text(json.dumps({"simple": {t: {"sets": v} for t, v in sets.items()}}, indent=1))
        n = {name: len(hs) for name, hs in [(s[1], s[2]) for s in sets[simple.TASKS[0]]]}
        print(f"fold {fold}: {len(sets[simple.TASKS[0]])} sets per task; sizes {n}")


if __name__ == "__main__":
    main()
