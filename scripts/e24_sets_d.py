"""E24 step D (no GPU): the rule-built sets of every size, patched with the example mean.

Step A patched one size per rule (each rule's best), so sets of different sizes could not be
compared. Step D (user, 2026-11-16): the same rules in both scopes (per task and shared) at every
size N in 20, 40, 60, 80, 100, 120:

- Causal and Causal - generic (E13: ``causal{N}``, ``causal{N}_nogen``);
- DBMN, DBTMS, DBTLS, DBTM, the four combinations among E23's 15 best in both scopes (E22 at 40,
  E23 at the other sizes);
- Yang terms (E13: ``yang{N}``; per task, where too few heads decode a term, E13 built a smaller set
  at 60 and 80 and none at 100 and 120 on some tasks; every existing set is patched, and the report
  shows "--" wherever a task's set has fewer than N heads);
- random: as many heads in each layer as the example-built DB of the same scope and N, drawn
  outside it (``scripts/e24_sets_b.py``; five draws, seeds 0-4 per task, 100-104 shared), patched with
  rule means and with the example mean (step B ran 40 and 100; here the other sizes).

Heads are copied per fold from the runs' own ``heads.json``, unchanged; the style tag becomes
``example`` so that ``scripts/patch.py --same-style`` gives them the example mean. Their rule-mean
accuracies are those of E13/E22/E23.

Writes ``results/e24_cross_style/d/fold{f}/heads.json``.

    .venv/bin/python scripts/e24_sets_d.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS, simple  # noqa: E402

import e24_sets_b as b  # noqa: E402
from e11_sets import layer_matched  # noqa: E402
from e13_lists import JH, N_GENERIC, shares, token_words  # noqa: E402
from e13_sets import N_RANDOM, SIZES  # noqa: E402

OUT = RESULTS / "e24_cross_style" / "d"
RULES = ("causal", "causal_nogen", "DBMN", "DBTMS", "DBTLS", "DBTM", "yang")
SCOPES = {"task": "", "shared": "shared_"}
RANDOM_DONE = (40, 100)  # step B


def set_name(prefix: str, rule: str, n: int) -> str:
    if rule == "causal_nogen":
        return f"{prefix}causal{n}_nogen"
    return f"{prefix}{rule}{n}"


def main() -> None:
    from lth.evaluation import tokenizer

    tok = tokenizer()
    tmap, words = token_words(tok)
    wiki = shares(np.load(DATA / "readings" / "neutral" / "wikitext.npz")["top"][:, JH], tmap, len(words))
    generic = {words[i] for i in np.argsort(-wiki, kind="stable")[:N_GENERIC]}
    for fold, half in b.HALF.items():
        known = b.patched(fold)
        _, ex_score, _ = b.example_scores(half, tmap, words, generic)
        ex = b.build(ex_score)
        step_b = json.loads((RESULTS / "e24_cross_style" / "b" / f"fold{fold}" / "heads.json").read_text())["simple"]
        sets = {t: [] for t in simple.TASKS}
        missing = []
        for scope, prefix in SCOPES.items():
            for rule in RULES:
                for n in SIZES:
                    name = set_name(prefix, rule, n)
                    for t in simple.TASKS:
                        if name in known[t]:
                            sets[t].append(["example", name, known[t][name]])
                        else:
                            missing.append((t, name))
            for n in SIZES:
                for i in range(N_RANDOM):
                    rname = f"random_ex_{prefix}db{n}_{i}"
                    for t in simple.TASKS:
                        base = ex[scope, "DB", n][t if scope == "task" else simple.TASKS[0]]
                        draw = layer_matched(base, seed=i if scope == "task" else 100 + i)
                        if n in RANDOM_DONE:  # the same draw as step B, not rerun
                            assert [s[2] for s in step_b[t]["sets"] if s[1] == rname][0] == draw
                            continue
                        for style in ("rule", "example"):
                            sets[t].append([style, rname, draw])
        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "heads.json").write_text(json.dumps({"simple": {t: {"sets": v} for t, v in sets.items()}}, indent=1))
        miss = sorted({m[1] for m in missing})
        assert set(miss) <= {"yang100", "yang120"}, miss
        print(f"fold {fold}: {len(sets[simple.TASKS[0]])} set-styles per task; not built on some tasks: {miss}")


if __name__ == "__main__":
    main()
