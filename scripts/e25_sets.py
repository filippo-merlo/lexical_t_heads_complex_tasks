"""E25 step 1 (no GPU): punctuation and stopwords as separate filters, at N = 40 (user, 2026-11-16).

S (``scripts/e21_sets.py``) is attention to the instruction's punctuation and NLTK stopwords
together, Bai et al.'s stopword type. Here it is split into its two parts, both already stored
separately by ``scripts/e20_bai_attention.py``:

- **P** attention to punctuation (tokens without a letter or digit, outside the query and markers);
- **W** attention to stopwords (tokens whose words are all in NLTK's English list).

S = P + W. In the three leading combinations that hold S, S is replaced by P and by W (agreed with
the user before any run): per task DBTMS; shared DBTMS and DBTLS. 40 heads only, both folds.
New sets: per task DBTMP40, DBTMW40; shared_DBTMP40, shared_DBTMW40, shared_DBTLP40, shared_DBTLW40.

Scores and combination rule exactly as ``scripts/e22_sets.py`` (per task, each filter's within-task
rank averaged over the combination's filters; shared, within-task ranks averaged over tasks, ranked
again, averaged over the combination). Ties: lower layer, then head.

Check: the same code with S rebuilds the patched DBTMS40, shared_DBTMS40 and shared_DBTLS40 of
``results/e22_all_combos`` exactly.

Writes ``results/e25_punct_split/fold{f}/heads.json`` (``scripts/patch.py`` set-list layout) and
``sets.csv`` (size, median layer, heads shared with the S version of the same set).

    .venv/bin/python scripts/e25_sets.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS, simple  # noqa: E402

from e13_sets import ranks, top  # noqa: E402
from e20_bai_attention import CATS  # noqa: E402
from e21_sets import PARTS, SIGN  # noqa: E402

OUT = RESULTS / "e25_punct_split"
HALF = {1: "A", 2: "B"}
N = 40
PARTS25 = {**PARTS, "P": ("punct",), "W": ("function",)}
SIGN25 = {**SIGN, "P": 1, "W": 1}
BASE = {"task": ("DBTMS",), "shared": ("DBTMS", "DBTLS")}  # the S combinations whose S is replaced


def variants(c: str) -> list[str]:
    return [c.replace("S", x) for x in "PW"]


def name(scope: str, c: str) -> str:
    return f"{'shared_' if scope == 'shared' else ''}{c}{N}"


def main() -> None:
    assert PARTS25["S"] == PARTS25["P"] + PARTS25["W"]
    for fold, half in HALF.items():
        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        dsc = np.load(RESULTS / "e13_capped" / f"fold{fold}" / "d_scores.npz")
        att = {t: np.load(DATA / "readings" / "heldout" / f"attention20_{half}" / f"{t}.npy").astype(np.float64).mean(0)
               for t in simple.TASKS}
        score = {t: {"D": dsc[t], **{f: att[t][:, [CATS.index(c) for c in cs]].sum(1) for f, cs in PARTS25.items()}}
                 for t in simple.TASKS}
        letters = sorted(SIGN25)

        def per_task(t: str, c: str) -> list:
            r = {f: ranks(SIGN25[f] * score[t][f]) for f in c}
            return top(np.argsort(np.mean([r[f] for f in c], axis=0), kind="stable"), N)

        within = {f: np.stack([ranks(SIGN25[f] * score[t][f]) for t in simple.TASKS]).mean(0) for f in letters}
        combined = {f: ranks(-within[f]) for f in letters}

        def shared(c: str) -> list:
            return top(np.argsort(np.mean([combined[f] for f in c], axis=0), kind="stable"), N)

        old = {t: {nm: hs for _, nm, hs in json.loads((RESULTS / "e22_all_combos" / f"fold{fold}" / "heads.json")
                                                     .read_text())["simple"][t]["sets"]} for t in simple.TASKS}
        for t in simple.TASKS:
            for c in BASE["task"]:
                assert per_task(t, c) == old[t][name("task", c)], (fold, t, c)
            for c in BASE["shared"]:
                assert shared(c) == old[t][name("shared", c)], (fold, c)
        print(f"fold {fold}: DBTMS40, shared_DBTMS40 and shared_DBTLS40 reproduced exactly")

        sets = {t: [] for t in simple.TASKS}
        rows = []
        for t in simple.TASKS:
            for scope, cs in BASE.items():
                for c in cs:
                    ref = per_task(t, c) if scope == "task" else shared(c)
                    for v in variants(c):
                        hs = per_task(t, v) if scope == "task" else shared(v)
                        sets[t].append(["rule", name(scope, v), hs])
                        rows.append(dict(fold=fold, task=t, scope=scope, set=name(scope, v), replaces=name(scope, c),
                                         n_heads=len(hs), layer_median=float(np.median([l for l, _ in hs])),
                                         shared_with_S_version=len(set(map(tuple, hs)) & set(map(tuple, ref))) / N))
        heads = {"simple": {t: {"sets": sets[t]} for t in simple.TASKS}}
        (out / "heads.json").write_text(json.dumps(heads, indent=1))
        with (out / "sets.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        d = {}
        for r in rows:
            d.setdefault(r["set"], []).append(r["shared_with_S_version"])
        print(f"fold {fold}: {len(sets[simple.TASKS[0]])} sets per task written; share of heads in the S version: "
              + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in d.items()))


if __name__ == "__main__":
    main()
