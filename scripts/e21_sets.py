"""E21 step 1 (no GPU): head rankings on Bai et al.'s token types (E20's attention split) as new
conditions of E13's capped grid, at N = 40, per task and shared.

New scores (``scripts/e20_bai_attention.py``; per task the mean over the choosing half's pooled
correct prompts of the five rule templates, last token; higher ranks higher):

- **L** attention to the last token itself;
- **M** attention to the template markers " Q:" and "\\nA" (0 on the cloze templates, which have none);
- **S** attention to punctuation and NLTK stopwords of the instruction (Bai et al.'s stopword type);
- **N** attention to the whole instruction (task words + S + other instruction words): T without
  the word list.

Each X in L, M, S, N is run alone, with B and with DB (X, BX, DBX: 12 rules), combined exactly as
E13 (``scripts/e13_sets.py``): per task, each filter's within-task rank averaged over the
combination; shared, E10's rule (within-task ranks averaged over tasks, ranked again, averaged over
the combination). B and D are E13's (B = BOS mass, identical in E20; D from
``results/e13_capped/fold{f}/d_scores.npz``). Ties: lower layer, then head. No random sets (E13's
layer-matched draws stay the reference) and no generic-head removal.

Check: the same code on E20's task-word mass reproduces E13's T40, BT40 and DBT40 (per task and
shared) exactly.

Writes ``results/e21_bai_rules/fold{f}/heads.json`` (``scripts/patch.py`` set-list layout) and
``sets.csv`` (size, median layer, overlap with E13's sets of the same scope).

    .venv/bin/python scripts/e21_sets.py
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

from e13_sets import heads_of, ranks, top  # noqa: E402
from e20_bai_attention import CATS  # noqa: E402

OUT = RESULTS / "e21_bai_rules"
HALF = {1: "A", 2: "B"}
N = 40
PARTS = {"L": ("self",), "M": ("template",), "S": ("punct", "function"),
         "N": ("task", "punct", "function", "content"), "T": ("task",), "B": ("bos",)}
SIGN = {"D": 1, "B": -1, "T": 1, "L": 1, "M": 1, "S": 1, "N": 1}
NEW = [p + x for x in "LMSN" for p in ("", "B", "DB")]
CHECK = ("T", "BT", "DBT")
COMPARE = ("T", "BT", "DBT", "causal")  # E13 sets reported for overlap


def main() -> None:
    for fold, half in HALF.items():
        out = OUT / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        dsc = np.load(RESULTS / "e13_capped" / f"fold{fold}" / "d_scores.npz")
        att = {t: np.load(DATA / "readings" / "heldout" / f"attention20_{half}" / f"{t}.npy").astype(np.float64).mean(0)
               for t in simple.TASKS}
        score = {t: {"D": dsc[t], **{f: att[t][:, [CATS.index(c) for c in cs]].sum(1) for f, cs in PARTS.items()}}
                 for t in simple.TASKS}
        filters = "DBTLMSN"
        e13 = {t: {name: hs for _, name, hs in json.loads((RESULTS / "e13_capped" / f"fold{fold}" / "heads.json")
                                                          .read_text())["simple"][t]["sets"]}
               for t in simple.TASKS}

        per_task = {t: [] for t in simple.TASKS}
        for t in simple.TASKS:
            r = {f: ranks(SIGN[f] * score[t][f]) for f in filters}
            for c in NEW + list(CHECK):
                order = np.argsort(np.mean([r[f] for f in c], axis=0), kind="stable")
                hs = top(order, N)
                if c in CHECK:
                    assert hs == e13[t][f"{c}{N}"], (fold, t, c, "E13 per-task set not reproduced")
                else:
                    per_task[t].append(["rule", f"{c}{N}", hs])

        within = {f: np.stack([ranks(SIGN[f] * score[t][f]) for t in simple.TASKS]).mean(0) for f in filters}
        combined = {f: ranks(-within[f]) for f in filters}
        shared = []
        for c in NEW + list(CHECK):
            order = np.argsort(np.mean([combined[f] for f in c], axis=0), kind="stable")
            hs = top(order, N)
            if c in CHECK:
                assert hs == e13[simple.TASKS[0]][f"shared_{c}{N}"], (fold, c, "E13 shared set not reproduced")
            else:
                shared.append(["rule", f"shared_{c}{N}", hs])
        print(f"fold {fold}: E13 T/BT/DBT-{N} reproduced per task and shared")

        heads = {"simple": {t: {"sets": per_task[t] + shared} for t in simple.TASKS}}
        (out / "heads.json").write_text(json.dumps(heads, indent=1))
        rows = []
        for t in simple.TASKS:
            for _, name, hs in per_task[t] + shared:
                sh = name.startswith("shared_")
                hs = {tuple(h) for h in hs}
                ref = {c: {tuple(h) for h in e13[t][("shared_" if sh else "") + f"{c}{N}"]} for c in COMPARE}
                rows.append(dict(fold=fold, task=t, set=name, n_heads=len(hs),
                                 layer_median=float(np.median([l for l, _ in hs])),
                                 **{f"overlap_{c}": len(hs & ref[c]) for c in COMPARE}))
        with (out / "sets.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"fold {fold}: {len(per_task[simple.TASKS[0]])} per-task + {len(shared)} shared sets")


if __name__ == "__main__":
    main()
