"""E4 (paper/e4_heldout.tex): the function-vector head sets of one fold, added to its heads.json.

From the fold's CIE (``scripts/davidson_fv.py --half``, ``results/heldout/fold{f}/fv/cie.npz``,
computed on the choosing half): a head's score on a task is its CIE averaged over the three
kinds of uninformative instruction. Sets, per task: ``fv20``, ``fv40``, ``fvNt`` (the first N
heads by the task's score, N_t = size of the fold's ``cc`` set); ``fvuniv20``, ``fvuniv40`` (the
first N by the score averaged over the 14 tasks, Davidson et al.'s choice). Ties: lower index
(layer, then head). Existing ``fv*`` sets are replaced.

    .venv/bin/python scripts/heldout_fv_sets.py --fold 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import RESULTS, simple  # noqa: E402

KINDS = ("equiprobable", "wikitext", "other_task")


def top(score: np.ndarray, n: int) -> list[list[int]]:
    order = np.argsort(-score.ravel(), kind="stable")[:n]
    return sorted([int(i) // score.shape[1], int(i) % score.shape[1]] for i in order)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    d = RESULTS / "heldout" / f"fold{args.fold}"
    cie = np.load(d / "fv" / "cie.npz")
    score = {t: np.mean([cie[f"{t}/{k}"] for k in KINDS], axis=0) for t in simple.TASKS}
    univ = np.mean([score[t] for t in simple.TASKS], axis=0)
    heads = json.loads((d / "heads.json").read_text())
    for t in simple.TASKS:
        sets = [s for s in heads["simple"][t]["sets"] if not s[1].startswith("fv")]
        n_t = len(next(hs for _, n, hs in sets if n == "cc"))
        sets += [["rule", "fv20", top(score[t], 20)], ["rule", "fv40", top(score[t], 40)],
                 ["rule", "fvNt", top(score[t], n_t)],
                 ["rule", "fvuniv20", top(univ, 20)], ["rule", "fvuniv40", top(univ, 40)]]
        heads["simple"][t]["sets"] = sets
        cc = {tuple(h) for _, n, hs in sets if n == "cc" for h in hs}
        fv = {tuple(h) for _, n, hs in sets if n == "fvNt" for h in hs}
        print(f"{t:17s} N_t {n_t:3d}  fvNt shares {len(cc & fv):3d} heads with cc")
    partial = d / "heads.json.partial"  # replaced at once: patch runs may be reading it
    partial.write_text(json.dumps(heads, indent=1))
    partial.replace(d / "heads.json")


if __name__ == "__main__":
    main()
