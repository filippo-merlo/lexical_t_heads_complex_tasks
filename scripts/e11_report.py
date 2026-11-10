"""E11 (scripts/e11_sets.py, e11_patch.sh, e11_fv_residual.py): accuracy files in the layout of the
other held-out experiments, for ``scripts/synthesis_report.py``.

- ``results/e11_complete/accuracy_alpha.csv``: head-patched sets (fold, family, task, set, source,
  alpha, accuracy = mean over the five rule means, n_heads), as ``scripts/e10_report.py``;
- ``results/e11_complete/fv_residual_alpha.csv``: Davidson et al.'s intervention (fv30, fv30_nogen),
  same columns (one task mean, so no averaging).

    .venv/bin/python scripts/e11_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import heldout_report as hr  # noqa: E402
from lth import RESULTS  # noqa: E402

OUT = RESULTS / "e11_complete"


def main() -> None:
    hr.HELD = OUT
    curves, _ = hr.load()
    curves.to_csv(OUT / "accuracy_alpha.csv", index=False)
    rows = []
    for fold in (1, 2):
        for path in sorted((OUT / f"fold{fold}").glob("fv_residual*.csv")):
            rows.append(pd.read_csv(path).assign(fold=fold))
    fv = pd.concat(rows, ignore_index=True)
    fv = fv[fv.source != "none"][["fold", "family", "task", "set", "source", "alpha", "accuracy", "n_heads"]]
    assert not fv.duplicated(["fold", "family", "task", "set", "source", "alpha"]).any()
    fv.to_csv(OUT / "fv_residual_alpha.csv", index=False)
    m = curves[curves.source == "task"].groupby(["family", "set"]).accuracy.max()
    print(f"{len(curves)} curve rows, {curves.set.nunique()} sets; fv rows {len(fv)}")
    print(fv.groupby(["family", "set", "source", "alpha"]).accuracy.mean().round(3).to_string())
    del m


if __name__ == "__main__":
    main()
