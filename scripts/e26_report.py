"""E26 report: the capped comparison's sets with Davidson et al.'s summed intervention (``scripts/e26_summed.py``)
against the same sets patched head by head (E13, E22, E23), check 1.

The summed runs are reduced exactly as E13's (``scripts/e13_report.py``): accuracy averaged over the five rule
means, alpha cross-fitted over {1, 2} between the folds (ties to 1), random draws at the alpha of the DBT set of
the same scope and N and averaged over the five draws, mean of the two folds. Head by head: E13's
``summary.csv`` and ``per_task.csv`` with the best combinations of E22 and E23 (``e13_report.add_best``); E13's
pooled-mean summed vector (``fvsum``) is kept as a reference row. Only sets run on all 14 tasks in both folds
enter the tables and figure (per-task Yang terms at 60 and 80 are left out on the tasks where it has fewer
than N heads).

Writes ``results/e26_summed/{accuracy_alpha,per_task,summary,compare}.csv``, ``paper/tables/e26_{task,shared}.tex``
and ``paper/figures/e26_diff.pdf``.

    .venv/bin/python scripts/e26_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e13_report as e13  # noqa: E402
import heldout_report as hr  # noqa: E402
import results_tex as rt  # noqa: E402
from lth import RESULTS  # noqa: E402

OUT = RESULTS / "e26_summed"
SIZES = e13.SIZES
ROWS = ("yang", "BT", "DBT", "DBT_nogen", "DBMN", "DBTMS", "DBTLS", "causal", "causal_nogen", "random")
LABEL = {"yang": "Yang terms", "BT": "BT", "DBT": "DBT", "DBT_nogen": "DBT $-$ generic", "DBMN": "DBMN",
         "DBTMS": "DBTMS", "DBTLS": "DBTLS", "causal": "Causal", "causal_nogen": "Causal $-$ generic",
         "random": "Random (DBT layers)"}
FAMILY = {"yang": "yang", "BT": "filter", "DBT": "filter", "DBT_nogen": "filter", "DBMN": "best", "DBTMS": "best",
          "DBTLS": "best", "causal": "causal", "causal_nogen": "causal", "random": "random"}
FAMILY_TITLE = {"yang": "Baseline", "filter": "Decoding and attention filters",
                "best": "Best combinations with L, M, S, N (chosen on the test items, head by head)",
                "causal": "Davidson et al.\\ (causal score)", "random": "Control"}


def load_summed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hr.HELD = OUT
    curves, base = hr.load()
    curves = curves[curves.family == "simple"]
    meta = pd.DataFrame([e13.parse(s) for s in curves.set], columns=["scope", "condition", "n"], index=curves.index)
    curves = pd.concat([curves, meta], axis=1)
    per, s = e13.summarise(e13.crossfit(curves))
    return curves, per, s


def head_by_head() -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_csv(e13.OUT / "summary.csv")
    per = pd.read_csv(e13.OUT / "per_task.csv")
    return e13.add_best(s, per)


def compare(summed: pd.DataFrame, hbh: pd.DataFrame) -> pd.DataFrame:
    full = lambda d: d[(d.tasks == 14) & (d.folds == 2)]  # noqa: E731
    keys = ["scope", "condition", "n"]
    a = full(hbh)[keys + ["check1_task", "check1_control", "alpha2_share"]]
    b = full(summed)[keys + ["check1_task", "check1_control", "alpha2_share"]]
    c = a.merge(b, on=keys, how="outer", suffixes=("_hbh", "_summed"))
    c["diff"] = c.check1_task_summed - c.check1_task_hbh
    return c.sort_values(keys).reset_index(drop=True)


def tex_scope(c: pd.DataFrame, scope: str) -> None:
    """Rows: conditions; per N two columns, head by head (HbH) and summed (Sum)."""
    g = c[c.scope == scope].set_index(["condition", "n"])
    best = {(n, col): g.xs(n, level="n").drop(["random", "fvsum", "fvsum_nogen"], errors="ignore")[col].max()
            for n in SIZES for col in ("check1_task_hbh", "check1_task_summed")}
    lines, fam = [], None
    rows = list(ROWS)
    if scope == "shared":  # the pooled-mean reference rows right after the causal rankings
        k = rows.index("causal_nogen") + 1
        rows[k:k] = ["fvsum", "fvsum_nogen"]
    for cond in rows:
        family = FAMILY.get(cond, "causal")
        if family != fam:
            if fam is not None:
                lines.append(r"\midrule")
            lines.append(r"\multicolumn{" + str(1 + 2 * len(SIZES)) + r"}{@{}l}{\textit{" + FAMILY_TITLE[family] + r"}} \\")
            fam = family
        cells = []
        for n in SIZES:
            for col in ("check1_task_hbh", "check1_task_summed"):
                if cond.startswith("fvsum"):  # E13's pooled-mean summed vector (stored with E13): Sum column only
                    v = g.check1_task_hbh.get((cond, n), np.nan) if col.endswith("summed") else np.nan
                else:
                    v = g[col].get((cond, n), np.nan)
                cells.append(e13.cell(v, cond != "random" and not cond.startswith("fvsum") and v == best[(n, col)]))
        label = {"fvsum": "Causal, pooled mean", "fvsum_nogen": r"Causal $-$ generic, pooled mean"}.get(cond, LABEL.get(cond))
        lines.append(r"\quad " + label + " & " + " & ".join(cells) + r" \\")
    header = [" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for n in SIZES) + r" \\",
              "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(SIZES))),
              r"Heads (N) & " + " & ".join(["HbH & Sum"] * len(SIZES)) + r" \\"]
    text = "\n".join([r"\begin{tabular}{@{}l" + "c" * (2 * len(SIZES)) + r"@{}}", r"\toprule", *header, r"\midrule",
                      *lines, r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / f"e26_{scope}.tex").write_text("% Generated by scripts/e26_report.py.\n" + text + "\n")


def figure_diff(c: pd.DataFrame) -> None:
    """Summed minus head by head against N, in the panels of e13_size (fixed sets above, the best
    further combinations below)."""
    e13.size_panels(c, "diff", "e26_diff", "summed $-$ head by head (check 1 accuracy, mean of 14 tasks)",
                    (-0.36, 0.1), zero=True)


def main() -> None:
    curves, per, s = load_summed()
    curves.to_csv(OUT / "accuracy_alpha.csv", index=False)
    per.to_csv(OUT / "per_task.csv", index=False)
    s.to_csv(OUT / "summary.csv", index=False)
    s_hbh, _ = head_by_head()
    c = compare(s, s_hbh)
    c.to_csv(OUT / "compare.csv", index=False)
    for scope in ("task", "shared"):
        tex_scope(c, scope)
    figure_diff(c)
    pd.set_option("display.width", 220)
    for col in ("check1_task_hbh", "check1_task_summed", "check1_control_summed", "alpha2_share_summed"):
        print(col)
        print(c.pivot_table(index="condition", columns=["scope", "n"], values=col).round(2).to_string())
    print("incomplete summed sets:", s[(s.tasks < 14) | (s.folds < 2)].set.tolist())


if __name__ == "__main__":
    main()
