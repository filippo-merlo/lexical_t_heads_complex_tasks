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
    best = {(n, col): g.xs(n, level="n").drop("random", errors="ignore")[col].max()
            for n in SIZES for col in ("check1_task_hbh", "check1_task_summed")}
    lines, fam = [], None
    rows = ROWS + (("fvsum", "fvsum_nogen") if scope == "shared" else ())
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
        label = {"fvsum": "Causal, pooled mean (E13)", "fvsum_nogen": r"Causal $-$ generic, pooled mean (E13)"}.get(cond, LABEL.get(cond))
        lines.append(r"\quad " + label + " & " + " & ".join(cells) + r" \\")
    header = [" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for n in SIZES) + r" \\",
              "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(SIZES))),
              r"Heads (N) & " + " & ".join(["HbH & Sum"] * len(SIZES)) + r" \\"]
    text = "\n".join([r"\begin{tabular}{@{}l" + "c" * (2 * len(SIZES)) + r"@{}}", r"\toprule", *header, r"\midrule",
                      *lines, r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / f"e26_{scope}.tex").write_text("% Generated by scripts/e26_report.py.\n" + text + "\n")


def figure_diff(c: pd.DataFrame) -> None:
    """Summed minus head by head against N, per scope; lines and colours of e13_size."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), sharey=True)
    for ax, scope, title in ((axes[0], "task", "(a) One set per task"), (axes[1], "shared", "(b) One set for all 14 tasks")):
        g = c[(c.scope == scope) & c["diff"].notna()]
        ax.axhline(0, color=rt.INK_2, lw=0.8, zorder=1)
        ends = []
        for cond, (ls, lw, mk, lab) in e13.LINE.items():
            r = g[g.condition == cond].sort_values("n")
            if r.empty or cond.startswith("fvsum") or (scope, cond) in e13.FIG_SKIP:
                continue
            ax.plot(r.n, r["diff"], color=e13.COLOUR[e13.CONDITIONS[cond][1]], ls=ls, lw=lw, marker=mk, ms=4.5,
                    mec=rt.SURFACE, mew=0.8, zorder=3)
            ends.append((r["diff"].iloc[-1], lab, r.n.iloc[-1]))
        ends.sort()
        last = -9.0
        for y, lab, x in ends:
            y = max(y, last + 0.028)
            last = y
            ax.annotate(lab, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=5.8,
                        color=rt.INK_2, annotation_clip=False)
        ax.set_xticks(SIZES)
        ax.set_xlim(14, SIZES[-1] + 6)
        ax.set_xlabel("heads in the set (N)")
        ax.set_title(title, loc="left")
        rt.style_axes(ax)
    axes[0].set_ylabel("summed $-$ head by head\n(check 1 accuracy, mean of 14 tasks)")
    handles = [plt.Line2D([], [], color=e13.COLOUR[f], lw=2, marker=m, ms=4) for f, m in
               (("filter", "o"), ("best", "h"), ("causal", "s"), ("yang", "^"), ("random", "v"))]
    labels = ["filters D, B, T", "best combinations with L, M, S, N", "Davidson et al. (causal)",
              "Yang terms (baseline)", "random heads"]
    handles.append(plt.Line2D([], [], color=rt.INK_2, lw=1.4, ls=(0, (4, 2))))
    labels.append("dashed: generic heads removed")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=6.6)
    fig.tight_layout(rect=(0, 0, 0.97, 0.85), w_pad=3.5)
    fig.savefig(rt.FIGURES / "e26_diff.pdf")
    plt.close(fig)


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
