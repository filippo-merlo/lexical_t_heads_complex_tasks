"""Figures for E22/E23 in paper/head-sets-simple (run after scripts/e22_report.py and e23_report.py).

- ``paper/figures/e23_size.pdf``: check-1 accuracy against set size for the 15 best E22 combinations
  per scope (E23), three of them highlighted (the same three in both panels), next to E13's DBT,
  Causal, Causal - generic and random.
- ``paper/figures/e22_letters.pdf``: what each letter adds at 40 heads (E22): accuracy of a
  combination with the letter minus the same combination without it, mean over every such pair,
  per task and shared.

Colours: results_tex's palette as in Figure e13_size (filters aqua, causal orange, random grey);
the new combinations blue. Checked with the dataviz validator (blue, aqua, orange: all pairs pass;
aqua below 3:1 contrast, so every line is labelled directly).

    .venv/bin/python scripts/e23_figures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import results_tex as rt  # noqa: E402
from e22_sets import COMBOS, LETTERS  # noqa: E402
from lth import RESULTS  # noqa: E402

SIZES = (20, 40, 60, 80, 100, 120)
HIGHLIGHT = {"DBMN": "o", "DBTMS": "D", "DBTLS": "^"}  # best at 40 per task, best peak, best at 40 shared
REF = {  # condition -> (colour, style, marker, label)
    "DBT": (rt.AQUA, "-", "o", "DBT (E13)"),
    "causal": (rt.ORANGE, "-", "s", "Causal"),
    "causal_nogen": (rt.ORANGE, (0, (4, 2)), "s", "Causal $-$ gen."),
    "random": (rt.GREY, "-", "v", "random"),
}
NAMES = {"D": "D  decodes task words", "B": "B  low BOS attention", "T": "T  attends to task words",
         "L": "L  attends to last token", "M": "M  attends to Q/A markers", "S": "S  attends to punct. + stopwords",
         "N": "N  attends to whole instruction"}


def save(fig, name: str) -> None:
    fig.savefig(rt.FIGURES / f"{name}.pdf")
    if os.environ.get("SYNTH_PNG"):
        fig.savefig(Path(os.environ["SYNTH_PNG"]) / f"{name}.png", dpi=200)
    plt.close(fig)


def label_ends(ax, ends: list) -> None:
    """Direct labels at the right end, nudged apart (as Figure e13_size)."""
    ends.sort()
    last = -1.0
    for y, lab, x in ends:
        y = max(y, last + 0.03)
        ax.annotate(lab, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=5.8,
                    color=rt.INK_2, annotation_clip=False)
        last = y


def figure_size() -> None:
    cur = pd.read_csv(RESULTS / "e23_best_sizes" / "curves.csv")
    e13 = pd.read_csv(RESULTS / "e13_capped" / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), sharey=True)
    for ax, scope, title in ((axes[0], "task", "(a) One set per task"), (axes[1], "shared", "(b) One set for all 14 tasks")):
        g = cur[cur.scope == scope]
        assert g.condition.nunique() == 15 and set(HIGHLIGHT) <= set(g.condition)
        ends = []
        for cond, r in g.groupby("condition"):
            r = r.sort_values("n")
            if cond in HIGHLIGHT:
                ax.plot(r.n, r.check1_task, color=rt.BLUE, lw=1.8, marker=HIGHLIGHT[cond], ms=4.5, mec=rt.SURFACE,
                        mew=0.8, zorder=4)
                ends.append((r.check1_task.iloc[-1], cond, r.n.iloc[-1]))
            else:
                ax.plot(r.n, r.check1_task, color=rt.BLUE, lw=0.8, alpha=0.28, zorder=2)
        for cond, (col, ls, mk, lab) in REF.items():
            r = e13[(e13.scope == scope) & (e13.condition == cond)].sort_values("n")
            assert list(r.n) == list(SIZES), cond
            ax.plot(r.n, r.check1_task, color=col, ls=ls, lw=1.6, marker=mk, ms=4.2, mec=rt.SURFACE, mew=0.8, zorder=3)
            ends.append((r.check1_task.iloc[-1], lab, r.n.iloc[-1]))
        label_ends(ax, ends)
        ax.set_xticks(SIZES)
        ax.set_xlim(14, SIZES[-1] + 6)
        ax.set_ylim(-0.02, 0.75)
        ax.set_xlabel("heads in the set (N)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=rt.GREY, alpha=0.18, lw=0.6)
        rt.style_axes(ax)
    axes[0].set_ylabel("check 1 accuracy, mean of 14 tasks")
    handles = [plt.Line2D([], [], color=rt.BLUE, lw=1.8, marker="o", ms=4),
               plt.Line2D([], [], color=rt.BLUE, lw=0.8, alpha=0.4),
               plt.Line2D([], [], color=rt.AQUA, lw=1.6, marker="o", ms=4),
               plt.Line2D([], [], color=rt.ORANGE, lw=1.6, marker="s", ms=4),
               plt.Line2D([], [], color=rt.GREY, lw=1.6, marker="v", ms=4)]
    labels = ["new combinations (labelled)", "other new combinations of the 15", "DBT (E13)",
              "Davidson et al. (causal; dashed: $-$ generic)", "random heads"]
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=6.6)
    fig.tight_layout(rect=(0, 0, 0.96, 0.85), w_pad=3.5)
    save(fig, "e23_size")


def figure_letters() -> None:
    s = pd.read_csv(RESULTS / "e22_all_combos" / "all_combos.csv")
    acc = s.set_index(["scope", "condition"]).check1_task
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    rows = list(LETTERS)[::-1]
    for scope, mk, off, fill in (("task", "o", 0.13, rt.INK_2), ("shared", "o", -0.13, rt.SURFACE)):
        for i, x in enumerate(rows):
            d = [acc[(scope, "".join(l for l in LETTERS if l in c + x))] - acc[(scope, c)]
                 for c in COMBOS if x not in c and "".join(l for l in LETTERS if l in c + x) in COMBOS]
            d = np.array(d)
            ax.plot(np.quantile(d, [0.25, 0.75]), [i + off] * 2, color=rt.INK_2, lw=1.0, alpha=0.6, zorder=2)
            ax.plot(d.mean(), i + off, mk, color=rt.INK_2, mfc=fill, ms=5, mew=1.0, zorder=3)
    ax.axvline(0, color=rt.GREY, lw=0.8, zorder=1)
    ax.set_yticks(range(len(rows)), [NAMES[x] for x in rows], fontsize=6.6)
    ax.set_xlabel("accuracy gain from adding the filter (40 heads)")
    ax.grid(axis="x", color=rt.GREY, alpha=0.18, lw=0.6)
    handles = [plt.Line2D([], [], color=rt.INK_2, marker="o", ls="", mfc=rt.INK_2, ms=5),
               plt.Line2D([], [], color=rt.INK_2, marker="o", ls="", mfc=rt.SURFACE, ms=5),
               plt.Line2D([], [], color=rt.INK_2, lw=1.0, alpha=0.6)]
    fig.legend(handles, ["per task (mean)", "shared (mean)", "middle half of the pairs"], fontsize=6.2, frameon=False,
               loc="upper center", ncol=3)
    rt.style_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "e22_letters")


def main() -> None:
    figure_size()
    figure_letters()


if __name__ == "__main__":
    main()
