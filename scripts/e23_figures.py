"""Figure for E22 in paper/head-sets-simple (run after scripts/e22_report.py).

``paper/figures/e22_filters.pdf``: at 40 heads, (a) each of the seven filters alone, (b) what each adds:
accuracy of a combination with the filter minus that of the same combination without it, over every
such pair among the 79 combinations; per task and shared. The best combinations at every size are
drawn in Figure e13_size (``scripts/e13_report.py``).

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

NAMES = {"D": "D  decodes task words", "B": "B  low BOS attention", "T": "T  attends to task words",
         "L": "L  attends to last token", "M": "M  attends to Q/A markers", "S": "S  attends to punct. + stopwords",
         "N": "N  attends to whole instruction"}


def save(fig, name: str) -> None:
    fig.savefig(rt.FIGURES / f"{name}.pdf")
    if os.environ.get("SYNTH_PNG"):
        fig.savefig(Path(os.environ["SYNTH_PNG"]) / f"{name}.png", dpi=200)
    plt.close(fig)


def figure_filters() -> None:
    """(a) each filter alone at 40 heads; (b) what it adds to the combinations without it."""
    s = pd.read_csv(RESULTS / "e22_all_combos" / "all_combos.csv")
    acc = s.set_index(["scope", "condition"]).check1_task
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True, gridspec_kw={"width_ratios": [1, 1.25]})
    rows = list(LETTERS)[::-1]
    for scope, off, fill in (("task", 0.14, rt.INK_2), ("shared", -0.14, rt.SURFACE)):
        for i, x in enumerate(rows):
            axes[0].plot([0, acc[(scope, x)]], [i + off] * 2, color=rt.INK_2, lw=1.0, alpha=0.35, zorder=2)
            axes[0].plot(acc[(scope, x)], i + off, "o", color=rt.INK_2, mfc=fill, ms=5, mew=1.0, zorder=3)
            d = np.array([acc[(scope, "".join(l for l in LETTERS if l in c + x))] - acc[(scope, c)]
                          for c in COMBOS if x not in c and "".join(l for l in LETTERS if l in c + x) in COMBOS])
            axes[1].plot(np.quantile(d, [0.25, 0.75]), [i + off] * 2, color=rt.INK_2, lw=1.0, alpha=0.6, zorder=2)
            axes[1].plot(d.mean(), i + off, "o", color=rt.INK_2, mfc=fill, ms=5, mew=1.0, zorder=3)
    axes[0].set_yticks(range(len(rows)), [NAMES[x] for x in rows], fontsize=6.6)
    axes[0].set_xlim(0, 0.6)
    axes[0].set_xlabel("accuracy of the filter alone")
    axes[0].set_title("(a) Alone", loc="left")
    axes[1].axvline(0, color=rt.GREY, lw=0.8, zorder=1)
    axes[1].set_xlabel("accuracy with the filter $-$ without it")
    axes[1].set_title("(b) Added to the other filters", loc="left")
    for ax in axes:
        ax.grid(axis="x", color=rt.GREY, alpha=0.18, lw=0.6)
        rt.style_axes(ax)
    handles = [plt.Line2D([], [], color=rt.INK_2, marker="o", ls="", mfc=rt.INK_2, ms=5),
               plt.Line2D([], [], color=rt.INK_2, marker="o", ls="", mfc=rt.SURFACE, ms=5),
               plt.Line2D([], [], color=rt.INK_2, lw=1.0, alpha=0.6)]
    fig.legend(handles, ["one set per task", "one set for all 14 tasks", "(b) middle half of the pairs"], fontsize=6.4,
               frameon=False, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    save(fig, "e22_filters")


def main() -> None:
    figure_filters()


if __name__ == "__main__":
    main()
