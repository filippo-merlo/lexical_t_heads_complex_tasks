"""E6: accuracies of the difference-reading sets next to E4's sets (no statistical tests).

Uses E4's report code (``scripts/heldout_report.py``: the task accuracy is the mean over the five
rule means of the choosing half; a set's accuracy at its cross-fitted alpha, the alpha with the
highest task accuracy for the same task and set in the other fold; random sets at the alpha of
the set they were drawn around) on ``results/diff_heads/fold{1,2}`` and on ``results/heldout``.
E6's random sets are drawn around ``dcc20``, ``dcc40`` and ``dcc``; they are run in check 1 only. The
task-words-only sets (``dccT*``, ``scripts/diff_patch_T.sh``) have no random sets of their own.

Check: E4's ``cc20``, patched again in E6, must give E4's rows (every fold, check, task, source
and alpha), and the unpatched baselines must agree.

Writes ``results/diff_heads/accuracy_alpha.csv``, ``accuracy_crossfit.csv``, ``report.md``
(per task and check: E6 and E4 sets at the cross-fitted alpha, task accuracy and control) and
``paper/figures/diff_heads_check{1,2}.pdf`` (accuracy against alpha, mean of the folds).

    .venv/bin/python scripts/diff_report.py
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

import heldout_report as hr  # noqa: E402
from lth import RESULTS, simple  # noqa: E402

E6 = RESULTS / "diff_heads"
E6_MAIN = ("dcc", "dcc20", "dcc40", "dccA", "dccT", "dccT20", "dccT40", "dccTA")
E4_MAIN = ("e1", "cc", "cc20", "cc40", "fv20", "fv40", "fvNt")
E6_RANDOM_OF = {"20": "dcc20", "40": "dcc40", "Nt": "dcc"}
FIGURES = Path(__file__).resolve().parents[1] / "paper" / "figures"


def curves_of(folder: Path, random_of: dict):
    held, rand = hr.HELD, hr.RANDOM_OF
    hr.HELD, hr.RANDOM_OF = folder, random_of
    try:
        c, base = hr.load()
        cf = hr.crossfit(c)
    finally:
        hr.HELD, hr.RANDOM_OF = held, rand
    return c, base, cf


def main() -> None:
    c6, base6, cf6 = curves_of(E6, E6_RANDOM_OF)
    c4, base4, cf4 = curves_of(hr.HELD, hr.RANDOM_OF)

    # Check: cc20 again = E4's cc20; same baselines.
    key = ["fold", "family", "task", "source", "alpha"]
    mine = c6[c6.set == "cc20"].set_index(key).accuracy.sort_index()
    theirs = c4[c4.set == "cc20"].set_index(key).accuracy.reindex(mine.index)
    diff = float((mine - theirs).abs().max())
    b = base6.merge(base4, on=["fold", "family", "task"], suffixes=("", "_e4"))
    base_diff = float((b.accuracy - b.accuracy_e4).abs().max())
    check = f"cc20 patched again vs E4: max |diff| {diff:.4f} over {len(mine)} rows; baselines max |diff| {base_diff:.4f}"
    print(check)

    c6 = c6[c6.set != "cc20"]
    cf6 = cf6[cf6.set != "cc20"]
    both_c = pd.concat([c6.assign(study="E6"), c4[c4.set.isin(E4_MAIN) | c4.set.str.startswith("random")].assign(study="E4")])
    both_c.to_csv(E6 / "accuracy_alpha.csv", index=False)
    cf = pd.concat([cf6.assign(study="E6"), cf4[cf4.set.isin(E4_MAIN) | cf4.set.str.startswith("random")].assign(study="E4")])
    cf.to_csv(E6 / "accuracy_crossfit.csv", index=False)

    # Table: per task and check, mean over folds (random draws averaged).
    def label(row):
        if row.set.startswith("random"):
            size = row.set.split("_")[0][6:]
            return f"rand-{'d' if row.study == 'E6' else ''}cc{'' if size == 'Nt' else size}"
        return row.set

    s = cf.assign(name=cf.apply(label, axis=1)).groupby(["family", "task", "name", "source"], as_index=False).accuracy.mean()
    n_heads = (c6[c6.source == "task"].groupby(["fold", "task", "set"]).n_heads.first()
               .groupby(["task", "set"]).mean().unstack())
    order = [*E6_MAIN, "rand-dcc20", "rand-dcc40", "rand-dcc", *E4_MAIN, "rand-cc20", "rand-cc40", "rand-cc"]
    lines = ["# E6: sets from the instruction-difference readings, vs E4", "", f"Check: {check}.", "",
             "Accuracy at the cross-fitted alpha, mean of the two folds; random sets: mean of the five draws "
             "(check 1 only in E6). ctrl = the control mean at the same alpha.", ""]
    for family, title in hr.CHECKS.items():
        t = s[s.family == family].pivot_table(index="task", columns=["name", "source"], values="accuracy")
        names = [n for n in order if (n, "task") in t.columns]
        lines += [f"## {title}", "", "| task | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for task in simple.TASKS:
            cells = []
            for n in names:
                v, ctrl = t.loc[task, (n, "task")], t.loc[task].get((n, "control"), np.nan)
                cells.append(f"{v:.2f}" + (f" ({ctrl:.2f})" if not np.isnan(ctrl) else ""))
            lines.append(f"| {task} | " + " | ".join(cells) + " |")
        means = [t[(n, "task")].mean() for n in names]
        lines += ["| **mean** | " + " | ".join(f"**{m:.2f}**" for m in means) + " |", ""]
    lines += ["## Set sizes (mean of the two folds)", "", "| task | " + " | ".join(E6_MAIN) + " |",
              "|---|" + "---|" * len(E6_MAIN)]
    for task in simple.TASKS:
        lines.append(f"| {task} | " + " | ".join(f"{n_heads.loc[task, n]:.1f}" if n in n_heads.columns else "-"
                                                for n in E6_MAIN) + " |")
    (E6 / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Figures: accuracy against alpha, mean of the folds, task means; dcc vs cc vs fvNt.
    FIGURES.mkdir(parents=True, exist_ok=True)
    shown = [("E6", "dcc", "C0", "-"), ("E6", "dcc40", "C0", "--"), ("E6", "dccT", "C3", "-"), ("E4", "cc", "C1", "-"),
             ("E4", "cc40", "C1", "--"), ("E4", "fvNt", "C2", "-"), ("E4", "e1", "C7", ":")]
    for i, (family, title) in enumerate(hr.CHECKS.items(), 1):
        fig, axes = plt.subplots(3, 5, figsize=(15, 8), sharex=True)
        x = both_c[(both_c.family == family) & (both_c.source == "task")]
        for ax, task in zip(axes.flat, simple.TASKS):
            for study, name, color, style in shown:
                g = x[(x.study == study) & (x.set == name) & (x.task == task)].groupby("alpha").accuracy.mean()
                if len(g):
                    ax.plot(g.index, g.values, color=color, ls=style, lw=1.4, label=f"{study} {name}")
            ax.set_title(task, fontsize=9)
            ax.set_ylim(-0.02, 1.02)
        axes.flat[-1].axis("off")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"E6 vs E4, {title}: task accuracy against alpha (mean of the two folds)")
        fig.tight_layout()
        fig.savefig(FIGURES / f"diff_heads_check{i}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
