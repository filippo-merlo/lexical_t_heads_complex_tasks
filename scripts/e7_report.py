"""E7 (scripts/e7_sets.py, e7_patch.sh): accuracies of the most-active-word sets with and without
the BOS filter, E4 protocol, no statistical tests.

As in ``scripts/heldout_report.py``: task accuracy = mean over the five rule means of the choosing
half; a set's accuracy at its cross-fitted alpha (fold 1 at the best alpha of fold 2 and the
reverse; ties: smaller |alpha|, then the positive one), mean of the two folds. The random controls
of a filtered set (``random_drop_<list>_<i>``, ``random_layer_<list>_<i>``) are taken at the alpha
of ``<list>_att`` and averaged over the five draws. E4's sets on the same items and folds
(``results/heldout/accuracy_crossfit.csv``) are given as references: CC, CC-20, CC-40, FV-N_t.

Writes ``results/e7_attfilter/accuracy_alpha.csv``, ``accuracy_crossfit.csv``, ``report.md`` and
``alpha_curves.pdf`` (check 1 and 2, mean over tasks and folds, against alpha).

    .venv/bin/python scripts/e7_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import heldout_report as hr  # noqa: E402
from lth import RESULTS, simple  # noqa: E402

OUT = RESULTS / "e7_attfilter"
MAIN = ("raw", "raw_att", "clean", "clean_att")
REFS = ("cc", "cc20", "cc40", "fvNt")
CHECKS = {"simple": "check 1", "simple-wrong": "check 2"}


def md(g: pd.DataFrame) -> str:
    """A DataFrame as a Markdown table, two decimals (no tabulate in the venv)."""
    head = "| " + " | ".join([str(g.index.name or ""), *map(str, g.columns)]) + " |"
    rule = "|" + "---|" * (len(g.columns) + 1)
    body = ["| " + " | ".join([str(i), *("" if pd.isna(v) else f"{v:.2f}" for v in r)]) + " |"
            for i, r in zip(g.index, g.to_numpy())]
    return "\n".join([head, rule, *body])


def crossfit(c: pd.DataFrame) -> pd.DataFrame:
    best = hr.best_alpha(c)
    other = best.assign(fold=3 - best.fold)
    sets = c[["fold", "family", "task", "set"]].drop_duplicates()
    base = sets.set.str.extract(r"^random_(?:drop|layer)_(\w+)_\d$")[0]
    sets = sets.assign(base=base.where(base.isna(), base + "_att").fillna(sets.set))
    out = sets.merge(other.rename(columns={"set": "base"}), on=["fold", "family", "task", "base"], how="left")
    return out.merge(c, on=["fold", "family", "task", "set", "alpha"], how="left")


def main() -> None:
    hr.HELD = OUT
    curves, base = hr.load()
    curves.to_csv(OUT / "accuracy_alpha.csv", index=False)
    cf = crossfit(curves)
    cf.to_csv(OUT / "accuracy_crossfit.csv", index=False)
    cf = cf.assign(set=cf.set.str.replace(r"^(random_(?:drop|layer)_\w+?)_\d$", r"\1", regex=True))
    s = cf.groupby(["family", "task", "set", "source"], as_index=False).accuracy.mean()
    e4 = pd.read_csv(RESULTS / "heldout" / "accuracy_crossfit.csv")
    e4 = e4[e4.set.isin(REFS)].groupby(["family", "task", "set", "source"], as_index=False).accuracy.mean()
    s = pd.concat([s, e4.assign(set="E4 " + e4.set)])
    cols = ["unpatched", *MAIN, "random_drop_raw", "random_layer_raw", "random_drop_clean",
            "random_layer_clean", *("E4 " + r for r in REFS)]

    lines = ["# E7: most-active-word heads, with and without the BOS filter",
             "", "Accuracy at the cross-fitted alpha, mean of the two folds (random controls: also mean of "
             "five draws, at the alpha of their filtered set). E4 columns: E4's sets on the same items.", ""]
    for family, check in CHECKS.items():
        for source in ("task", "control"):
            g = s[(s.family == family) & (s.source == source)].pivot_table(index="task", columns="set", values="accuracy")
            g = g.reindex(list(simple.TASKS))
            if source == "task":
                g["unpatched"] = base[base.family == family].groupby("task").accuracy.mean()
            g = g[[c for c in cols if c in g.columns]]
            g.loc["mean"] = g.mean()
            lines += [f"## {check}, {source} mean", "", md(g), ""]
    lines += ["## Set sizes (mean of the two folds)", "",
              md(curves[curves.set.isin(MAIN)].groupby(["task", "set", "fold"]).n_heads.first()
              .groupby(["task", "set"]).mean().unstack().reindex(list(simple.TASKS))[list(MAIN)]), ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    t = curves[curves.source == "task"].assign(
        set=lambda x: x.set.str.replace(r"^(random_(?:drop|layer)_\w+?)_\d$", r"\1", regex=True))
    m = t.groupby(["family", "set", "alpha"]).accuracy.mean().reset_index()
    for ax, (family, check) in zip(axes, CHECKS.items()):
        for name, style in [("raw", "C0-"), ("raw_att", "C0--"), ("clean", "C1-"), ("clean_att", "C1--"),
                            ("random_drop_raw", "C0:"), ("random_drop_clean", "C1:")]:
            r = m[(m.family == family) & (m.set == name)].sort_values("alpha")
            if len(r):
                ax.plot(r.alpha, r.accuracy, style, marker="o", ms=3, label=name)
        ax.axhline(base[base.family == family].accuracy.mean(), color="grey", lw=0.8)
        ax.set_title(check)
        ax.set_xlabel("alpha")
    axes[0].set_ylabel("task accuracy (mean over tasks, folds)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "alpha_curves.pdf")


if __name__ == "__main__":
    main()
