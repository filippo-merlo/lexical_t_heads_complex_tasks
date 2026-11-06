"""E4 (paper/e4_heldout.tex): tables and figures of the accuracies, no statistical tests.

Reads ``results/heldout/fold{1,2}/patching_{simple,simple-wrong}.part-*.csv`` (``scripts/
heldout_patch.sh``) and the FV reference (``fold{f}/fv/patching_simple.csv``). For every fold,
check, task, set and alpha, the task accuracy is the mean over the five rule means of the
choosing half; the control accuracy is the control mean's. A set's accuracy at its
**cross-fitted alpha**: in fold 1, at the alpha with the highest task accuracy for the same task
and set in fold 2, and the reverse (ties: smaller |alpha|, then the positive one); random sets
at the alpha of the CC set of their size (``random20`` at ``cc20``'s, ``random40`` at ``cc40``'s,
``randomNt`` at ``cc``'s). The tables give the mean of the two folds. Secondary sets (fold 1
only) are given at their best alpha in fold 1. The figures give accuracy against alpha, mean of
the two folds (secondary sets: not drawn); random sets as the range of the five draws in check 1,
and in check 2, where they are run only at their cross-fitted alpha, as one point per draw and fold.

Writes ``results/heldout/accuracy_alpha.csv`` (long: fold, check, task, set, source, alpha,
accuracy), ``results/heldout/accuracy_crossfit.csv``, ``results/heldout/random_alphas_wrong_fold{f}.json``
(the cross-fitted alphas of the random sets in check 2, for ``patch.py --set-alphas``), the
tables ``paper/tables/heldout_*.tex`` and the figures ``paper/figures/heldout_*.pdf``.

    .venv/bin/python scripts/heldout_report.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import results_tex as rt  # noqa: E402  (palette, rcParams, table helpers)
from lth import RESULTS, simple  # noqa: E402

HELD = RESULTS / "heldout"
CHECKS = {"simple": "check 1", "simple-wrong": "check 2"}
MAIN = ("e1", "cc", "cc20", "cc40", "fv20", "fv40", "fvNt")
SECONDARY = ("ccclean", "ad20", "ad40", "fvuniv20", "fvuniv40")
RANDOM_OF = {"20": "cc20", "40": "cc40", "Nt": "cc"}
LABEL = {"e1": "E1", "cc": "CC", "cc20": "CC-20", "cc40": "CC-40", "fv20": "FV-20", "fv40": "FV-40",
         "fvNt": "FV-$N_t$", "random20": "Random-20", "random40": "Random-40", "randomNt": "Random-$N_t$",
         "ccclean": "CC-clean", "ad20": "A+D-20", "ad40": "A+D-40", "fvuniv20": "FV-univ-20",
         "fvuniv40": "FV-univ-40", "fv30": "FV-30 ($\\alpha{=}1$)"}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(curves, baselines): curves = fold, family, task, set, source, alpha, accuracy (task: mean
    over the five means; random draws keep their own name); baselines = unpatched accuracy and
    patched items per fold, family, task."""
    rows = []
    for fold in (1, 2):
        for family in CHECKS:
            for path in glob.glob(str(HELD / f"fold{fold}" / f"patching_{family}.part-*.csv")):
                rows.append(pd.read_csv(path).assign(fold=fold))
    d = pd.concat(rows, ignore_index=True)
    base = (d[d.source == "none"].drop_duplicates(["fold", "family", "task"])
            [["fold", "family", "task", "accuracy", "n_prompts"]])
    p = d[d.source != "none"]
    curves = (p.groupby(["fold", "family", "task", "heads", "source", "alpha"], as_index=False)
              .agg(accuracy=("accuracy", "mean"), means=("accuracy", "size"), n_heads=("n_heads", "first")))
    assert (curves[curves.source == "task"].means == 5).all(), "a set lacks one of the five means"
    return curves.rename(columns={"heads": "set"}), base


def best_alpha(c: pd.DataFrame) -> pd.DataFrame:
    """fold, family, task, set -> the alpha with the highest task accuracy (ties: |alpha|, sign)."""
    t = c[c.source == "task"].assign(key=lambda x: list(zip(-x.accuracy, x.alpha.abs(), -x.alpha)))
    t = t.sort_values("key").drop_duplicates(["fold", "family", "task", "set"])
    return t[["fold", "family", "task", "set", "alpha"]]


def crossfit(c: pd.DataFrame) -> pd.DataFrame:
    """Accuracy (task and control) of every set at its cross-fitted alpha, per fold."""
    best = best_alpha(c)
    other = best.assign(fold=3 - best.fold)  # fold f takes the alpha of fold 3 - f
    sets = c[["fold", "family", "task", "set"]].drop_duplicates()
    rand = sets[sets.set.str.startswith("random")]
    rand = rand.assign(base=rand.set.str.extract(r"random(\w+)_\d")[0].map(RANDOM_OF))
    chosen = pd.concat([sets[~sets.set.str.startswith("random")].assign(base=lambda x: x.set),
                        rand], ignore_index=True)
    chosen = chosen.merge(other.rename(columns={"set": "base"}), on=["fold", "family", "task", "base"], how="left")
    out = chosen.merge(c, on=["fold", "family", "task", "set", "alpha"], how="left")
    return out


def secondary_best(c: pd.DataFrame) -> pd.DataFrame:
    best = best_alpha(c[(c.fold == 1) & c.set.isin(SECONDARY)])
    return best.merge(c, on=["fold", "family", "task", "set", "alpha"])


def summary(x: pd.DataFrame, by_random: bool = True) -> pd.DataFrame:
    """task, set, source -> accuracy, mean over folds (random draws: mean over the five draws)."""
    if by_random:
        x = x.assign(set=x.set.str.replace(r"^(random\w+?)_\d$", r"\1", regex=True))
    return x.groupby(["family", "task", "set", "source"], as_index=False).accuracy.mean()


def fv30() -> pd.DataFrame:
    rows = []
    for fold in (1, 2):
        d = pd.read_csv(HELD / f"fold{fold}" / "fv" / "patching_simple.csv")
        d = d[(d.source != "none") & (d.alpha == 1)]
        rows.append(d.assign(fold=fold))
    d = pd.concat(rows)
    return d.groupby(["task", "source"], as_index=False).accuracy.mean()


def random_alphas(c: pd.DataFrame) -> None:
    """Check 2: the cross-fitted alpha of each random set (that of its CC set), per fold."""
    best = best_alpha(c[c.family == "simple-wrong"])
    heads = {f: json.loads((HELD / f"fold{f}" / "heads.json").read_text())["simple"] for f in (1, 2)}
    for fold in (1, 2):
        other = best[best.fold == 3 - fold].set_index(["task", "set"]).alpha
        out = {}
        for t in simple.TASKS:
            names = [n for _, n, _ in heads[fold][t]["sets"] if n.startswith("random")]
            out[t] = {n: [float(other[(t, RANDOM_OF[n.split("_")[0][6:]])])] for n in names}
        (HELD / f"random_alphas_wrong_fold{fold}.json").write_text(json.dumps(out, indent=1))


def tex_table(name: str, s: pd.DataFrame, family: str, columns: list[tuple[str, str]], base: pd.DataFrame,
              extra=None, note_items: bool = False) -> None:
    """Rows: tasks and their mean; columns: (set, source)."""
    def value(task, set_, source):
        if set_ == "unpatched":
            return base[(base.family == family) & (base.task == task)].accuracy.mean()
        if set_ == "fv30":
            return extra[(extra.task == task) & (extra.source == source)].accuracy.mean()
        r = s[(s.family == family) & (s.task == task) & (s.set == set_) & (s.source == source)]
        return r.accuracy.mean() if len(r) else np.nan

    grid = pd.DataFrame([[value(t, a, b) for a, b in columns] for t in simple.TASKS], index=list(simple.TASKS))
    header = ["Task" + (" & items" if note_items else "") + " & " +
              " & ".join(LABEL.get(a, a) + (" ctrl" if b == "control" else "") for a, b in columns) + r" \\"]
    lines = []
    for t in simple.TASKS:
        items = ""
        if note_items:
            n = base[(base.family == family) & (base.task == t)].sort_values("fold").n_prompts.tolist()
            items = " & " + "/".join(str(int(x)) for x in n)
        lines.append(rt.tex_name("simple", t) + items + " & " + " & ".join(rt.num(v) for v in grid.loc[t]) + r" \\")
    lines.append(r"\midrule")
    lines.append("Mean" + (" & " if note_items else "") + " & " + " & ".join(rt.num(v) for v in grid.mean()) + r" \\")
    spec = "l" + ("r" if note_items else "") + "r" * len(columns)
    (rt.TABLES / f"{name}.tex").write_text("% Generated by scripts/heldout_report.py from results/heldout/.\n"
                                            + rt.tabular(spec, header, lines))


def figure(name: str, c: pd.DataFrame, cf: pd.DataFrame, base: pd.DataFrame, n: str) -> None:
    """Accuracy against alpha (mean of the two folds), check 1 and check 2 per task; the sets of
    size ``n`` (Nt, 20 or 40), the random draws as a grey band (min-max of the draws' fold means)."""
    cc, fv = ("cc", "fvNt") if n == "Nt" else (f"cc{n}", f"fv{n}")
    series = [(cc, rt.BLUE, "CC"), (fv, rt.ORANGE, "FV")] + ([("e1", rt.AQUA, "E1")] if n == "Nt" else [])
    m = c.groupby(["family", "task", "set", "source", "alpha"], as_index=False).accuracy.mean()
    fig, axes = plt.subplots(7, 4, figsize=(6.3, 8.4), sharex=True, sharey=True)
    for i, task in enumerate(simple.TASKS):
        for j, family in enumerate(CHECKS):
            ax = axes[i // 2, 2 * (i % 2) + j]
            r = m[(m.family == family) & (m.task == task) & m.set.str.match(rf"random{n}_\d$") & (m.source == "task")]
            if family == "simple":
                band = r.groupby("alpha").accuracy.agg(["min", "max"])
                ax.fill_between(band.index, band["min"], band["max"], color=rt.GREY, alpha=0.3, lw=0, zorder=1)
            else:  # check 2: each draw only at its cross-fitted alpha, one point per draw and fold
                rc = cf[(cf.family == family) & (cf.task == task) & cf.set.str.match(rf"random{n}_\d$")
                        & (cf.source == "task")]
                ax.plot(rc.alpha, rc.accuracy, ls="none", marker="o", ms=2.5, color=rt.GREY, zorder=2)
            for s, colour, _ in series:
                for source, ls in (("task", "-"), ("control", rt.CONTROL_STYLE)):
                    r = m[(m.family == family) & (m.task == task) & (m.set == s) & (m.source == source)]
                    if len(r):
                        ax.plot(r.alpha, r.accuracy, color=colour, ls=ls, lw=1.1 if source == "task" else 0.9,
                                zorder=3)
            b = base[(base.family == family) & (base.task == task)].accuracy.mean()
            ax.axhline(b, color=rt.INK_2, lw=0.7, ls=(0, (4, 2)), zorder=2)
            ax.set_title(f"{rt.plot_name('simple', task)}, {CHECKS[family]}", loc="left", fontsize=7)
            ax.set_ylim(-0.03, 1.03)
            rt.style_axes(ax)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\alpha$")
        ax.set_xticks([-10, -5, 0, 5, 10])
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")
    size = "$N_t$" if n == "Nt" else n
    handles = [plt.Line2D([], [], color=col) for _, col, _ in series]
    labels = [f"{lab}, task mean" for *_, lab in series]
    handles += [plt.Line2D([], [], color=rt.INK_2, ls=rt.CONTROL_STYLE), plt.Line2D([], [], color=rt.INK_2, ls=(0, (4, 2)), lw=0.7),
                plt.Rectangle((0, 0), 1, 1, color=rt.GREY, alpha=0.3, lw=0)]
    handles.append(plt.Line2D([], [], color=rt.GREY, marker="o", ms=2.5, ls="none"))
    labels += ["control mean (same colour)", "unpatched", f"random, {size} heads (range of 5)",
               "random in check 2 (each draw and fold, at its alpha)"]
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(rt.FIGURES / f"{name}.pdf")
    plt.close(fig)


def main() -> None:
    c, base = load()
    c.to_csv(HELD / "accuracy_alpha.csv", index=False)
    cf = crossfit(c)
    cf.to_csv(HELD / "accuracy_crossfit.csv", index=False)
    random_alphas(c)
    s = summary(cf[~cf.set.isin(SECONDARY)])
    sec = summary(secondary_best(c), by_random=False)
    ref = fv30()
    for family, tag in (("simple", "check1"), ("simple-wrong", "check2")):
        cols = [("unpatched", "task"), ("e1", "task"), ("cc", "task"), ("fvNt", "task"), ("randomNt", "task"),
                ("e1", "control"), ("cc", "control"), ("fvNt", "control")]
        if family == "simple":
            cols.insert(5, ("fv30", "task"))
        tex_table(f"heldout_{tag}_nt", s, family, cols, base, extra=ref, note_items=True)
        cols = [(f"{k}{n}", "task") for n in ("20", "40") for k in ("cc", "fv", "random")]
        tex_table(f"heldout_{tag}_sizes", s, family, cols, base)
        tex_table(f"heldout_{tag}_secondary", sec, family, [("unpatched", "task")] + [(x, "task") for x in SECONDARY],
                  base[base.fold == 1])
    for n in ("Nt", "20", "40"):
        figure(f"heldout_curves_{n.lower()}", c, cf, base, n)
    mean = s[s.source == "task"].groupby(["family", "set"]).accuracy.mean().unstack(0).round(3)
    print(mean.to_string())


if __name__ == "__main__":
    main()
