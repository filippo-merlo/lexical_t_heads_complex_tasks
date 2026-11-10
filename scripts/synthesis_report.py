"""Synthesis of the simple-task head-selection experiments (paper/head-sets-simple.tex).

Puts every held-out head set (E4, E6, E7, E8, E10) on one footing: J-head, rule style, E4's item
split and folds, rule means of the choosing half, accuracy at the **cross-fitted alpha over
alpha in {1, 2}** (fold 1 at the better of the two on fold 2 and the reverse; ties to alpha 1),
mean of the two folds. E8 and E10 were run at alpha 1 and 2 only; E4, E6 and E7 at ten values
that include both, so their numbers here are lower than in their own reports where those picked
alpha >= 3. Random sets are scored at the alpha of the set they are matched to. Secondary E4 sets
(FV-univ, attend + decode, CC-clean) were run in fold 1 only: given at their better alpha of the
two there (not cross-fitted). Davidson et al.'s 30-head function vector (residual stream, layer
11) on the same held-out items at their alpha = 1. Nothing is re-run; every number is read from
the saved accuracy files.

Writes ``results/synthesis/{sets,per_task}.csv``, the tables ``paper/tables/synth_*.tex`` and the
figures ``paper/figures/synth_*.pdf``.

    .venv/bin/python scripts/synthesis_report.py
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

import results_tex as rt  # noqa: E402  (palette, rcParams, names)
from lth import RESULTS, simple  # noqa: E402

OUT = RESULTS / "synthesis"
ALPHAS = (1.0, 2.0)
CHECKS = {"simple": "check1", "simple-wrong": "check2"}

# Task groups, in the order of the rows of the per-task figure and tables.
GROUPS = {
    "Word relations": ("antonym", "synonym"),
    "Morphology": ("singular-plural", "present-past"),
    "Facts": ("country-capital", "country-currency", "landmark-country", "park-country", "product-company"),
    "Sequences": ("next_item", "prev_item"),
    "Translations": ("english-french", "english-german", "english-spanish"),
}
TASK_ORDER = [t for ts in GROUPS.values() for t in ts]
assert sorted(TASK_ORDER) == sorted(simple.TASKS)

# (source folder, set key) -> (label, approach, scope). Approach = how heads are scored; scope =
# one set per task or one set for all tasks.
LEXICAL, CAUSAL, ATTENTION, RANDOM = "lexical", "causal", "attention", "random"
SETS = [
    ("heldout", "e1", "E1: Yang term list", LEXICAL, "task"),
    ("heldout", "cc20", "CC-20", LEXICAL, "task"),
    ("heldout", "cc40", "CC-40", LEXICAL, "task"),
    ("heldout", "cc", "CC-pooled", LEXICAL, "task"),
    ("e7_attfilter", "raw", "E7 raw words", LEXICAL, "task"),
    ("e7_attfilter", "raw_att", "E7 raw words + B", LEXICAL, "task"),
    ("diff_heads", "dcc", "E6 instruction diff.", LEXICAL, "task"),
    ("heldout", "fv20", "FV-20", CAUSAL, "task"),
    ("heldout", "fv40", "FV-40", CAUSAL, "task"),
    ("heldout", "fvNt", "FV-$N_t$", CAUSAL, "task"),
    ("e8_filters", "D", "E8 D", ATTENTION, "task"),
    ("e8_filters", "DB", "E8 DB", ATTENTION, "task"),
    ("e8_filters", "DBT", "E8 DBT", ATTENTION, "task"),
    ("e8_filters", "BT", "E8 BT", ATTENTION, "task"),
    ("e8_filters", "T", "E8 T", ATTENTION, "task"),
    ("fv30", "fv30", "Davidson FV-30 ($\\alpha{=}1$)", CAUSAL, "shared"),
    ("heldout", "fvuniv20", "FV-univ-20", CAUSAL, "shared"),
    ("heldout", "fvuniv40", "FV-univ-40", CAUSAL, "shared"),
    ("e10_shared", "rank_DBT", "E10 shared DBT", ATTENTION, "shared"),
    ("e10_shared", "mean_BT", "E10 shared BT", ATTENTION, "shared"),
    ("e10_shared", "rank_DB", "E10 shared DB", ATTENTION, "shared"),
    ("heldout", "random20", "Random-20", RANDOM, "task"),
    ("heldout", "random40", "Random-40", RANDOM, "task"),
    ("heldout", "randomNt", "Random-$N_t$", RANDOM, "task"),
    ("e7_attfilter", "random_drop_raw", "Random from E7 raw", RANDOM, "task"),
]
ONE_FOLD = {"fvuniv20", "fvuniv40", "ad20", "ad40", "ccclean"}
RANDOM_BASE = {  # random set -> the set whose cross-fitted alpha it takes
    "random20": "cc20", "random40": "cc40", "randomNt": "cc",
    "random_drop_raw": "raw_att", "random_layer_raw": "raw_att",
    "random_drop_clean": "clean_att", "random_layer_clean": "clean_att"}
FOLDERS = ("heldout", "diff_heads", "e7_attfilter", "e8_filters", "e10_shared")
COLOUR = {LEXICAL: rt.BLUE, CAUSAL: rt.ORANGE, ATTENTION: rt.AQUA, RANDOM: rt.GREY}


def curves() -> pd.DataFrame:
    """Every held-out set at alpha 1 and 2: folder, fold, family, task, set (random draws pooled
    under their family name, keeping ``draw``), source, alpha, accuracy, n_heads."""
    out = []
    for folder in FOLDERS:
        d = pd.read_csv(RESULTS / folder / "accuracy_alpha.csv")
        d = d[d.alpha.isin(ALPHAS)].copy()
        if folder == "diff_heads":  # E4 sets re-patched there: take them from E4 itself
            d = d[d.set.str.startswith("dcc")]
        d["draw"] = d.set.str.extract(r"_(\d)$")[0].where(d.set.str.match(r"^random"))
        d["set"] = d.set.where(d.draw.isna(), d.set.str.replace(r"_\d$", "", regex=True))
        out.append(d.assign(folder=folder)[["folder", "fold", "family", "task", "set", "draw", "source", "alpha",
                                            "accuracy", "n_heads"]])
    return pd.concat(out, ignore_index=True)


def crossfit(c: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of every (folder, fold, family, task, set[, draw], source) at its cross-fitted
    alpha; one-fold sets at their best alpha in fold 1."""
    t = c[(c.source == "task") & c.draw.isna()]
    t = t.assign(key=list(zip(-t.accuracy, t.alpha.abs())))
    best = t.sort_values("key").drop_duplicates(["folder", "fold", "family", "task", "set"])
    best = best[["folder", "fold", "family", "task", "set", "alpha"]]
    two = best[~best.set.isin(ONE_FOLD)].assign(fold=lambda x: 3 - x.fold)  # take the other fold's
    one = best[best.set.isin(ONE_FOLD)]
    chosen = pd.concat([two, one]).rename(columns={"set": "base"})
    keys = c[["folder", "fold", "family", "task", "set"]].drop_duplicates()
    keys = keys.assign(base=keys.set.map(RANDOM_BASE).fillna(keys.set))
    keys = keys.merge(chosen, on=["folder", "fold", "family", "task", "base"])
    return keys.merge(c, on=["folder", "fold", "family", "task", "set", "alpha"])


def fv30() -> pd.DataFrame:
    """Davidson et al.'s 30-head FV on the held-out items of each fold, alpha = 1 (check 1 only)."""
    rows = []
    for fold in (1, 2):
        d = pd.read_csv(RESULTS / "heldout" / f"fold{fold}" / "fv" / "patching_simple.csv")
        d = d[(d.source != "none") & (d.alpha == 1)]
        rows.append(d.assign(fold=fold, folder="fv30", family="simple", set="fv30", draw=np.nan)
                    [["folder", "fold", "family", "task", "set", "draw", "source", "alpha", "accuracy", "n_heads"]])
    return pd.concat(rows, ignore_index=True)


def baselines() -> pd.DataFrame:
    rows = []
    for fold in (1, 2):
        for family in CHECKS:
            for path in sorted((RESULTS / "heldout" / f"fold{fold}").glob(f"patching_{family}.part-*.csv")):
                d = pd.read_csv(path)
                rows.append(d[d.source == "none"].assign(fold=fold)[["fold", "family", "task", "accuracy", "n_prompts"]])
    return pd.concat(rows).drop_duplicates(["fold", "family", "task"])


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    c = curves()
    cf = pd.concat([crossfit(c), fv30()], ignore_index=True)
    # per task: mean over draws, then over folds
    per = (cf.groupby(["folder", "fold", "family", "task", "set", "source"], as_index=False)
           .agg(accuracy=("accuracy", "mean"), n_heads=("n_heads", "mean"), alpha=("alpha", "first")))
    # check 2 random draws were run only at their old (ten-alpha) cross-fitted alpha: incomplete here
    per = per[~((per.family == "simple-wrong") & per.set.str.startswith("random"))]
    fold1 = per[per.fold == 1][["folder", "family", "task", "set", "source", "accuracy"]].rename(
        columns={"accuracy": "fold1"})
    per = per.groupby(["folder", "family", "task", "set", "source"], as_index=False).agg(
        accuracy=("accuracy", "mean"), n_heads=("n_heads", "mean"), folds=("fold", "nunique"))
    per = per.merge(fold1, on=["folder", "family", "task", "set", "source"], how="left")
    return c, cf, per


def summary(per: pd.DataFrame) -> pd.DataFrame:
    """One row per curated set: heads (median and range over tasks), mean over the 14 tasks of
    task and control accuracy in both checks."""
    rows = []
    for folder, key, label, approach, scope in SETS:
        p = per[(per.folder == folder) & (per.set == key)]
        if p.empty:
            raise SystemExit(f"missing {folder}/{key}")
        heads = p[(p.family == "simple") & (p.source == "task")].set_index("task").n_heads
        r = dict(folder=folder, set=key, label=label, approach=approach, scope=scope,
                 heads_median=heads.median(), heads_min=heads.min(), heads_max=heads.max(),
                 tasks=heads.size, folds=int(p.folds.max()))
        q = p[(p.family == "simple") & (p.source == "task")]
        r["check1_fold1"] = q.fold1.mean()
        for family, tag in CHECKS.items():
            for source in ("task", "control"):
                q = p[(p.family == family) & (p.source == source)]
                r[f"{tag}_{source}"] = q.accuracy.mean() if len(q) == 14 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def all_combos(per: pd.DataFrame) -> pd.DataFrame:
    """Every E8 and E10 set: heads (median over tasks) and check-1 task mean over tasks."""
    p = per[per.folder.isin(["e8_filters", "e10_shared"]) & (per.family == "simple") & (per.source == "task")]
    g = p.groupby(["folder", "set"]).agg(acc=("accuracy", "mean"), heads=("n_heads", "median"),
                                         tasks=("task", "nunique")).reset_index()
    return g[g.tasks == 14]


# --------------------------------------------------------------------------- tables

def tex_main(s: pd.DataFrame) -> None:
    heads = {"lexical": "Decoded words (lens)", "causal": "Causal score (Davidson et al.)",
             "attention": "Attention and decoding filters", "random": "Random heads"}
    lines = []
    for scope, title in (("task", "One set per task"), ("shared", "One set for all 14 tasks")):
        lines.append(r"\multicolumn{7}{@{}l}{\textit{" + title + r"}} \\")
        for approach in (LEXICAL, CAUSAL, ATTENTION, RANDOM):
            g = s[(s.scope == scope) & (s.approach == approach)]
            if g.empty:
                continue
            lines.append(r"\quad " + heads[approach] + r" & & & & & & \\")
            for r in g.itertuples():
                size = (f"{r.heads_median:.0f}" if r.heads_min == r.heads_max
                        else f"{r.heads_median:.0f} ({r.heads_min:.0f}--{r.heads_max:.0f})")
                mark = r"$^{\dagger}$" if r.folds == 1 else ""
                lines.append(r"\qquad " + r.label + mark + f" & {size} & " + " & ".join(
                    rt.num(getattr(r, k)) for k in ("check1_task", "check1_fold1", "check1_control", "check2_task",
                                                    "check2_control")) + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    header = [r"Head set & Heads & \multicolumn{3}{c}{Check 1} & \multicolumn{2}{c}{Check 2} \\",
              r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
              r" & median (range) & task & fold 1 & control & task & control \\"]
    text = "\n".join([r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule", *header, r"\midrule", *lines,
                      r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / "synth_main.tex").write_text("% Generated by scripts/synthesis_report.py.\n" + text + "\n")


PER_TASK_SETS = [  # (folder, key, short column label)
    ("heldout", "e1", "E1"), ("heldout", "cc", "CC"), ("e7_attfilter", "raw_att", "E7 +B"),
    ("heldout", "fv20", "FV-20"), ("heldout", "fvNt", "FV-$N_t$"),
    ("e8_filters", "T", "T"), ("e8_filters", "DB", "DB"), ("e8_filters", "DBT", "DBT"),
    ("fv30", "fv30", "FV-30"), ("heldout", "fvuniv20", "FV-univ-20$^{\\dagger}$"),
    ("e10_shared", "rank_DBT", "E10 DBT"), ("e10_shared", "mean_BT", "E10 BT"),
]


def per_task_grid(per: pd.DataFrame, family: str, source: str = "task") -> pd.DataFrame:
    g = {}
    for folder, key, label in PER_TASK_SETS:
        p = per[(per.folder == folder) & (per.set == key) & (per.family == family) & (per.source == source)]
        g[label] = p.set_index("task").accuracy.reindex(TASK_ORDER)
    return pd.DataFrame(g)


def tex_per_task(per: pd.DataFrame, base: pd.DataFrame, family: str, name: str) -> None:
    g = per_task_grid(per, family)
    sizes = per_task_grid(per.assign(accuracy=per.n_heads), "simple")
    b = base[base.family == family].groupby("task").accuracy.mean().reindex(TASK_ORDER)
    n = base[base.family == family].sort_values("fold").groupby("task").n_prompts.apply(
        lambda x: "/".join(str(int(v)) for v in x)).reindex(TASK_ORDER)
    cols = list(g.columns)
    lines = []
    for group, tasks in GROUPS.items():
        lines.append(r"\multicolumn{" + str(len(cols) + 3) + r"}{@{}l}{\textit{" + group + r"}} \\")
        for t in tasks:
            vals = g.loc[t]
            top = vals[[c for c in cols if "dagger" not in c]].max()  # bold: two-fold sets only
            cells = [(r"\textbf{" + rt.num(v) + "}") if (not np.isnan(v) and v == top and top > 0) else rt.num(v)
                     for v in vals]
            lines.append(rt.tex_name("simple", t) + f" & {n[t]} & {rt.num(b[t])} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append("Mean & & " + rt.num(b.mean()) + " & " + " & ".join(rt.num(v) for v in g.mean()) + r" \\")
    lines.append(r"Heads (median) & & & " + " & ".join(f"{v:.0f}" for v in sizes.median()) + r" \\")
    header = [r"Task & Items & Unp. & " + " & ".join(cols) + r" \\"]
    text = "\n".join([r"\begin{tabular}{@{}lcc" + "c" * len(cols) + r"@{}}", r"\toprule", *header, r"\midrule",
                      *lines, r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / f"{name}.tex").write_text("% Generated by scripts/synthesis_report.py.\n" + text + "\n")


def tex_groups(per: pd.DataFrame) -> None:
    """Check 1: mean accuracy per task group for the per-task-figure sets."""
    g = per_task_grid(per, "simple")
    rows = []
    for group, tasks in GROUPS.items():
        vals = g.loc[list(tasks)].mean()
        top = vals[[c for c in g.columns if "dagger" not in c]].max()
        rows.append(group + f" ({len(tasks)}) & " + " & ".join(
            (r"\textbf{" + rt.num(v) + "}") if v == top else rt.num(v) for v in vals) + r" \\")
    header = [r"Task group & " + " & ".join(g.columns) + r" \\"]
    text = "\n".join([r"\begin{tabular}{@{}l" + "c" * len(g.columns) + r"@{}}", r"\toprule", *header, r"\midrule",
                      *rows, r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / "synth_groups.tex").write_text("% Generated by scripts/synthesis_report.py.\n" + text + "\n")


def tex_readouts() -> None:
    """E1 (not held out): heads found and mean causal effect over the 14 tasks (best alpha of the
    15, accuracy averaged over the style's five means, minus the unpatched accuracy) per readout."""
    d = rt.load()
    eff = pd.DataFrame([(f, st, h, t, v) for (f, t, st, h, src), v in rt.effects(d["patching"]).items()
                        if f in CHECKS and src == "task"], columns=["family", "style", "heads", "task", "effect"])
    mean = eff.groupby(["family", "style", "heads"]).effect.mean()
    heads = d["heads"]["simple"]
    lines = []
    for style in ("rule", "example"):
        sets = {r: [{tuple(x[:2]) for x in heads[t][style][r]["heads"]} for t in heads] for r in ("logithead", "jhead")}
        n_l = sum(map(len, sets["logithead"]))
        n_j = sum(map(len, sets["jhead"]))
        n_both = sum(len(a & b) for a, b in zip(sets["logithead"], sets["jhead"]))
        cells = [str(n_l), str(n_j), str(n_both)]
        for family in CHECKS:
            cells += [rt.num(mean[(family, style, r)]) for r in ("logithead", "jhead", "jhead_only")]
        lines.append(style.capitalize() + " & " + " & ".join(cells) + r" \\")
    header = [r" & \multicolumn{3}{c}{Heads (sum over tasks)} & \multicolumn{3}{c}{Check 1 effect} & "
              r"\multicolumn{3}{c}{Check 2 effect} \\",
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
              r"Style & L & J & L$\cap$J & L & J & J only & L & J & J only \\"]
    text = "\n".join([r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule", *header, r"\midrule", *lines,
                      r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / "synth_readouts.tex").write_text("% Generated by scripts/synthesis_report.py.\n" + text + "\n")


# --------------------------------------------------------------------------- figures

def preview(fig, name: str) -> None:
    """Also write a PNG where ``SYNTH_PNG`` points (for looking at the figure; not a deliverable)."""
    import os
    if os.environ.get("SYNTH_PNG"):
        fig.savefig(Path(os.environ["SYNTH_PNG"]) / f"{name}.png", dpi=200)


DAV = "Davidson FV-30 ($\\alpha{=}1$)"
LABEL_OFFSET = {  # check -> label -> (dx points, dy points, ha) for the size figure
    "check1": {
        "E1: Yang term list": (0, -9, "center"), "CC-20": (-6, 0, "right"), "CC-40": (0, -9, "center"),
        "CC-pooled": (0, -9, "center"), "E7 raw words + B": (6, -5, "left"), "E6 instruction diff.": (6, -4, "left"),
        "FV-20": (-6, 0, "right"), "FV-40": (6, -1, "left"), "FV-$N_t$": (6, 2, "left"),
        "E8 D": (0, 8, "center"), "E8 DB": (4, 7, "left"), "E8 DBT": (-4, 7, "right"), "E8 BT": (6, -4, "left"),
        "E8 T": (6, 0, "left"), DAV: (-7, 1, "right"), "E10 shared DBT": (-7, -1, "right"),
        "FV-univ-20": (-7, 0, "right"), "Random-$N_t$": (6, 0, "left")},
    "check2": {
        "E1: Yang term list": (6, -3, "left"), "CC-pooled": (0, -9, "center"), "FV-$N_t$": (0, -9, "center"),
        "E8 T": (0, 8, "center"), "E10 shared DBT": (0, 9, "center"), "FV-univ-20": (-7, 0, "right")},
}
NAMES = {"E1: Yang term list": "E1 (Yang list)", DAV: "Davidson FV-30", "FV-univ-20": "FV-univ-20*",
         "E7 raw words + B": "E7 words + B"}
SIZE_PLOT = ["E1: Yang term list", "CC-20", "CC-40", "CC-pooled", "E7 raw words + B", "E6 instruction diff.",
             "FV-20", "FV-40", "FV-$N_t$", "E8 D", "E8 DB", "E8 DBT", "E8 BT", "E8 T",
             DAV, "FV-univ-20", "E10 shared DBT", "Random-20", "Random-40", "Random-$N_t$"]
LINES = [("CC-20", "CC-40", "CC-pooled"), ("FV-20", "FV-40", "FV-$N_t$"), ("Random-20", "Random-40", "Random-$N_t$")]


def figure_size(s: pd.DataFrame, combos: pd.DataFrame) -> None:
    """Mean check-1 (left) and check-2 (right) accuracy over the 14 tasks against the number of
    heads (median over tasks), log x. Colour = how heads are scored; filled = one set per task,
    open = one set for all tasks. Faint dots: every E8 and E10 combination."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharex=True)
    idx = s.set_index("label")
    for ax, tag, title in ((axes[0], "check1", "Check 1: zero-shot prompts"),
                           (axes[1], "check2", "Check 2: wrong 1-shot prompts")):
        if tag == "check1":
            for folder, mk, col in (("e8_filters", "o", rt.AQUA), ("e10_shared", "o", rt.AQUA)):
                g = combos[combos.folder == folder]
                ax.scatter(g.heads, g.acc, s=9, marker=mk, facecolor=col if folder == "e8_filters" else "none",
                           edgecolor=col, alpha=0.25, lw=0.6, zorder=1)
        for chain in LINES:
            r = idx.loc[list(chain)]
            ax.plot(r.heads_median, r[f"{tag}_task"], color=COLOUR[r.approach.iloc[0]], lw=1.0, alpha=0.7, zorder=2)
        for label in SIZE_PLOT:
            r = idx.loc[label]
            y = r[f"{tag}_task"]
            if np.isnan(y):
                continue
            col = COLOUR[r.approach]
            ax.scatter(r.heads_median, y, s=34, marker="s" if label == "E1: Yang term list" else "o",
                       facecolor=col if r.scope == "task" else rt.SURFACE, edgecolor=col if r.scope == "shared"
                       else rt.SURFACE, lw=1.6 if r.scope == "shared" else 0.8, zorder=4)
            if label in LABEL_OFFSET[tag]:
                dx, dy, ha = LABEL_OFFSET[tag][label]
                name = NAMES.get(label, label).replace("$N_t$", "$N_t$")
                ax.annotate(name, (r.heads_median, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                            va="center", fontsize=6.3, color=rt.INK_2, zorder=5)
        ax.set_xscale("log")
        ticks = [10, 20, 30, 40, 60, 100, 150, 250]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(8, 340)
        ax.set_ylim(-0.02, 0.66 if tag == "check1" else 0.5)
        ax.set_xlabel("heads in the set (median over tasks)")
        ax.set_title(title, loc="left")
        rt.style_axes(ax)
    axes[0].set_ylabel("accuracy, mean of 14 tasks")
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=5, color=COLOUR[a]) for a in (LEXICAL, CAUSAL, ATTENTION, RANDOM)]
    labels = ["decoded words (lens)", "causal score (FV)", "attention + decoding filters", "random heads"]
    handles += [plt.Line2D([], [], ls="none", marker="o", ms=5, mfc=rt.SURFACE, mec=rt.INK_2, mew=1.4),
                plt.Line2D([], [], ls="none", marker="o", ms=3.5, color=rt.AQUA, alpha=0.3)]
    labels += ["open: one set for all tasks", "faint: other E8 / E10 combinations"]
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=6.8)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(rt.FIGURES / "synth_size.pdf")
    preview(fig, "synth_size")
    plt.close(fig)


def figure_tasks(per: pd.DataFrame) -> None:
    """Check 1 per task (rows, grouped) and head set (columns): accuracy as a one-hue ramp, values
    printed."""
    g = per_task_grid(per, "simple")
    sizes = per_task_grid(per.assign(accuracy=per.n_heads), "simple").median()
    data = g.to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", [rt.SURFACE, "#cde2fb", "#7fb2ee", rt.BLUE, "#174a8a"])
    ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    two_fold = np.array(["dagger" not in c for c in g.columns])
    for i in range(data.shape[0]):
        top = np.nanmax(data[i][two_fold])
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}".replace("0.", ".") if v < 1 else "1", ha="center", va="center", fontsize=6.3,
                    color=rt.SURFACE if v > 0.55 else rt.INK, fontweight="bold" if v == top and top > 0 else "normal")
    ax.set_xticks(range(data.shape[1]))
    names = [c.replace("$N_t$", "Nt").replace(chr(92) + " ", " ").replace("-20$^{" + chr(92) + "dagger}$", "*")
             for c in g.columns]
    ax.set_xticklabels([f"{n}\n{sizes[c]:.0f}" for n, c in zip(names, g.columns)],
                       fontsize=6.5)
    ax.set_yticks(range(len(TASK_ORDER)))
    ax.set_yticklabels([rt.plot_name("simple", t) for t in TASK_ORDER], fontsize=6.8)
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    edges = np.cumsum([len(v) for v in GROUPS.values()])[:-1] - 0.5
    for e in edges:
        ax.axhline(e, color=rt.SURFACE, lw=2.5)
    n_task = sum(1 for _, k, _ in PER_TASK_SETS if k not in ("fv30", "fvuniv20", "rank_DBT", "mean_BT"))
    ax.axvline(n_task - 0.5, color=rt.SURFACE, lw=3)
    ax.xaxis.tick_top()
    ax.text(n_task / 2 - 0.5, -2.0, "one set per task", ha="center", fontsize=7, color=rt.INK_2)
    ax.text(n_task + 1.5, -2.0, "one set for all tasks", ha="center", fontsize=7, color=rt.INK_2)
    fig.tight_layout()
    fig.savefig(rt.FIGURES / "synth_tasks.pdf")
    preview(fig, "synth_tasks")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    c, cf, per = build()
    base = baselines()
    per.to_csv(OUT / "per_task.csv", index=False)
    s = summary(per)
    s.to_csv(OUT / "sets.csv", index=False)
    combos = all_combos(per)
    tex_main(s)
    tex_per_task(per, base, "simple", "synth_check1_tasks")
    tex_per_task(per, base, "simple-wrong", "synth_check2_tasks")
    tex_groups(per)
    tex_readouts()
    figure_size(s, combos)
    figure_tasks(per)
    pd.set_option("display.width", 200)
    print(s.drop(columns=["folder"]).round(3).to_string())
    print("unpatched", base.groupby("family").accuracy.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
