"""Synthesis of the simple-task head-selection experiments (paper/head-sets-simple.tex).

Puts every held-out head set (E4, E6, E7, E8, E10, E11) on one footing: J-head, rule style, E4's
item split and folds, rule means of the choosing half, accuracy at the **cross-fitted alpha over
alpha in {1, 2}** (fold 1 at the better of the two on fold 2 and the reverse; ties to alpha 1),
mean of the two folds. E8, E10 and E11 were run at alpha 1 and 2 only; E4, E6 and E7 at ten values
that include both, so their numbers here are lower than in their own reports where those picked
alpha >= 3. Random sets are scored at the alpha of the set they are matched to (possibly in another
experiment's folder: E11's per-task draws take E8's set's alpha, its shared DBT draws E10's). E4's
random sets: check 1 from E4, check 2 from E11 (E4 ran them there only at their ten-alpha
cross-fitted alpha). Davidson et al.'s intervention (sum of the heads' means added after layer 11):
E11's held-out runs, FV-30 at their alpha = 1, FV-30 without its generic heads cross-fitted.
Nothing is re-run here; every number is read from the saved accuracy files.

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

# (source folder, set key, label, approach, scope, in the main table). Approach = how heads are
# scored; scope = one set per task or one set for all tasks.
#
# Naming scheme (labels only; the set keys and result folders keep their experiment names):
#   family name  = how heads are chosen: "Yang terms" / "Task+common" (word lists decoded under
#                  J-head), filter letters D B T (D decodes the task's most active words, B does not
#                  look mainly at BOS, T attends to the instruction's task words), "Causal" (Davidson
#                  et al.'s causal score);
#   "-N"         = the top N heads of a ranking; no suffix = every head passing a threshold;
#   "Shared"     = one set for all 14 tasks (otherwise one set per task);
#   "FV"         = Davidson et al.'s intervention (means summed, added after layer 11); every other
#                  set is patched head by head;
#   "- generic"  = without the generic heads;
#   "Random, X layers" = X's heads per layer drawn outside X; "Random from D, X size" = X's size
#                  drawn from D.
LEXICAL, CAUSAL, ATTENTION, RANDOM = "lexical", "causal", "attention", "random"
DAV = "Davidson FV-30 ($\\alpha{=}1$)"
DAV_NOGEN = "Davidson FV-30 $-$ generic"
CC_DIFF = "Task+common, instr.\\ diff."
SETS = [
    ("heldout", "e1", "Yang terms", LEXICAL, "task", True),
    ("heldout", "cc20", "Task+common-20", LEXICAL, "task", True),
    ("heldout", "cc40", "Task+common-40", LEXICAL, "task", True),
    ("heldout", "cc", "Task+common", LEXICAL, "task", True),
    ("diff_heads", "dcc", CC_DIFF, LEXICAL, "task", True),
    ("heldout", "fv20", "Causal-20", CAUSAL, "task", True),
    ("heldout", "fv40", "Causal-40", CAUSAL, "task", True),
    ("heldout", "fvNt", "Causal-matched", CAUSAL, "task", True),
    ("e7_attfilter", "raw", "D$_{10}$", ATTENTION, "task", True),  # = E8's first-run D (same heads)
    ("e7_attfilter", "raw_att", "DB$_{10}$", ATTENTION, "task", True),  # = E8's first-run DB
    ("e8_filters", "D", "D", ATTENTION, "task", True),
    ("e8_filters", "DB", "DB", ATTENTION, "task", True),
    ("e8_filters", "DBT", "DBT", ATTENTION, "task", True),
    ("e8_filters", "BT", "BT", ATTENTION, "task", True),
    ("e8_filters", "T", "T", ATTENTION, "task", True),
    ("heldout", "random20", "Random, Task+common-20 layers", RANDOM, "task", True),
    ("heldout", "random40", "Random, Task+common-40 layers", RANDOM, "task", True),
    ("heldout", "randomNt", "Random, Task+common layers", RANDOM, "task", True),
    ("e7_attfilter", "random_drop_raw", "Random from D$_{10}$, DB$_{10}$ size", RANDOM, "task", True),
    ("e11_complete", "random_drop_DB", "Random from D, DB size", RANDOM, "task", True),
    ("e11_complete", "random_layer_DB", "Random, DB layers", RANDOM, "task", True),
    ("e11_complete", "random_drop_DBT", "Random from D, DBT size", RANDOM, "task", True),
    ("e11_complete", "random_layer_DBT", "Random, DBT layers", RANDOM, "task", True),
    ("e11_complete", "random_layer_T", "Random, T layers", RANDOM, "task", True),
    ("fvres", "fv30", DAV, CAUSAL, "shared", True),
    ("fvres", "fv30_nogen", DAV_NOGEN, CAUSAL, "shared", True),
    ("e11_complete", "fvuniv5", "Shared causal-5", CAUSAL, "shared", False),
    ("e11_complete", "fvuniv10", "Shared causal-10", CAUSAL, "shared", False),
    ("e11_complete", "fvuniv20", "Shared causal-20", CAUSAL, "shared", True),
    ("e11_complete", "fvuniv30", "Shared causal-30", CAUSAL, "shared", False),
    ("e11_complete", "fvuniv40", "Shared causal-40", CAUSAL, "shared", True),
    ("e11_complete", "fvuniv20_nogen", "Shared causal-20 $-$ generic", CAUSAL, "shared", True),
    ("e11_complete", "e10dbt5", "Shared DBT-5", ATTENTION, "shared", False),
    ("e11_complete", "e10dbt10", "Shared DBT-10", ATTENTION, "shared", False),
    ("e11_complete", "e10dbt20", "Shared DBT-20", ATTENTION, "shared", False),
    ("e10_shared", "rank_DBT", "Shared DBT-30", ATTENTION, "shared", True),
    ("e11_complete", "e10dbt40", "Shared DBT-40", ATTENTION, "shared", False),
    ("e10_shared", "mean_BT", "Shared BT-30", ATTENTION, "shared", True),
    ("e10_shared", "rank_DB", "Shared DB-30", ATTENTION, "shared", True),
    ("e11_complete", "randomuniv20", "Random shared, causal-20 layers", RANDOM, "shared", True),
    ("e11_complete", "randomdbt30", "Random shared, DBT-30 layers", RANDOM, "shared", True),
]
SHARED_SIZES = {  # ranking -> [(N, label)]
    "Shared causal": [(n, f"Shared causal-{n}") for n in (5, 10, 20, 30, 40)],
    "Shared DBT": [(n, f"Shared DBT-{n}") for n in (5, 10, 20, 30, 40)],
}
ONE_FOLD = {("heldout", s) for s in ("fvuniv20", "fvuniv40", "ad20", "ad40", "ccclean")}
RANDOM_BASE = {  # (folder, random set) -> (folder, set) whose cross-fitted alpha it takes
    ("heldout", "random20"): ("heldout", "cc20"), ("heldout", "random40"): ("heldout", "cc40"),
    ("heldout", "randomNt"): ("heldout", "cc"),
    **{("e7_attfilter", f"random_{k}_{w}"): ("e7_attfilter", f"{w}_att") for k in ("drop", "layer") for w in ("raw", "clean")},
    ("e11_complete", "randomuniv20"): ("e11_complete", "fvuniv20"),
    ("e11_complete", "randomdbt30"): ("e10_shared", "rank_DBT"),
    **{("e11_complete", f"random_{k}_{s}"): ("e8_filters", s) for k in ("drop", "layer") for s in ("DB", "DBT", "T")},
}
FOLDERS = ("heldout", "diff_heads", "e7_attfilter", "e8_filters", "e10_shared", "e11_complete")
COLOUR = {LEXICAL: rt.BLUE, CAUSAL: rt.ORANGE, ATTENTION: rt.AQUA, RANDOM: rt.GREY}
COLUMNS = ["folder", "fold", "family", "task", "set", "draw", "source", "alpha", "accuracy", "n_heads"]


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
        d = d.assign(folder=folder)
        if folder == "heldout":  # E4's check-2 draws ran only at their ten-alpha alpha: E11's instead
            d = d[~((d.family == "simple-wrong") & d.set.str.startswith("random"))]
        if folder == "e11_complete":  # E4's draws re-run in check 2 belong with E4's sets
            e4 = d.set.isin(["random20", "random40", "randomNt"])
            assert (d[e4].family == "simple-wrong").all()
            d.loc[e4, "folder"] = "heldout"
        out.append(d[COLUMNS])
    fv = pd.read_csv(RESULTS / "e11_complete" / "fv_residual_alpha.csv")
    fv = fv[(fv.set != "fv30") | (fv.alpha == 1)]  # Davidson et al.'s FV at their alpha only
    out.append(fv.assign(folder="fvres", draw=np.nan)[COLUMNS])
    return pd.concat(out, ignore_index=True)


def crossfit(c: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of every (folder, fold, family, task, set[, draw], source) at its cross-fitted
    alpha (random sets: that of their base set); one-fold sets at their best alpha in fold 1."""
    t = c[(c.source == "task") & c.draw.isna()]
    t = t.assign(key=list(zip(-t.accuracy, t.alpha.abs())))
    best = t.sort_values("key").drop_duplicates(["folder", "fold", "family", "task", "set"])
    best = best[["folder", "fold", "family", "task", "set", "alpha"]]
    one = pd.Series([(f, s) in ONE_FOLD for f, s in zip(best.folder, best.set)], index=best.index)
    chosen = pd.concat([best[~one].assign(fold=lambda x: 3 - x.fold), best[one]])  # the other fold's alpha
    chosen = chosen.rename(columns={"folder": "bfolder", "set": "bset"})
    keys = c[["folder", "fold", "family", "task", "set"]].drop_duplicates()
    base = [RANDOM_BASE.get((f, s), (f, s)) for f, s in zip(keys.folder, keys.set)]
    keys = keys.assign(bfolder=[b[0] for b in base], bset=[b[1] for b in base])
    keys = keys.merge(chosen, on=["bfolder", "fold", "family", "task", "bset"])
    return keys.drop(columns=["bfolder", "bset"]).merge(c, on=["folder", "fold", "family", "task", "set", "alpha"])


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
    cf = crossfit(c)
    # per task: mean over draws, then over folds
    per = (cf.groupby(["folder", "fold", "family", "task", "set", "source"], as_index=False)
           .agg(accuracy=("accuracy", "mean"), n_heads=("n_heads", "mean"), alpha=("alpha", "first")))
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
    for folder, key, label, approach, scope, main in SETS:
        p = per[(per.folder == folder) & (per.set == key)]
        if p.empty:
            raise SystemExit(f"missing {folder}/{key}")
        heads = p[(p.family == "simple") & (p.source == "task")].set_index("task").n_heads
        r = dict(folder=folder, set=key, label=label, approach=approach, scope=scope, main=main,
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
             "attention": "Filters: decoding (D), not BOS (B), task-word attention (T)", "random": "Random heads"}
    lines = []
    s = s[s.main]
    for scope, title in (("task", "One set per task"), ("shared", "One set for all 14 tasks")):
        lines.append(r"\multicolumn{6}{@{}l}{\textit{" + title + r"}} \\")
        for approach in (LEXICAL, CAUSAL, ATTENTION, RANDOM):
            g = s[(s.scope == scope) & (s.approach == approach)]
            if g.empty:
                continue
            lines.append(r"\quad " + heads[approach] + r" & & & & & \\")
            for r in g.itertuples():
                size = (f"{r.heads_median:.0f}" if r.heads_min == r.heads_max
                        else f"{r.heads_median:.0f} ({r.heads_min:.0f}--{r.heads_max:.0f})")
                assert r.folds == 2, r.label
                lines.append(r"\qquad " + r.label + f" & {size} & " + " & ".join(
                    rt.num(getattr(r, k)) for k in ("check1_task", "check1_control", "check2_task", "check2_control"))
                    + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    header = [r"Head set & Heads & \multicolumn{2}{c}{Check 1} & \multicolumn{2}{c}{Check 2} \\",
              r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
              r" & median (range) & task & control & task & control \\"]
    text = "\n".join([r"\begin{tabular}{@{}lccccc@{}}", r"\toprule", *header, r"\midrule", *lines,
                      r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / "synth_main.tex").write_text("% Generated by scripts/synthesis_report.py.\n" + text + "\n")


def tex_shared_sizes(s: pd.DataFrame) -> None:
    """The two shared rankings at 5 to 40 heads: task and control mean in check 1, task mean in check 2."""
    idx = s.set_index("label")
    lines = []
    for n in (5, 10, 20, 30, 40):
        cells = []
        for ranking, labels in SHARED_SIZES.items():
            r = idx.loc[dict(labels)[n]]
            cells += [rt.num(r.check1_task), rt.num(r.check1_control), rt.num(r.check2_task)]
        lines.append(f"{n} & " + " & ".join(cells) + r" \\")
    header = [r" & \multicolumn{3}{c}{Shared causal-$N$} & \multicolumn{3}{c}{Shared DBT-$N$} \\",
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
              r"$N$ & check 1 & control & check 2 & check 1 & control & check 2 \\"]
    text = "\n".join([r"\begin{tabular}{@{}ccccccc@{}}", r"\toprule", *header, r"\midrule", *lines,
                      r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / "synth_shared_sizes.tex").write_text("% Generated by scripts/synthesis_report.py.\n" + text + "\n")


PER_TASK_SETS = [  # (folder, key, short column label; "\n" = line break)
    ("heldout", "e1", "Yang\nterms"), ("heldout", "cc", "Task+\ncommon"),
    ("heldout", "fv20", "Causal-20"), ("heldout", "fvNt", "Causal-\nmatched"),
    ("e7_attfilter", "raw_att", "DB$_{10}$"), ("e8_filters", "DB", "DB"), ("e8_filters", "DBT", "DBT"),
    ("e8_filters", "T", "T"),
    ("fvres", "fv30", "Davidson\nFV-30"), ("fvres", "fv30_nogen", "FV-30\n$-$generic"),
    ("e11_complete", "fvuniv40", "Shared\ncausal-40"), ("e10_shared", "rank_DBT", "Shared\nDBT-30"),
]


def tex_head(label: str) -> str:
    """A column label for a LaTeX table, its line breaks as a stacked cell."""
    if "\n" not in label:
        return label
    return r"\begin{tabular}[b]{@{}c@{}}" + label.replace("\n", r"\\") + r"\end{tabular}"


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
    header = [r"Task & Items & Unp. & " + " & ".join(map(tex_head, cols)) + r" \\"]
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
    header = [r"Task group & " + " & ".join(map(tex_head, g.columns)) + r" \\"]
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


PER_TASK_PLOT = {  # label -> (dx, dy, ha) of its direct label, None = no label
    "Yang terms": (0, -9, "center"), "Task+common-20": (7, 17, "center"), "Task+common-40": None,
    "Task+common": (0, -9, "center"), "DB$_{10}$": (6, 0, "left"), CC_DIFF: (0, 12, "center"),
    "Causal-20": (-6, 0, "right"), "Causal-40": None, "Causal-matched": (6, 2, "left"),
    "D": (0, 8, "center"), "DB": (4, 7, "left"), "DBT": (-4, 7, "right"), "BT": (6, -4, "left"),
    "T": (6, 0, "left"), "Random, Task+common-20 layers": None, "Random, Task+common-40 layers": None,
    "Random, Task+common layers": (6, 0, "left"),
    "Random from D$_{10}$, DB$_{10}$ size": None, "Random from D, DB size": (6, 0, "left"), "Random, DB layers": None,
    "Random from D, DBT size": None, "Random, DBT layers": None, "Random, T layers": (6, 0, "left")}
SHARED_PLOT = {
    DAV: (6, -2, "left"), DAV_NOGEN: (6, 2, "left"), "Shared causal-20 $-$ generic": (-6, 3, "right"),
    "Shared BT-30": None, "Shared DB-30": None,
    "Shared causal-5": (0, 9, "center"), "Shared DBT-5": (0, -9, "center"),
    "Random shared, causal-20 layers": None, "Random shared, DBT-30 layers": (6, 0, "left")}
NAMES = {DAV: "Davidson FV-30 ($\\alpha$=1)", CC_DIFF: "Task+common,\ninstr. diff.",
         "Causal-20": "Causal-$N$", "Task+common-20": "Task+common-$N$",
         "Shared causal-5": "Shared causal-$N$", "Shared DBT-5": "Shared DBT-$N$",
         "Random from D, DB size": "random from D", "Random, T layers": "random, T layers",
         "Random shared, DBT-30 layers": "random shared", "Random, Task+common layers": "random"}
LINES = {"task": [("Task+common-20", "Task+common-40", "Task+common"), ("Causal-20", "Causal-40", "Causal-matched"),
                  ("Random, Task+common-20 layers", "Random, Task+common-40 layers", "Random, Task+common layers")],
         "shared": [tuple(lab for _, lab in labs) for labs in SHARED_SIZES.values()]}


def point(ax, r, label, offset) -> None:
    col = COLOUR[r.approach]
    ax.scatter(r.heads_median, r.check1_task, s=30, marker="s" if label == "Yang terms" else "o",
               facecolor=col if r.scope == "task" else rt.SURFACE, edgecolor=col if r.scope == "shared" else rt.SURFACE,
               lw=1.5 if r.scope == "shared" else 0.8, zorder=4)
    if offset:
        dx, dy, ha = offset
        ax.annotate(NAMES.get(label, label), (r.heads_median, r.check1_task), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=6.2, color=rt.INK_2, zorder=5)


def figure_size(s: pd.DataFrame, combos: pd.DataFrame) -> None:
    """Check 1, mean accuracy over the 14 tasks against the number of heads (median over tasks),
    log x: (a) one set per task, (b) one set for all tasks. Colour = how heads are scored; filled =
    per task, open = shared. Faint dots: the other filter combinations, per task (a) and shared (b);
    faint line in (b): the per-task Causal-N sets of (a), for reference."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
    idx = s.set_index("label")
    for ax, scope, plot, folder, title in (
            (axes[0], "task", PER_TASK_PLOT, "e8_filters", "(a) One set per task"),
            (axes[1], "shared", SHARED_PLOT, "e10_shared", "(b) One set for all 14 tasks")):
        g = combos[combos.folder == folder]
        ax.scatter(g.heads, g.acc, s=8, facecolor=rt.AQUA if scope == "task" else "none", edgecolor=rt.AQUA,
                   alpha=0.25, lw=0.6, zorder=1)
        if scope == "shared":
            r = idx.loc[["Causal-20", "Causal-40", "Causal-matched"]]
            ax.plot(r.heads_median, r.check1_task, color=rt.ORANGE, lw=0.9, alpha=0.35, ls=(0, (3, 2)), zorder=1)
            ax.annotate("per-task Causal-$N$ (a)", (r.heads_median.iloc[-1], r.check1_task.iloc[-1]), xytext=(4, -8),
                        textcoords="offset points", fontsize=6, color=rt.INK_2, alpha=0.8)
        for chain in LINES[scope]:
            r = idx.loc[list(chain)]
            ax.plot(r.heads_median, r.check1_task, color=COLOUR[r.approach.iloc[0]], lw=1.0, alpha=0.7, zorder=2)
        labels = list(plot) + [lab for chain in LINES[scope] for lab in chain if lab not in plot]
        for label in labels:
            point(ax, idx.loc[label], label, plot.get(label))
        ax.set_xscale("log")
        ticks = [5, 10, 20, 30, 40, 60, 100, 150, 250] if scope == "task" else [5, 10, 20, 30, 40, 60, 100]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlim((10, 340) if scope == "task" else (3.5, 150))
        ax.set_ylim(-0.02, 0.7)
        ax.set_xlabel("heads in the set (median over tasks)")
        ax.set_title(title, loc="left")
        rt.style_axes(ax)
    axes[0].set_ylabel("check 1 accuracy, mean of 14 tasks")
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=5, color=COLOUR[a]) for a in (LEXICAL, CAUSAL, ATTENTION, RANDOM)]
    labels = ["decoded word lists (lens)", "causal score (Davidson et al.)", "filters D, B, T", "random heads"]
    handles += [plt.Line2D([], [], ls="none", marker="o", ms=5, mfc=rt.SURFACE, mec=rt.INK_2, mew=1.4),
                plt.Line2D([], [], ls="none", marker="o", ms=3.5, color=rt.AQUA, alpha=0.3)]
    labels += ["open: one set for all tasks", "faint: other filter combinations"]
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=6.8)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(rt.FIGURES / "synth_size.pdf")
    preview(fig, "synth_size")
    plt.close(fig)


CHECK_LABELS = {"Causal-matched": (4, -7, "left"), DAV_NOGEN: (2, 10, "right"),
                "D": (5, -3, "left"), "Yang terms": (5, -3, "left"), "Random, Task+common layers": (5, 0, "left"),
                "Random, T layers": (5, 0, "left")}


def figure_checks(s: pd.DataFrame) -> None:
    """Check 2 against check 1, mean over the 14 tasks, every set of the main table run in both."""
    r = s[s.main & s.check2_task.notna()]
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.plot([0, 0.7], [0, 0.7], color=rt.AXIS_LINE, lw=0.8, zorder=1)
    for x in r.itertuples():
        col = COLOUR[x.approach]
        ax.scatter(x.check1_task, x.check2_task, s=22, marker="s" if x.label == "Yang terms" else "o",
                   facecolor=col if x.scope == "task" else rt.SURFACE, edgecolor=col if x.scope == "shared" else rt.SURFACE,
                   lw=1.3 if x.scope == "shared" else 0.6, zorder=3)
        if x.label in CHECK_LABELS:
            dx, dy, ha = CHECK_LABELS[x.label]
            ax.annotate(NAMES.get(x.label, x.label), (x.check1_task, x.check2_task), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va="center", fontsize=6, color=rt.INK_2)
    ax.set_xlim(-0.02, 0.7)
    ax.set_ylim(-0.02, 0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("check 1 accuracy (zero-shot prompts)")
    ax.set_ylabel("check 2 accuracy (wrong 1-shot)")
    rt.style_axes(ax)
    fig.tight_layout()
    fig.savefig(rt.FIGURES / "synth_checks.pdf")
    preview(fig, "synth_checks")
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
    names = [c.replace("$N_t$", "Nt").replace(chr(92) + " ", " ").replace("$-$", "\u2212") for c in g.columns]
    ax.set_xticklabels([f"{n}\n{sizes[c]:.0f}" for n, c in zip(names, g.columns)],
                       fontsize=6.1)
    ax.set_yticks(range(len(TASK_ORDER)))
    ax.set_yticklabels([rt.plot_name("simple", t) for t in TASK_ORDER], fontsize=6.8)
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    edges = np.cumsum([len(v) for v in GROUPS.values()])[:-1] - 0.5
    for e in edges:
        ax.axhline(e, color=rt.SURFACE, lw=2.5)
    n_task = sum(1 for _, k, _ in PER_TASK_SETS if k not in ("fv30", "fv30_nogen", "fvuniv40", "rank_DBT"))
    ax.axvline(n_task - 0.5, color=rt.SURFACE, lw=3)
    ax.xaxis.tick_top()
    ax.text(n_task / 2 - 0.5, -2.55, "one set per task", ha="center", fontsize=7, color=rt.INK_2)
    ax.text(n_task + 1.5, -2.55, "one set for all tasks", ha="center", fontsize=7, color=rt.INK_2)
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
    tex_shared_sizes(s)
    tex_per_task(per, base, "simple", "synth_check1_tasks")
    tex_per_task(per, base, "simple-wrong", "synth_check2_tasks")
    tex_groups(per)
    tex_readouts()
    figure_size(s, combos)
    figure_checks(s)
    figure_tasks(per)
    pd.set_option("display.width", 200)
    print(s.drop(columns=["folder"]).round(3).to_string())
    print("unpatched", base.groupby("family").accuracy.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
