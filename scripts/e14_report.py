"""E14 report: E13's main sets re-run with all 35 rule templates (scripts/e14_sets.py, e14_run.sh), check 1.

Accuracies only (no intervals), at the **cross-fitted alpha over {1, 2}** as in E13: fold 1 at the alpha
that was better on fold 2 for the same task and set, and the reverse (ties to alpha 1), mean of the two
folds. A set's task accuracy at an alpha is the mean over its task means, one per rule template with a
correct prompt on the choosing half (at most 35). Random draws take the alpha of the a35 DBT-80 set of
the same scope and are averaged over the five draws. The breakdown by template group (Yang's five; the
new cloze, question-answer and label templates, named or not) is read at the same alpha.

Built-in check: the E13 sets (``y5_*``) with Yang's five means only, cross-fitted on those, must give
E13's numbers (``results/e13_capped/per_task.csv``): same prompts, means and alphas.

Writes ``results/e14_wordings/{accuracy_alpha,per_task,summary,groups,e0}.csv``, the tables
``paper/tables/e14_*.tex`` and the figure ``paper/figures/e14_groups.pdf``.

    .venv/bin/python scripts/e14_report.py
"""

from __future__ import annotations

import glob
import json
import re
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
from lth import DATA, RESULTS, simple  # noqa: E402

OUT = RESULTS / "e14_wordings"
E13 = RESULTS / "e13_capped"
SIZES = (40, 80)
GROUPS = {
    "Word relations": ("antonym", "synonym"),
    "Morphology": ("singular-plural", "present-past"),
    "Facts": ("country-capital", "country-currency", "landmark-country", "park-country", "product-company"),
    "Sequences": ("next_item", "prev_item"),
    "Translations": ("english-french", "english-german", "english-spanish"),
}
TASK_ORDER = [t for ts in GROUPS.values() for t in ts]
CONDITIONS = {"DBT": "DBT", "causal": "Causal", "causal_nogen": "Causal $-$ generic"}
SCOPES = {"task": "One set per task", "shared": "One set for all 14 tasks"}
KINDS = {"y5": "chosen on Yang's 5", "a35": "chosen on all 35"}
# Template groups of the means, in table order.
WGROUPS = ["yang"] + list(simple.E4B_CELLS)
WLABEL = {"yang": "Yang's 5", "cloze-named": "cloze, named", "cloze-unnamed": "cloze, not named",
          "qa-named": "Q--A, named", "qa-unnamed": "Q--A, not named", "label-named": "label, named",
          "label-unnamed": "label, not named"}


def parse(name: str) -> tuple[str, str, str, int]:
    """set name -> (kind, scope, condition, N); random draws: kind 'random'."""
    if name.startswith("random"):
        return "random", ("shared" if "shared" in name else "task"), "random", int(re.search(r"dbt(\d+)_\d$", name).group(1))
    kind, core = name.split("_", 1)
    scope = "shared" if core.startswith("shared_") else "task"
    m = re.fullmatch(r"([A-Za-z]+?)(\d+)(_nogen)?", core.removeprefix("shared_"))
    return kind, scope, m.group(1) + (m.group(3) or ""), int(m.group(2))


def load_rows(root: Path, kinds=None) -> pd.DataFrame:
    rows = []
    for fold in (1, 2):
        for path in glob.glob(str(root / f"fold{fold}" / "patching_simple.part-*.csv")):
            rows.append(pd.read_csv(path).assign(fold=fold))
    d = pd.concat(rows, ignore_index=True)
    return d[d.family == "simple"]


def curves(d: pd.DataFrame, wordings=None) -> pd.DataFrame:
    """fold, task, set, source, alpha -> accuracy (task: mean over the means of ``wordings``, default all)."""
    p = d[d.source != "none"]
    if wordings is not None:
        p = p[(p.source == "control") | p.mean_wording.isin(wordings)]
    c = (p.groupby(["fold", "task", "heads", "source", "alpha"], as_index=False)
         .agg(accuracy=("accuracy", "mean"), means=("accuracy", "size"), n_heads=("n_heads", "first")))
    return c.rename(columns={"heads": "set"})


def crossfit(c: pd.DataFrame, base_of) -> pd.DataFrame:
    """Per fold, task, set, source: the accuracy at the alpha chosen on the other fold (task source,
    ties to the smaller alpha); a random draw takes the alpha of ``base_of(set)``."""
    t = c[(c.source == "task") & ~c.set.str.startswith("random")]
    best = (t.assign(key=list(zip(-t.accuracy, t.alpha.abs()))).sort_values("key")
            .drop_duplicates(["fold", "task", "set"])[["fold", "task", "set", "alpha"]])
    best = best.assign(fold=3 - best.fold).rename(columns={"set": "base"})
    c = c.assign(base=[base_of(s) for s in c.set])
    return c.merge(best, on=["fold", "task", "base", "alpha"])


def base_of(s: str) -> str:
    if s.startswith("random_shared"):
        return "a35_shared_DBT80"
    if s.startswith("random"):
        return "a35_DBT80"
    return s


def e13_check(d: pd.DataFrame) -> float:
    """Largest difference between E13's cross-fitted per-task accuracies and the y5 sets' here, with Yang's
    five means only."""
    y = d[d.heads.str.startswith("y5_")].assign(heads=lambda x: x.heads.str.removeprefix("y5_"))
    cf = crossfit(curves(y, simple.WORDINGS["rule"]), lambda s: s)
    mine = cf.groupby(["task", "set", "source"]).accuracy.mean()
    ref = pd.read_csv(E13 / "per_task.csv").set_index(["task", "set", "source"]).accuracy
    joined = pd.concat([mine.rename("e14"), ref.rename("e13")], axis=1, join="inner")
    assert len(joined) == len(mine), "an E13 set is missing from E13's per_task.csv"
    return float((joined.e14 - joined.e13).abs().max())


def main() -> None:
    d = load_rows(OUT)
    base = d[d.source == "none"].drop_duplicates(["fold", "task"])[["fold", "task", "accuracy", "n_prompts"]]
    diff = e13_check(d)
    print(f"check: y5 sets with Yang's five means vs E13, largest per-task difference {diff:.2e}")

    c = curves(d)
    c.to_csv(OUT / "accuracy_alpha.csv", index=False)
    cf = crossfit(c, base_of)
    meta = pd.DataFrame([parse(s) for s in cf.set], columns=["kind", "scope", "condition", "n"], index=cf.index)
    cf = pd.concat([cf, meta], axis=1)
    cf.loc[cf.kind == "random", "set"] = cf.loc[cf.kind == "random", "set"].str.replace(r"_\d$", "", regex=True)
    cf = (cf.groupby(["fold", "task", "set", "kind", "scope", "condition", "n", "source"], as_index=False)
          .agg(accuracy=("accuracy", "mean"), n_heads=("n_heads", "mean"), means=("means", "mean"),
               alpha=("alpha", "first")))
    per = (cf.groupby(["task", "set", "kind", "scope", "condition", "n", "source"], as_index=False)
           .agg(accuracy=("accuracy", "mean"), n_heads=("n_heads", "mean"), means=("means", "mean"),
                folds=("fold", "nunique"), alpha2=("alpha", lambda a: float(np.mean(a == 2)))))
    per.to_csv(OUT / "per_task.csv", index=False)
    task = per[per.source == "task"]
    ctrl = per[per.source == "control"].set_index(["set"]).groupby("set").accuracy.mean()
    s = (task.groupby(["set", "kind", "scope", "condition", "n"], as_index=False)
         .agg(check1=("accuracy", "mean"), tasks=("task", "nunique"), means_min=("means", "min"),
              alpha2_share=("alpha2", "mean")))
    s["control"] = s.set.map(ctrl)
    s13 = pd.read_csv(E13 / "summary.csv").set_index("set").check1_task
    s["e13"] = [s13.get(x.removeprefix("y5_"), np.nan) if k == "y5" else np.nan for x, k in zip(s.set, s.kind)]
    s.to_csv(OUT / "summary.csv", index=False)

    # Breakdown by template group, at the cross-fitted alpha of the 35-mean accuracy.
    p = d[(d.source == "task")].assign(group=lambda x: x.mean_wording.map(simple.wording_group))
    g = (p.groupby(["fold", "task", "heads", "alpha", "group"], as_index=False).accuracy.mean()
         .rename(columns={"heads": "set"}))
    chosen = cf[cf.source == "task"][["fold", "task", "set", "alpha"]]
    # a random set stands for its five draws: expand it back to them
    exp = []
    for _, r in chosen.iterrows():
        if r.set.startswith("random"):
            exp += [dict(fold=r.fold, task=r.task, set=r.set, alpha=r.alpha, raw=f"{r.set}_{i}") for i in range(5)]
        else:
            exp.append(dict(fold=r.fold, task=r.task, set=r.set, alpha=r.alpha, raw=r.set))
    exp = pd.DataFrame(exp)
    gg = exp.merge(g.rename(columns={"set": "raw"}), on=["fold", "task", "raw", "alpha"])
    gg = gg.groupby(["fold", "task", "set", "group"], as_index=False).accuracy.mean()
    groups = gg.groupby(["task", "set", "group"], as_index=False).accuracy.mean()
    groups.to_csv(OUT / "groups.csv", index=False)
    gsum = groups.groupby(["set", "group"]).accuracy.mean().unstack()[WGROUPS]

    e0 = e0_table()
    tex_main(s)
    tex_groups(s, gsum)
    tex_tasks(per, base)
    tex_e0(e0)
    tex_sets()
    figure_groups(s, gsum)
    pd.set_option("display.width", 220)
    print(s.pivot_table(index=["scope", "condition"], columns=["kind", "n"], values="check1").round(3).to_string())
    print(gsum.round(3).to_string())
    print("unpatched", round(base.accuracy.mean(), 3), "| controls max", round(s.control.max(), 3))


# --------------------------------------------------------------------------- E0 on the templates

def e0_table() -> pd.DataFrame:
    """Per task and template: E0 accuracy (flexible) on the E14 items, and correct prompts per half."""
    from e14_outputs import items_of

    from lth.store import load, readings_dir

    split = json.loads((RESULTS / "heldout" / "split.json").read_text())
    rows = []
    for t in simple.TASKS:
        items = set(items_of(t))
        for w in simple.ALL_RULE:
            rec = [r for r in load(readings_dir("simple", t), w)["records"] if int(r["item"]) in items]
            ok = [int(r["item"]) for r in rec if r["correct"]]
            rows.append(dict(task=t, wording=w, group=simple.wording_group(w), n=len(rec),
                             accuracy=len(ok) / len(rec),
                             correct_A=min(100, len(set(ok) & set(split[t]["A"]))),
                             correct_B=min(100, len(set(ok) & set(split[t]["B"])))))
    e0 = pd.DataFrame(rows)
    e0.to_csv(OUT / "e0.csv", index=False)
    return e0


# --------------------------------------------------------------------------- tables

def cell(v, bold=False) -> str:
    s = rt.num(v)
    return r"\textbf{" + s + "}" if bold else s


def tex_main(s: pd.DataFrame) -> None:
    """Rows: scope x condition (+ random, control); columns: N x (E13, y5 + 35 means, a35 + 35 means)."""
    idx = s.set_index(["kind", "scope", "condition", "n"])
    lines = []
    for scope, title in SCOPES.items():
        if lines:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{@{}l}{\textit{" + title + r"}} \\")
        for cond, label in list(CONDITIONS.items()) + [("random", "Random (DBT-80 layers)")]:
            cells = []
            for n in SIZES:
                if cond == "random":
                    v = idx.check1.get(("random", scope, "random", n), np.nan)
                    cells += ["--", "--", rt.num(v)]
                    continue
                e13 = idx.e13.get(("y5", scope, cond, n), np.nan)
                y5 = idx.check1.get(("y5", scope, cond, n), np.nan)
                a35 = idx.check1.get(("a35", scope, cond, n), np.nan)
                top = np.nanmax([e13, y5, a35])
                cells += [cell(v, v == top) for v in (e13, y5, a35)]
            lines.append(r"\quad " + label + " & " + " & ".join(cells) + r" \\")
    ctrl = s[s.kind != "random"].control
    lines += [r"\midrule", r"Control mean, largest over these sets & \multicolumn{6}{c}{"
              + rt.num(ctrl.max()) + r"} \\"]
    header = [r" & \multicolumn{3}{c}{40 heads} & \multicolumn{3}{c}{80 heads} \\",
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
              r"Heads chosen on & Yang's 5 & Yang's 5 & all 35 & Yang's 5 & Yang's 5 & all 35 \\",
              r"Means from & Yang's 5 & all 35 & all 35 & Yang's 5 & all 35 & all 35 \\"]
    text = "\n".join([r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule", *header, r"\midrule", *lines,
                      r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / "e14_main.tex").write_text("% Generated by scripts/e14_report.py.\n" + text + "\n")


def tex_groups(s: pd.DataFrame, gsum: pd.DataFrame) -> None:
    """Accuracy by template group of the means, for every non-random set."""
    lines = []
    for scope, title in SCOPES.items():
        if lines:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{" + str(len(WGROUPS) + 2) + r"}{@{}l}{\textit{" + title + r"}} \\")
        for n in SIZES:
            for cond, label in CONDITIONS.items():
                for kind in KINDS:
                    name = f"{kind}_{'shared_' if scope == 'shared' else ''}{cond.replace('_nogen', '')}{n}" + (
                        "_nogen" if cond.endswith("_nogen") else "")
                    if name not in gsum.index:
                        continue
                    vals = gsum.loc[name]
                    lines.append(rf"\quad {label}-{n}, {'Yang' if kind == 'y5' else 'all'} & "
                                 + rt.num(s.set_index("set").check1[name]) + " & "
                                 + " & ".join(rt.num(v) for v in vals) + r" \\")
    head = [r" & & \multicolumn{" + str(len(WGROUPS)) + r"}{c}{Means from} \\",
            r"\cmidrule(lr){3-" + str(len(WGROUPS) + 2) + "}",
            r"Set, heads chosen on & All 35 & " + " & ".join(
                r"\begin{tabular}[b]{@{}c@{}}" + WLABEL[g].replace(", ", r",\\") + r"\end{tabular}" for g in WGROUPS) + r" \\"]
    text = "\n".join([r"\begin{tabular}{@{}lc" + "c" * len(WGROUPS) + r"@{}}", r"\toprule", *head, r"\midrule",
                      *lines, r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / "e14_groups.tex").write_text("% Generated by scripts/e14_report.py.\n" + text + "\n")


def tex_tasks(per: pd.DataFrame, base: pd.DataFrame) -> None:
    """Per task, 40 and 80 heads: y5 and a35 of each condition and scope (35 means)."""
    cols = [(scope, cond, kind) for scope in SCOPES for cond in CONDITIONS for kind in KINDS]
    q = per[per.source == "task"].set_index(["task", "kind", "scope", "condition", "n"]).accuracy
    b = base.groupby("task").accuracy.mean()
    blocks = []
    for n in SIZES:
        lines = [r"\multicolumn{" + str(len(cols) + 2) + r"}{@{}l}{\textit{" + f"{n} heads" + r"}} \\"]
        for t in TASK_ORDER:
            vals = [q.get((t, k, sc, c, n), np.nan) for sc, c, k in cols]
            lines.append(rt.tex_name("simple", t) + f" & {rt.num(b[t])} & " + " & ".join(rt.num(v) for v in vals) + r" \\")
        means = [np.nanmean([q.get((t, k, sc, c, n), np.nan) for t in TASK_ORDER]) for sc, c, k in cols]
        lines.append(r"\textit{Mean} & " + rt.num(b.mean()) + " & " + " & ".join(rt.num(v) for v in means) + r" \\")
        blocks.append("\n".join(lines))
    lab = {"DBT": "DBT", "causal": "Causal", "causal_nogen": r"Causal\,$-$g."}
    head = [r" & & \multicolumn{6}{c}{One set per task} & \multicolumn{6}{c}{One set for all tasks} \\",
            r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}",
            r" & & " + " & ".join(r"\multicolumn{2}{c}{" + lab[c] + "}" for _ in SCOPES for c in CONDITIONS) + r" \\",
            r"Task & Unp. & " + " & ".join("Y5 & A35" for _ in range(6)) + r" \\"]
    text = "\n".join([r"\begin{tabular}{@{}lc" + "c" * len(cols) + r"@{}}", r"\toprule", *head, r"\midrule",
                      (r"\midrule" + "\n").join(blocks), r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / "e14_tasks.tex").write_text("% Generated by scripts/e14_report.py.\n" + text + "\n")


def tex_e0(e0: pd.DataFrame) -> None:
    """Per task: mean E0 accuracy by template group, and templates with fewer than 30 (or no) correct prompts on
    a half."""
    m = e0.groupby(["task", "group"]).accuracy.mean().unstack()[WGROUPS]
    fewest = e0[["correct_A", "correct_B"]].min(axis=1)
    short = e0.assign(s=fewest < 30).groupby("task").s.sum()
    zero = e0.assign(z=fewest == 0).groupby("task").z.sum()
    lines = []
    for t in TASK_ORDER:
        lines.append(rt.tex_name("simple", t) + " & " + " & ".join(rt.num(v) for v in m.loc[t])
                     + f" & {int(short[t])} & {int(zero[t])}" + r" \\")
    lines += [r"\midrule", r"\textit{Mean} & " + " & ".join(rt.num(v) for v in m.mean())
              + f" & {int(short.sum())} & {int(zero.sum())}" + r" \\"]
    head = [r" & \multicolumn{" + str(len(WGROUPS)) + r"}{c}{E0 accuracy (flexible), mean of the group's templates}"
            r" & \multicolumn{2}{c}{Templates} \\",
            r"\cmidrule(lr){2-" + str(len(WGROUPS) + 1) + r"}\cmidrule(lr){" + str(len(WGROUPS) + 2) + "-"
            + str(len(WGROUPS) + 3) + "}",
            r"Task & " + " & ".join(r"\begin{tabular}[b]{@{}c@{}}" + WLABEL[g].replace(", ", r",\\") + r"\end{tabular}"
                                    for g in WGROUPS) + r" & $<30$ & 0 \\"]
    text = "\n".join([r"\begin{tabular}{@{}l" + "c" * (len(WGROUPS) + 2) + r"@{}}", r"\toprule", *head, r"\midrule",
                      *lines, r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / "e14_e0.tex").write_text("% Generated by scripts/e14_report.py.\n" + text + "\n")


def tex_sets() -> None:
    """Overlap of each a35 set with its y5 twin (mean over tasks and folds), and the generic heads."""
    rows = pd.concat([pd.read_csv(OUT / f"fold{f}" / "sets.csv") for f in (1, 2)])
    rows["scope"] = np.where(rows.set.str.contains("shared"), "shared", "task")
    rows["core"] = rows.set.str.replace("a35_", "").str.replace("shared_", "")
    o = rows.groupby(["scope", "core"]).overlap_y5.mean()
    lines = []
    for scope, title in SCOPES.items():
        cells = []
        for n in SIZES:
            for cond in CONDITIONS:
                core = f"{cond.replace('_nogen', '')}{n}" + ("_nogen" if cond.endswith("_nogen") else "")
                cells.append(f"{o.get((scope, core), np.nan):.1f}")
        lines.append(title + " & " + " & ".join(cells) + r" \\")
    head = [r" & \multicolumn{3}{c}{40 heads} & \multicolumn{3}{c}{80 heads} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            " & " + " & ".join(CONDITIONS[c] for _ in SIZES for c in CONDITIONS) + r" \\"]
    text = "\n".join([r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule", *head, r"\midrule", *lines,
                      r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / "e14_overlap.tex").write_text("% Generated by scripts/e14_report.py.\n" + text + "\n")
    gen = []
    for f in (1, 2):
        g = json.loads((OUT / f"fold{f}" / "generic.json").read_text())
        gen.append(f"fold {f}: " + ", ".join(g["generic"]) + f" (on Yang's 5: {', '.join(g['e13_generic'])}; "
                   f"Davidson top 30 shared: {g['fv30_shared_with_e13']})")
    (rt.TABLES / "e14_generic.tex").write_text("% Generated by scripts/e14_report.py.\n" + "; ".join(gen) + "\n")


# --------------------------------------------------------------------------- figure

def figure_groups(s: pd.DataFrame, gsum: pd.DataFrame) -> None:
    """Accuracy by template group of the means, for the three conditions at 80 heads, both scopes:
    heads chosen on Yang's 5 (open markers) or on all 35 (filled)."""
    colour = {"DBT": rt.AQUA, "causal": rt.ORANGE, "causal_nogen": rt.BLUE}
    marker = {"DBT": "o", "causal": "s", "causal_nogen": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
    x = np.arange(len(WGROUPS))
    for ax, (scope, title) in zip(axes, SCOPES.items()):
        for j, cond in enumerate(CONDITIONS):
            for kind in KINDS:
                name = f"{kind}_{'shared_' if scope == 'shared' else ''}{cond.replace('_nogen', '')}80" + (
                    "_nogen" if cond.endswith("_nogen") else "")
                v = gsum.loc[name].to_numpy()
                off = (j - 1) * 0.22 + (0.07 if kind == "a35" else -0.07)
                ax.plot(x + off, v, ls="none", marker=marker[cond], ms=4.8, mew=1.1,
                        mfc=colour[cond] if kind == "a35" else rt.SURFACE, mec=colour[cond], zorder=3)
        ax.set_xticks(x, [WLABEL[g].replace("--", "–").replace(", ", "\n") for g in WGROUPS], fontsize=6.2)
        ax.set_ylim(-0.02, 0.85)
        ax.set_title(f"({'a' if scope == 'task' else 'b'}) {title}, 80 heads", loc="left")
        rt.style_axes(ax)
    axes[0].set_ylabel("check 1 accuracy, mean of 14 tasks")
    axes[0].text(-0.5, -0.24, "means from:", transform=axes[0].get_xaxis_transform(), fontsize=6.2, color=rt.INK_2,
                 ha="left")
    handles = [plt.Line2D([], [], ls="none", marker=marker[c], ms=4.8, mfc=colour[c], mec=colour[c]) for c in CONDITIONS]
    labels = [CONDITIONS[c].replace("$-$", "−") for c in CONDITIONS]
    handles += [plt.Line2D([], [], ls="none", marker="o", ms=4.8, mfc=rt.SURFACE, mec=rt.INK_2),
                plt.Line2D([], [], ls="none", marker="o", ms=4.8, mfc=rt.INK_2, mec=rt.INK_2)]
    labels += ["heads chosen on Yang's 5", "heads chosen on all 35"]
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=6.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    fig.savefig(rt.FIGURES / "e14_groups.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
