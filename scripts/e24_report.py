"""E24 report: are the heads shared between rule (instruction) and example prompts?

Accuracies as ``scripts/e13_report.py``: check 1, held-out items, per fold at the alpha in {1, 2} that
was better on the other fold for the same task, set and mean style (ties to 1), mean of the two
folds, mean of the 14 tasks. Rule means: mean over the five rule templates' means; example mean: the
5-shot mean. Random draws take the alpha chosen on the other fold from the mean of their five draws.
Rule-mean accuracies of sets E13/E22/E23 already ran are read from their ``per_task.csv``.

Every table and figure is divided by set size N (user, 2026-11-16): sets are compared only with
sets of the same size. A set whose heads number fewer than N on some task or fold (per-task Yang
terms at 60 and above) is shown as "--".

- Step D (``results/e24_cross_style/d``): Causal, Causal - generic, DBMN, DBTMS, DBTLS, DBTM and Yang
  terms, per task and shared, at every N, with the example mean; random draws outside the
  example-built DB set of the same scope and N with both means (40 and 100 from step B).
- Step E (``e``): the other good filter combinations (``scripts/e24_sets_e.py``: 12 of four or more
  letters among E23's 15 best of either scope, and DBS) likewise, with rule means where E22/E23 never
  ran them.
- Step B (``b``): the example-built D/B/DB/DBMS sets with both means; the overlap of rule- and
  example-built sets (``b/overlap.csv``).
- Step C (``c``): the rule-built D/B/DB/DBMS sets with the example mean (and DBMS100 with rule means).
- Step A (``a``, superseded by D): the best set of each rule at one size; kept for E13's random
  draws (layer-matched to DBT-100, drawn from whole layers where DBT is dense), reported in the
  text, and as a rerun check of D.

Writes ``results/e24_cross_style/{per_task,summary}.csv``, ``report.md``, the tables
``paper/tables/e24_{sizes_task,sizes_shared,types_task,types_shared,grid}.tex`` and the figures
``paper/figures/e24_{sizes,overlap,grid}.pdf``.

    .venv/bin/python scripts/e24_report.py
"""

from __future__ import annotations

import glob
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
from e13_report import GROUPS  # noqa: E402
from e21_report import md  # noqa: E402
from lth import RESULTS  # noqa: E402

OUT = RESULTS / "e24_cross_style"
STEPS = ("b", "c", "d", "e")
EARLIER = ("e13_capped", "e21_bai_rules", "e22_all_combos", "e23_best_sizes")  # E21: DBS at 40
ALL_SIZES = (20, 40, 60, 80, 100, 120)
# Steps D and E: the same rules in both scopes, as (set-name pattern with {p} scope prefix and {n} size, label).
FILTERS = ("DBMN", "DBTMS", "DBTLS", "DBTM",  # step D
           "DBS", "DBLN", "DBLS", "DBMS", "DBTL", "DBTS", "DLMN", "DTLM", "DBLMN", "DBLMS", "DBTLM", "DTLMS",
           "DBTLMS")  # step E
RULES = ([("{p}causal{n}", "Causal"), ("{p}causal{n}_nogen", "Causal $-$ generic")]
         + [("{p}%s{n}" % c, c) for c in FILTERS]
         + [("{p}yang{n}", "Yang terms"), ("random_ex_{p}db{n}", "Random")])
PREFIX = {"task": "", "shared": "shared_"}
COMBOS = ("D", "B", "DB", "DBMS")
GRID_SIZES = (40, 100)
COLOUR = {"Causal": rt.ORANGE, "Causal $-$ generic": rt.YELLOW, "Yang terms": rt.MAGENTA, "Random": rt.GREY}


def load_step(step: str) -> pd.DataFrame:
    rows = []
    for fold in (1, 2):
        for p in glob.glob(str(OUT / step / f"fold{fold}" / "patching_simple.part-*.csv")):
            rows.append(pd.read_csv(p).assign(fold=fold))
    d = pd.concat(rows, ignore_index=True)
    d = d[d.source != "none"]
    assert (d["style"] == d["mean_style"]).all()
    n_means = d.groupby(["fold", "task", "heads", "style", "source", "alpha"]).mean_wording.nunique().reset_index()
    expect = np.where((n_means["style"] == "rule") & (n_means.source == "task"), 5, 1)
    assert (n_means.mean_wording == expect).all(), "a set lacks a mean (rule: five, example and control: one)"
    c = (d.groupby(["fold", "task", "heads", "style", "source", "alpha"], as_index=False)
         .agg(accuracy=("accuracy", "mean"), n_heads=("n_heads", "first")))
    return c.rename(columns={"heads": "set", "style": "means"}).assign(step=step)


def crossfit(c: pd.DataFrame) -> pd.DataFrame:
    c = c.assign(base=c.set.str.replace(r"_\d$", "", regex=True).where(c.set.str.startswith("random"), c.set))
    g = c.groupby(["fold", "task", "base", "means", "source", "alpha"], as_index=False).agg(
        accuracy=("accuracy", "mean"), n_heads=("n_heads", "min"))
    t = g[g.source == "task"].sort_values(["accuracy", "alpha"], ascending=[False, True])
    best = t.drop_duplicates(["fold", "task", "base", "means"])[["fold", "task", "base", "means", "alpha"]]
    best = best.assign(fold=3 - best.fold)
    out = g.merge(best, on=["fold", "task", "base", "means", "alpha"])
    return out.rename(columns={"base": "set"})


def earlier_rule() -> pd.DataFrame:
    """Rule-mean accuracies (task, control) of the sets E13/E22/E23 ran, per task (folds averaged)."""
    p = pd.concat([pd.read_csv(RESULTS / x / "per_task.csv") for x in EARLIER])
    p = p.drop_duplicates(["task", "set", "source"])
    p["set"] = p.set.str.replace(r"^shared_random_", "random_shared_", regex=True)
    return p[["task", "set", "source", "accuracy", "n_heads"]].assign(means="rule")


def summarise(cf: pd.DataFrame) -> pd.DataFrame:
    new = cf.groupby(["task", "set", "means", "source"], as_index=False).agg(
        accuracy=("accuracy", "mean"), n_heads=("n_heads", "min"), folds=("fold", "nunique"),
        alpha2=("alpha", lambda a: float(np.mean(a == 2))))
    assert (new.folds == 2).all()
    return new


def per_task() -> pd.DataFrame:
    cf = pd.concat([crossfit(load_step(s)).assign(step=s) for s in STEPS], ignore_index=True)
    key = ["fold", "task", "set", "means", "source"]
    # Step E reruns step C's DBMS-40/100 (rebuilt by E22's rule); the reruns must agree, then one is kept.
    dup = cf[cf.duplicated(key, keep=False)]
    assert set(dup.set.str.replace("shared_", "")) <= {"DBMS40", "DBMS100"}
    assert (dup.groupby(key).accuracy.agg(lambda a: a.max() - a.min()) == 0).all()
    cf = cf.drop_duplicates(key)
    new = summarise(cf)
    old = earlier_rule()
    old = old[~old.set.isin(new[new.means == "rule"].set)]
    return pd.concat([new, old.assign(folds=2)], ignore_index=True)


def size_of(name: str) -> int:
    return int(re.search(r"(\d+)(?:_nogen)?$", name).group(1))


def acc(per: pd.DataFrame, s: str, means: str, source: str = "task", tasks=None) -> float:
    """Mean over the tasks; NaN if the set is missing on a task or holds fewer than N heads there."""
    x = per[(per.set == s) & (per.means == means) & (per.source == source)]
    want = list(tasks) if tasks is not None else None
    if want is not None:
        x = x[x.task.isin(want)]
    if x.task.nunique() < (len(want) if want is not None else 14) or (x.n_heads < size_of(s)).any():
        return np.nan
    return x.accuracy.mean()


def cell(v: float, bold: bool = False) -> str:
    if np.isnan(v):
        return "--"
    s = "%.2f" % v
    return r"\textbf{" + s + "}" if bold else s


def rows_at(n: int, scope: str) -> list[tuple[str, str]]:
    return [(pat.format(p=PREFIX[scope], n=n), label) for pat, label in RULES]


def table_scope(per: pd.DataFrame, scope: str) -> tuple[str, list[dict]]:
    """Rows: the rules; columns: N, each with rule means and the example mean; bold: best non-random of the column."""
    rows = [r"\begin{tabular}{@{}l" + "rr" * len(ALL_SIZES) + "@{}}", r"\toprule",
            " & " + " & ".join(r"\multicolumn{2}{c}{$N=%d$}" % n for n in ALL_SIZES) + r" \\",
            "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(len(ALL_SIZES))),
            "Rule & " + " & ".join(["rule & ex."] * len(ALL_SIZES)) + r" \\", r"\midrule"]
    vals, out = {}, []
    for pat, label in RULES:
        v = []
        for n in ALL_SIZES:
            s = pat.format(p=PREFIX[scope], n=n)
            v += [acc(per, s, "rule"), acc(per, s, "example")]
        vals[label] = v
        out.append(dict(scope=scope, rule=label.replace("$-$", "-"),
                        **{f"{m}{n}": v[2 * i + j] for i, n in enumerate(ALL_SIZES) for j, m in enumerate(("rule", "ex"))}))
    best = [np.nanmax([vals[l][i] for l in vals if l != "Random"]) for i in range(2 * len(ALL_SIZES))]
    for k, (label, v) in enumerate(vals.items()):
        if label in ("DBMN", "Yang terms"):
            rows.append(r"\midrule")
        rows.append(label + " & " + " & ".join(
            cell(x, bold=label != "Random" and not np.isnan(x) and round(x, 2) == round(best[i], 2))
            for i, x in enumerate(v)) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(rows), out


def table_types(per: pd.DataFrame, scope: str) -> str:
    """By task type, one block per N, one scope; a long table (several pages)."""
    ncol = 1 + 2 * len(GROUPS)
    head = [" & " + " & ".join(r"\multicolumn{2}{c}{" + g + "}" for g in GROUPS) + r" \\",
            "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(len(GROUPS))),
            "Rule & " + " & ".join(["rule & ex."] * len(GROUPS)) + r" \\"]
    cap = (r"\caption{%s: the rules of Table~\ref{tab:e24-sizes-%s} by task type, with rule means (rule) and the "
           r"example mean (ex.), one block per set size $N$. Check~1, held-out items, cross-fitted $\alpha\in\{1,2\}$, "
           r"mean of the two folds and of the type's tasks. --: fewer than $N$ heads on some task.}\label{tab:e24-types-%s}\\"
           % ("One set per task" if scope == "task" else "One set for all tasks", scope, scope))
    rows = [r"\begin{longtable}{@{}l" + "rr" * len(GROUPS) + "@{}}", cap, r"\toprule", *head, r"\midrule",
            r"\endfirsthead", r"\multicolumn{%d}{@{}l}{\textit{Table~\ref{tab:e24-types-%s}, continued}} \\" % (ncol, scope),
            r"\toprule", *head, r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for k, n in enumerate(ALL_SIZES):
        if k:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{%d}{@{}l}{\textit{$N=%d$ heads}} \\*" % (ncol, n))
        for s, label in rows_at(n, scope):
            cells = []
            for ts in GROUPS.values():
                cells += [cell(acc(per, s, "rule", tasks=ts)), cell(acc(per, s, "example", tasks=ts))]
            rows.append(label + " & " + " & ".join(cells) + r" \\")
    rows += [r"\end{longtable}", ""]
    return "\n".join(rows)


def table_grid(per: pd.DataFrame) -> tuple[str, list[dict]]:
    """Built on rule or example prompts x patched with rule or example means, D/B/DB/DBMS, one block per N."""
    rows = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
            r" & & \multicolumn{2}{c}{Built on rule prompts} & \multicolumn{2}{c}{Built on example prompts} & \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r"Scope & Set & rule means & ex.\ mean & rule means & ex.\ mean & overlap \\"]
    ov = pd.read_csv(OUT / "b" / "overlap.csv")
    out = []
    for n in GRID_SIZES:
        rows += [r"\midrule", r"\multicolumn{7}{@{}l}{\textit{$N=%d$ heads}} \\" % n]
        for scope in ("task", "shared"):
            prefix = PREFIX[scope]
            for c in COMBOS + ("random",):
                if c == "random":
                    rule_set, ex_set, label = None, f"random_ex_{prefix}db{n}", "Random"
                else:
                    rule_set, ex_set, label = f"{prefix}{c}{n}", f"ex_{prefix}{c}{n}", c
                v = [acc(per, rule_set, "rule") if rule_set else np.nan,
                     acc(per, rule_set, "example") if rule_set else np.nan,
                     acc(per, ex_set, "rule"), acc(per, ex_set, "example")]
                o = ov[(ov.scope == scope) & (ov.combo == c) & (ov.n == n)] if c != "random" else None
                share = "--" if o is None else "%.2f (%.2f)" % ((o.shared / n).mean(), (o.chance_layer / n).mean())
                rows.append(("Per task" if scope == "task" else "Shared") * (c == COMBOS[0]) + f" & {label} & "
                            + " & ".join(cell(x) for x in v) + f" & {share} \\\\")
                out.append(dict(n=n, scope=scope, set=label, rule_built_rule=v[0], rule_built_example=v[1],
                                example_built_rule=v[2], example_built_example=v[3], overlap=share))
    rows += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(rows), out


def figure_sizes(per: pd.DataFrame) -> None:
    """Accuracy against N: rows per task / shared, columns rule means / example mean; filter combinations as
    thin lines, the causal rankings, Yang terms and random in colour."""
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), sharex=True, sharey=True)
    for i, scope in enumerate(("task", "shared")):
        for j, (means, title) in enumerate((("rule", "rule means"), ("example", "example mean"))):
            ax = axes[i, j]
            for k, (pat, label) in enumerate(RULES):
                y = [acc(per, pat.format(p=PREFIX[scope], n=n), means) for n in ALL_SIZES]
                if label in FILTERS:
                    ax.plot(ALL_SIZES, y, "-", color=rt.BLUE, lw=0.8, alpha=0.45,
                            label="filter combinations" if label == FILTERS[0] else None)
                else:
                    ax.plot(ALL_SIZES, y, "o-" if label != "Random" else "o--", color=COLOUR[label], ms=3, lw=1.3,
                            label=label.replace("$-$", "−"), zorder=3)
            ax.set_title(f"({'abcd'[2 * i + j]}) {'Per task' if scope == 'task' else 'Shared'}, {title}", loc="left")
            ax.set_xticks(ALL_SIZES)
            ax.set_ylim(0, 0.85)
            if i == 1:
                ax.set_xlabel("heads in the set ($N$)")
            if j == 0:
                ax.set_ylabel("accuracy, check 1")
    axes[1, 1].legend(frameon=False, fontsize=6.5, ncol=1, loc="upper left")
    fig.tight_layout()
    fig.savefig(rt.FIGURES / "e24_sizes.pdf")
    plt.close(fig)


def figure_grid(per: pd.DataFrame) -> None:
    """Built on rule or example prompts x patched with rule or example means, D/B/DB/DBMS: rows scope,
    columns N. Colour: the means; fill: the prompts the set was built on."""
    bars = [("rule", "rule", rt.BLUE, True), ("rule", "example", rt.ORANGE, True),
            ("example", "rule", rt.BLUE, False), ("example", "example", rt.ORANGE, False)]
    fig, axes = plt.subplots(2, len(GRID_SIZES), figsize=(6.6, 4.2), sharey=True)
    x = np.arange(len(COMBOS))
    w = 0.19
    for i, scope in enumerate(("task", "shared")):
        for j, n in enumerate(GRID_SIZES):
            ax = axes[i, j]
            p = PREFIX[scope]
            for k, (built, means, colour, filled) in enumerate(bars):
                v = [acc(per, (f"{p}{c}{n}" if built == "rule" else f"ex_{p}{c}{n}"), means) for c in COMBOS]
                ax.bar(x + (k - 1.5) * (w + 0.02), v, width=w, color=colour if filled else rt.SURFACE,
                       edgecolor=colour, linewidth=1.1, hatch=None if filled else "////", zorder=3)
            r = [acc(per, f"random_ex_{p}db{n}", m) for m in ("rule", "example")]
            ax.axhline(np.nanmax(r), color=rt.GREY, lw=0.8, ls=rt.CONTROL_STYLE, zorder=2)
            ax.set_xticks(x, COMBOS)
            ax.set_ylim(0, 0.85)
            ax.set_title(f"({'abcd'[2 * i + j]}) {'Per task' if scope == 'task' else 'Shared'}, $N={n}$",
                         loc="left", fontsize=7.5)
            rt.style_axes(ax)
    fig.supylabel("check 1 accuracy, mean of 14 tasks", fontsize=8, color=rt.INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c if f else rt.SURFACE, edgecolor=c, hatch=None if f else "////")
               for _, _, c, f in bars]
    labels = ["built on rule prompts, rule means", "built on rule prompts, example mean",
              "built on example prompts, rule means", "built on example prompts, example mean"]
    handles.append(plt.Line2D([], [], color=rt.GREY, lw=0.8, ls=rt.CONTROL_STYLE))
    labels.append("random heads (higher of the two means)")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=6.4)
    fig.tight_layout(rect=(0.02, 0, 1, 0.88), w_pad=1.0, h_pad=1.0)
    fig.savefig(rt.FIGURES / "e24_grid.pdf")
    plt.close(fig)


def figure_overlap() -> None:
    o = pd.read_csv(OUT / "b" / "overlap.csv")
    o = o.assign(share=o.shared / o.n, chance=o.chance_layer / o.n)
    m = o.groupby(["scope", "combo", "n"], as_index=False)[["share", "chance"]].mean()
    colour = {"D": rt.BLUE, "B": rt.GREY, "DB": rt.AQUA, "DBMS": rt.ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, (scope, title) in zip(axes, (("task", "(a) Per task"), ("shared", "(b) Shared"))):
        for c in COMBOS:
            x = m[(m.scope == scope) & (m.combo == c)]
            ax.plot(x.n, x.share, "o-", color=colour[c], ms=3.5, lw=1.2, label=c)
            ax.plot(x.n, x.chance, ls=rt.CONTROL_STYLE, color=colour[c], lw=1.0)
        ax.set_title(title, loc="left")
        ax.set_xticks(sorted(m.n.unique()))
        ax.set_xlabel("heads in the set ($N$)")
        ax.set_ylim(0, 0.7)
    axes[0].set_ylabel("share of heads in both sets")
    axes[1].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(rt.FIGURES / "e24_overlap.pdf")
    plt.close(fig)


def step_a_notes(per: pd.DataFrame) -> dict:
    """E13's random-100 draws with both means, and the rerun check of step A against step D."""
    a = summarise(crossfit(load_step("a")))
    notes = {}
    for s in ("random_dbt100", "random_shared_dbt100"):
        notes[s] = dict(rule=acc(per, s, "rule"), example=a[(a.set == s) & (a.source == "task")].accuracy.mean())
    both = a.merge(per[per.means == "example"], on=["task", "set", "means", "source"], suffixes=("_a", "_d"))
    notes["rerun_max_abs_diff_task_mean"] = float(
        both[both.source == "task"].groupby("set").apply(lambda g: abs(g.accuracy_a.mean() - g.accuracy_d.mean())).max())
    return notes


def main() -> None:
    per = per_task()
    per.to_csv(OUT / "per_task.csv", index=False)
    sizes_md = []
    for scope in ("task", "shared"):
        tex, rows = table_scope(per, scope)
        rt.write(f"e24_sizes_{scope}", tex)
        sizes_md += rows
    grid_tex, grid_md = table_grid(per)
    rt.write("e24_types_task", table_types(per, "task"))
    rt.write("e24_types_shared", table_types(per, "shared"))
    rt.write("e24_grid", grid_tex)
    figure_sizes(per)
    figure_overlap()
    figure_grid(per)
    pd.DataFrame(sizes_md).to_csv(OUT / "summary.csv", index=False)
    names = {pat.format(p=PREFIX[sc], n=n) for pat, _ in RULES[:-1] for sc in PREFIX for n in ALL_SIZES}
    ctrl = per[(per.source == "control") & per.set.isin(names)].groupby(["set", "means"]).accuracy.mean()
    notes = step_a_notes(per)
    lists = pd.read_csv(OUT / "b" / "lists.csv")
    ov = pd.read_csv(OUT / "b" / "overlap.csv")
    with (OUT / "report.md").open("w") as f:
        f.write("# E24: rule and example prompts\n\n")
        f.write("Check 1, held-out items, cross-fitted alpha in {1, 2}, mean of two folds and 14 tasks. "
                "Every comparison within one set size N.\n\n")
        f.write("## Step D: rules at every N, rule means vs example mean\n\n"
                + md(pd.DataFrame(sizes_md).round(3).set_index("rule")) + "\n\n")
        f.write(f"Control means of these sets: max {ctrl.max():.3f} (rule means {ctrl.xs('rule', level=1).max():.3f}, "
                f"example mean {ctrl.xs('example', level=1).max():.3f}).\n\n")
        f.write("## Steps B and C: built on rule or example prompts x rule or example means\n\n"
                + md(pd.DataFrame(grid_md).round(3).set_index("n")) + "\n\n")
        o = ov.assign(share=ov.shared / ov.n, chance=ov.chance_layer / ov.n)
        f.write("## Overlap of rule- and example-built sets (share of N; chance: layer-matched)\n\n"
                + o.groupby(["scope", "combo", "n"])[["share", "chance"]].mean().round(3).reset_index()
                .pipe(lambda d: md(d.set_index("scope"))) + "\n\n")
        f.write("## Task-word lists: words shared by the 5-shot and the rule list (of 30)\n\n"
                + md(lists.groupby("task")[["shared_words"]].mean()) + "\n\n")
        f.write("## Step A notes\n\nE13's random-100 draws (layer-matched to DBT-100): "
                + "; ".join(f"{k}: rule {v['rule']:.3f}, example {v['example']:.3f}" for k, v in notes.items()
                            if isinstance(v, dict))
                + f". Step A rerun in step D, largest difference of a set's 14-task mean: "
                  f"{notes['rerun_max_abs_diff_task_mean']:.4f}.\n")
    print((OUT / "report.md").read_text())


if __name__ == "__main__":
    main()
