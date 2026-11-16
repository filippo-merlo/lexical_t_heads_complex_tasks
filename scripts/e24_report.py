"""E24 report: are the best heads shared between rule (instruction) and example prompts?

Accuracies as ``scripts/e13_report.py``: check 1, held-out items, per fold at the alpha in {1, 2} that
was better on the other fold for the same task, set and mean style (ties to 1), mean of the two
folds, mean of the 14 tasks. Rule means: mean over the five rule templates' means; example means: the
5-shot mean. Random draws take the alpha chosen on the other fold from the mean of their five draws
(E13's random sets, patched with rule means, keep E13's rule: the alpha of DBT of the same scope and
N). Rule-mean accuracies of sets E13/E22/E23 already ran are read from their ``per_task.csv``.

- Step A (``results/e24_cross_style/a``): the best rule-built sets with the example mean.
- Step B (``b``): the example-built D/B/DB/DBMS sets with both means; the overlap of rule- and
  example-built sets (``b/overlap.csv``).
- Step C (``c``): the rule-built D/B/DB/DBMS sets with the example mean (and DBMS100 with rule means).

Writes ``results/e24_cross_style/{per_task,summary}.csv``, ``report.md``, the tables
``paper/tables/e24_{best,grid,types}.tex`` and the figure ``paper/figures/e24_overlap.pdf``.

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
STEPS = ("a", "b", "c")
EARLIER = ("e13_capped", "e22_all_combos", "e23_best_sizes")
BEST = {"task": [("DBTMS100", "DBTMS-100"), ("DBMN60", "DBMN-60"), ("DBT40", "DBT-40"), ("causal120", "Causal-120"),
                 ("random_dbt100", "Random-100 (DBT layers)"), ("random_ex_db100", "Random-100 (ex.\\ DB layers)")],
        "shared": [("shared_causal100_nogen", "Causal $-$ generic-100"), ("shared_causal80_nogen", "Causal $-$ generic-80"),
                   ("shared_DBTMS120", "DBTMS-120"), ("shared_DBMN80", "DBMN-80"),
                   ("random_shared_dbt100", "Random-100 (DBT layers)"),
                   ("random_ex_shared_db100", "Random-100 (ex.\\ DB layers)")]}
COMBOS = ("D", "B", "DB", "DBMS")
SIZES = (40, 100)


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
    g = c.groupby(["fold", "task", "base", "means", "source", "alpha"], as_index=False).accuracy.mean()
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
    return p[["task", "set", "source", "accuracy"]].assign(means="rule")


def per_task() -> pd.DataFrame:
    cf = pd.concat([crossfit(load_step(s)).assign(step=s) for s in STEPS], ignore_index=True)
    assert not cf.duplicated(["fold", "task", "set", "means", "source"]).any()
    new = cf.groupby(["task", "set", "means", "source"], as_index=False).agg(
        accuracy=("accuracy", "mean"), folds=("fold", "nunique"), alpha2=("alpha", lambda a: float(np.mean(a == 2))))
    assert (new.folds == 2).all()
    old = earlier_rule()
    old = old[~old.set.isin(new[new.means == "rule"].set)]
    return pd.concat([new, old.assign(folds=2)], ignore_index=True)


def acc(per: pd.DataFrame, s: str, means: str, source: str = "task", tasks=None) -> float:
    x = per[(per.set == s) & (per.means == means) & (per.source == source)]
    if tasks is not None:
        x = x[x.task.isin(tasks)]
    if not len(x):
        return np.nan
    assert x.task.nunique() == (len(tasks) if tasks is not None else 14), (s, means, source)
    return x.accuracy.mean()


def cell(v: float, bold: bool = False) -> str:
    if np.isnan(v):
        return "--"
    s = "%.2f" % v
    return r"\textbf{" + s + "}" if bold else s


def size_of(name: str) -> int:
    return int(re.search(r"(\d+)(?:_nogen)?$", name).group(1))


def by_size(scope: str) -> list[tuple[int, list[tuple[str, str]]]]:
    """The scope's Step A sets grouped by size N (ascending), so that only sets of one N share a block."""
    sizes = sorted({size_of(s) for s, _ in BEST[scope]})
    return [(n, [(s, l) for s, l in BEST[scope] if size_of(s) == n]) for n in sizes]


def table_best(per: pd.DataFrame) -> tuple[str, list[dict]]:
    """One block per scope and set size N."""
    rows, md = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
                r" & \multicolumn{2}{c}{Rule means} & \multicolumn{2}{c}{Example means} \\",
                r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
                r"Set (chosen on rule prompts) & task & control & task & control \\"], []
    for scope, title in (("task", "Per task"), ("shared", "Shared")):
        for n, sets in by_size(scope):
            rows += [r"\midrule", r"\multicolumn{5}{@{}l}{\textit{" + title + f", $N={n}$" + r"}} \\"]
            for s, label in sets:
                r_t, r_c = acc(per, s, "rule"), acc(per, s, "rule", "control")
                e_t, e_c = acc(per, s, "example"), acc(per, s, "example", "control")
                rows.append(f"{label} & {cell(r_t)} & {cell(r_c)} & {cell(e_t)} & {cell(e_c)} " + r"\\")
                md.append(dict(scope=scope, n=n, set=s, rule_task=r_t, rule_control=r_c, example_task=e_t,
                               example_control=e_c))
    rows += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(rows), md


def table_types(per: pd.DataFrame) -> str:
    """By task type, one block per scope and set size N."""
    ncol = 1 + 2 * len(GROUPS)
    rows = [r"\begin{tabular}{@{}l" + "rr" * len(GROUPS) + "@{}}", r"\toprule",
            " & " + " & ".join(r"\multicolumn{2}{c}{" + g + "}" for g in GROUPS) + r" \\",
            "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(len(GROUPS))),
            "Set & " + " & ".join(["rule & ex."] * len(GROUPS)) + r" \\"]
    for scope, title in (("task", "Per task"), ("shared", "Shared")):
        for n, sets in by_size(scope):
            rows += [r"\midrule", r"\multicolumn{%d}{@{}l}{\textit{%s, $N=%d$}} \\" % (ncol, title, n)]
            for s, label in sets:
                cells = []
                for ts in GROUPS.values():
                    cells += [cell(acc(per, s, "rule", tasks=ts)), cell(acc(per, s, "example", tasks=ts))]
                rows.append(label + " & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(rows)


def table_grid(per: pd.DataFrame) -> tuple[str, list[dict]]:
    """Built on rule or example prompts x patched with rule or example means, D/B/DB/DBMS at 40 and 100."""
    rows = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
            r" & & \multicolumn{2}{c}{Built on rule prompts} & \multicolumn{2}{c}{Built on example prompts} & \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r"Scope & Set & rule means & ex.\ means & rule means & ex.\ means & overlap \\", r"\midrule"]
    ov = pd.read_csv(OUT / "b" / "overlap.csv")
    md = []
    for scope, prefix in (("task", ""), ("shared", "shared_")):
        for n in SIZES:
            for c in COMBOS + ("random",):
                if c == "random":
                    rule_set = None
                    ex_set = f"random_ex_{prefix}db{n}"
                    label = f"Random-{n}"
                else:
                    rule_set, ex_set, label = f"{prefix}{c}{n}", f"ex_{prefix}{c}{n}", f"{c}-{n}"
                v = [acc(per, rule_set, "rule") if rule_set else np.nan,
                     acc(per, rule_set, "example") if rule_set else np.nan,
                     acc(per, ex_set, "rule"), acc(per, ex_set, "example")]
                o = ov[(ov.scope == scope) & (ov.combo == c) & (ov.n == n)] if c != "random" else None
                share = "--" if o is None else "%.2f (%.2f)" % ((o.shared / n).mean(), (o.chance_layer / n).mean())
                rows.append(("Per task" if scope == "task" else "Shared") * (c == COMBOS[0]) + f" & {label} & "
                            + " & ".join(cell(x) for x in v) + f" & {share} \\\\")
                md.append(dict(scope=scope, set=label, rule_built_rule=v[0], rule_built_example=v[1],
                               example_built_rule=v[2], example_built_example=v[3], overlap=share))
            rows.append(r"\midrule" if (scope, n) != ("shared", SIZES[-1]) else "")
    rows += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(r for r in rows if r), md


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


def main() -> None:
    per = per_task()
    per.to_csv(OUT / "per_task.csv", index=False)
    best_tex, best_md = table_best(per)
    grid_tex, grid_md = table_grid(per)
    rt.write("e24_best", best_tex)
    rt.write("e24_types", table_types(per))
    rt.write("e24_grid", grid_tex)
    figure_overlap()
    pd.DataFrame(best_md).to_csv(OUT / "summary.csv", index=False)
    lists = pd.read_csv(OUT / "b" / "lists.csv")
    ov = pd.read_csv(OUT / "b" / "overlap.csv")
    with (OUT / "report.md").open("w") as f:
        f.write("# E24: rule-built sets with example means, and sets built on example prompts\n\n")
        f.write("Check 1, held-out items, cross-fitted alpha in {1, 2}, mean of two folds and 14 tasks.\n\n")
        f.write("## Step A: best rule-built sets\n\n" + md(pd.DataFrame(best_md).round(3).set_index("set")) + "\n\n")
        f.write("## Steps B and C: built on rule or example prompts x rule or example means\n\n"
                + md(pd.DataFrame(grid_md).round(3).set_index("set")) + "\n\n")
        o = ov.assign(share=ov.shared / ov.n, chance=ov.chance_layer / ov.n)
        f.write("## Overlap of rule- and example-built sets (share of N; chance: layer-matched)\n\n"
                + o.groupby(["scope", "combo", "n"])[["share", "chance"]].mean().round(3).reset_index().pipe(lambda d: md(d.set_index("scope"))) + "\n\n")
        f.write("## Task-word lists: words shared by the 5-shot and the rule list (of 30)\n\n"
                + md(lists.groupby("task")[["shared_words"]].mean()) + "\n")
    print((OUT / "report.md").read_text())


if __name__ == "__main__":
    main()
