"""E13 report: the capped grid (scripts/e13_sets.py, e13_patch.sh, e13_fv_residual.py), check 1.

Accuracy of every set and task at the **cross-fitted alpha over {1, 2}** (fold 1 at the alpha that was
better on fold 2 for the same task and set, and the reverse; ties to alpha 1), mean of the two
folds. Head-patched sets: task accuracy = mean over the five rule means (``heldout_report.load``).
Random draws take the alpha of the DBT set of the same scope and N, and are averaged over the five
draws. Davidson et al.'s summed vector (``fvsum``) is cross-fitted the same way.

Writes ``results/e13_capped/{accuracy_alpha,per_task,summary}.csv``, the tables
``paper/tables/e13_*.tex`` and the figures ``paper/figures/e13_*.pdf``.

    .venv/bin/python scripts/e13_report.py
"""

from __future__ import annotations

import glob
import os
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

import heldout_report as hr  # noqa: E402
import results_tex as rt  # noqa: E402
from lth import RESULTS, simple  # noqa: E402

OUT = RESULTS / "e13_capped"
SIZES = (20, 40, 60, 80, 100, 120)  # 100 and 120 added 2026-11-12
N_DETAIL = int(os.environ.get("E13_N_DETAIL", 40))  # the N of the per-task table (task types: every N)
GROUPS = {
    "Word relations": ("antonym", "synonym"),
    "Morphology": ("singular-plural", "present-past"),
    "Facts": ("country-capital", "country-currency", "landmark-country", "park-country", "product-company"),
    "Sequences": ("next_item", "prev_item"),
    "Translations": ("english-french", "english-german", "english-spanish"),
}
TASK_ORDER = [t for ts in GROUPS.values() for t in ts]
# condition key -> (label, family)
CONDITIONS = {
    "D": ("D", "filter"), "B": ("B", "filter"), "T": ("T", "filter"), "DB": ("DB", "filter"),
    "DT": ("DT", "filter"), "BT": ("BT", "filter"), "DBT": ("DBT", "filter"),
    "DBT_nogen": ("DBT $-$ generic", "filter"),
    "causal": ("Causal", "causal"), "causal_nogen": ("Causal $-$ generic", "causal"),
    "fvsum": ("Davidson FV (summed)", "causal"), "fvsum_nogen": ("Davidson FV $-$ generic (summed)", "causal"),
    "yang": ("Yang terms", "yang"), "random": ("Random (DBT layers)", "random"),
}
FAMILY_TITLE = {"filter": "Decoding and attention filters", "causal": "Davidson et al.\\ (causal score)",
                "yang": "Baseline", "random": "Control"}
COLOUR = {"filter": rt.AQUA, "causal": rt.ORANGE, "yang": rt.BLUE, "random": rt.GREY}


def parse(name: str) -> tuple[str, str, int]:
    """set name -> (scope, condition key, N)."""
    shared = name.startswith(("shared_", "random_shared_", "fvsum"))
    if name.startswith("random"):
        return ("shared" if shared else "task"), "random", int(re.search(r"dbt(\d+)_\d$", name).group(1))
    core = name.removeprefix("shared_")
    m = re.fullmatch(r"([A-Za-z]+?)(\d+)(_nogen)?", core)
    return ("shared" if shared else "task"), m.group(1) + (m.group(3) or ""), int(m.group(2))


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    hr.HELD = OUT
    curves, base = hr.load()
    fv = []
    for fold in (1, 2):
        for path in glob.glob(str(OUT / f"fold{fold}" / "fv_residual.part-*.csv")):
            fv.append(pd.read_csv(path).assign(fold=fold))
    if fv:
        fv = pd.concat(fv, ignore_index=True)
        fv = fv[fv.source != "none"].assign(means=1)[list(curves.columns)]
        curves = pd.concat([curves, fv], ignore_index=True)
    curves = curves[curves.family == "simple"]
    meta = pd.DataFrame([parse(s) for s in curves.set], columns=["scope", "condition", "n"], index=curves.index)
    return pd.concat([curves, meta], axis=1), base[base.family == "simple"]


def crossfit(c: pd.DataFrame) -> pd.DataFrame:
    """Per fold, task, set, source: accuracy at the alpha chosen on the other fold (random draws: that
    of the DBT set of the same scope and N), draws averaged."""
    t = c[(c.source == "task") & (c.condition != "random")]
    t = t.assign(key=list(zip(-t.accuracy, t.alpha.abs())))
    best = t.sort_values("key").drop_duplicates(["fold", "task", "set"])[["fold", "task", "set", "alpha"]]
    best = best.assign(fold=3 - best.fold)  # fold f takes the alpha of fold 3 - f
    dbt = pd.Series(np.where(c.scope == "shared", "shared_DBT", "DBT"), index=c.index) + c.n.astype(str)
    base = c.set.where(c.condition != "random", dbt)
    c = c.assign(base=base)
    out = c.merge(best.rename(columns={"set": "base"}), on=["fold", "task", "base", "alpha"])
    out["set"] = np.where(out.condition == "random", out.base.str.replace("DBT", "random_dbt", regex=False), out.set)
    return (out.groupby(["fold", "task", "set", "scope", "condition", "n", "source"], as_index=False)
            .agg(accuracy=("accuracy", "mean"), n_heads=("n_heads", "mean"), alpha=("alpha", "first")))


def summarise(cf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per = (cf.groupby(["task", "set", "scope", "condition", "n", "source"], as_index=False)
           .agg(accuracy=("accuracy", "mean"), n_heads=("n_heads", "mean"), folds=("fold", "nunique"),
                alpha2=("alpha", lambda a: float(np.mean(a == 2)))))
    rows = []
    for (s, scope, cond, n), g in per.groupby(["set", "scope", "condition", "n"]):
        task, ctrl = g[g.source == "task"], g[g.source == "control"]
        f1 = cf[(cf.set == s) & (cf.source == "task")]
        rows.append(dict(set=s, scope=scope, condition=cond, n=n, tasks=task.task.nunique(), folds=int(task.folds.min()),
                         heads_median=task.n_heads.median(), heads_min=task.n_heads.min(),
                         check1_task=task.accuracy.mean(), check1_control=ctrl.accuracy.mean() if len(ctrl) else np.nan,
                         fold1=f1[f1.fold == 1].accuracy.mean(), fold2=f1[f1.fold == 2].accuracy.mean(),
                         alpha2_share=task.alpha2.mean()))
    return per, pd.DataFrame(rows)


# --------------------------------------------------------------------------- tables

def cell(v, bold=False) -> str:
    s = rt.num(v) if not (isinstance(v, float) and np.isnan(v)) else "--"
    return r"\textbf{" + s + "}" if bold else s


def tex_main(s: pd.DataFrame, value: str, name: str) -> None:
    """Rows: conditions by family; columns: per task and shared at each N."""
    idx = s.set_index(["condition", "scope", "n"])[value]
    best = {(scope, n): s[(s.scope == scope) & (s.n == n) & (s.condition != "random")][value].max()
            for scope in ("task", "shared") for n in SIZES}
    lines, fam = [], None
    for cond, (label, family) in CONDITIONS.items():
        if family != fam:
            if fam is not None:
                lines.append(r"\midrule")
            lines.append(r"\multicolumn{" + str(1 + 2 * len(SIZES)) + r"}{@{}l}{\textit{" + FAMILY_TITLE[family] + r"}} \\")
            fam = family
        cells = []
        for scope in ("task", "shared"):
            for n in SIZES:
                v = idx.get((cond, scope, n), np.nan)
                cells.append(cell(v, value == "check1_task" and cond != "random" and v == best[(scope, n)]))
        lines.append(r"\quad " + label + " & " + " & ".join(cells) + r" \\")
    k = len(SIZES)
    header = [rf" & \multicolumn{{{k}}}{{c}}{{One set per task}} & \multicolumn{{{k}}}{{c}}{{One set for all 14 tasks}} \\",
              rf"\cmidrule(lr){{2-{k + 1}}}\cmidrule(lr){{{k + 2}-{2 * k + 1}}}",
              r"Heads (N) & " + " & ".join(str(n) for n in SIZES * 2) + r" \\"]
    text = "\n".join([r"\begin{tabular}{@{}l" + "c" * (2 * len(SIZES)) + r"@{}}", r"\toprule", *header, r"\midrule", *lines,
                      r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / f"{name}.tex").write_text("% Generated by scripts/e13_report.py.\n" + text + "\n")


DETAIL = [  # (scope, condition, short column label; "\n" = line break)
    ("task", "DBT", "DBT"), ("task", "DB", "DB"), ("task", "T", "T"), ("task", "DBT_nogen", "DBT\n$-$gen."),
    ("task", "causal", "Causal"), ("task", "causal_nogen", "Causal\n$-$gen."), ("task", "yang", "Yang"),
    ("shared", "DBT", "DBT"), ("shared", "DBT_nogen", "DBT\n$-$gen."), ("shared", "causal", "Causal"),
    ("shared", "causal_nogen", "Causal\n$-$gen."), ("shared", "fvsum", "FV\nsum"),
    ("shared", "fvsum_nogen", "FV sum\n$-$gen."), ("shared", "yang", "Yang"),
]


def detail_grid(per: pd.DataFrame, n: int, value="accuracy") -> pd.DataFrame:
    q = per[(per.source == "task") & (per.n == n)].set_index(["scope", "condition", "task"])[value]
    return pd.DataFrame({i: [q.get((sc, c, t), np.nan) for t in TASK_ORDER] for i, (sc, c, _) in enumerate(DETAIL)},
                        index=TASK_ORDER)


def tex_head(label: str) -> str:
    return (r"\begin{tabular}[b]{@{}c@{}}" + label.replace("\n", r"\\") + r"\end{tabular}") if "\n" in label else label


def header_two_scopes(lead: int = 1) -> list[str]:
    """Scope header over the DETAIL columns, after ``lead`` leading columns."""
    n_task = sum(1 for sc, _, _ in DETAIL if sc == "task")
    a, b, c = lead + 1, lead + n_task, lead + len(DETAIL)
    return [" & " * lead + r"\multicolumn{" + str(n_task) + r"}{c}{One set per task} & \multicolumn{"
            + str(len(DETAIL) - n_task) + r"}{c}{One set for all tasks} \\",
            rf"\cmidrule(lr){{{a}-{b}}}\cmidrule(lr){{{b + 1}-{c}}}"]


def tex_groups(per: pd.DataFrame) -> None:
    """Task-type means of the DETAIL columns, one block per N."""
    rows = []
    for n in SIZES:
        if rows:
            rows.append(r"\addlinespace")
        rows.append(r"\multicolumn{" + str(len(DETAIL) + 1) + r"}{@{}l}{\textit{" + f"{n} heads" + r"}} \\")
        g = detail_grid(per, n)
        for group, tasks in GROUPS.items():
            vals = g.loc[list(tasks)].mean()
            top = vals.max()
            rows.append(r"\quad " + group + f" ({len(tasks)}) & " + " & ".join(cell(v, v == top) for v in vals)
                        + r" \\")
    head = header_two_scopes() + ["Task type & " + " & ".join(tex_head(lab) for _, _, lab in DETAIL) + r" \\"]
    text = "\n".join([r"\begin{tabular}{@{}l" + "c" * len(DETAIL) + r"@{}}", r"\toprule", *head, r"\midrule", *rows,
                      r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / "e13_groups.tex").write_text("% Generated by scripts/e13_report.py.\n" + text + "\n")


def tex_tasks(per: pd.DataFrame, base: pd.DataFrame, n: int) -> None:
    g = detail_grid(per, n)
    sizes = detail_grid(per, n, "n_heads")
    b = base.groupby("task").accuracy.mean().reindex(TASK_ORDER)
    lines = []
    for group, tasks in GROUPS.items():
        lines.append(r"\multicolumn{" + str(len(DETAIL) + 2) + r"}{@{}l}{\textit{" + group + r"}} \\")
        for t in tasks:
            vals = g.loc[t]
            top = vals.max()
            short = [f"$^{{{int(round(sizes.loc[t, i]))}}}$" if c == "yang" and sizes.loc[t, i] < n - 0.5 else ""
                     for i, (_, c, _) in enumerate(DETAIL)]
            lines.append(rt.tex_name("simple", t) + f" & {rt.num(b[t])} & "
                         + " & ".join(cell(v, v == top and top > 0) + sh for v, sh in zip(vals, short)) + r" \\")
    lines += [r"\midrule", "Mean & " + rt.num(b.mean()) + " & " + " & ".join(cell(v) for v in g.mean()) + r" \\"]
    head = header_two_scopes(lead=2) + ["Task & Unp. & " + " & ".join(tex_head(lab) for _, _, lab in DETAIL) + r" \\"]
    text = "\n".join([r"\begin{tabular}{@{}lc" + "c" * len(DETAIL) + r"@{}}", r"\toprule", *head, r"\midrule",
                      *lines, r"\bottomrule", r"\end{tabular}"])
    (rt.TABLES / "e13_tasks.tex").write_text("% Generated by scripts/e13_report.py.\n" + text + "\n")


def tex_word(w: str) -> str:
    """A list word for LaTeX: Chinese, Japanese and Cyrillic words in the doc's \\zh, \\ja, \\ru fonts;
    other non-Latin words (only among the generic words) by a description."""
    import unicodedata

    names = [unicodedata.name(ch, "") for ch in w]
    if any(n.startswith(("HIRAGANA", "KATAKANA")) for n in names):
        return r"\ja{" + w + "}"
    if any(n.startswith("CJK") for n in names):
        return r"\zh{" + w + "}"
    if any(n.startswith("CYRILLIC") for n in names):
        return r"\ru{" + w + "}"
    described = {"م": "[Arabic letter]", "क": "[Devanagari letter]", "ๆ": "[Thai repetition mark]",
                 "คำ": "[Thai `word']", "데이트": "[Korean `date']"}
    if w in described:
        return described[w]
    assert all(ch.isascii() or n.startswith("LATIN") for ch, n in zip(w, names)), w
    return w


def tex_lists() -> None:
    """Every task's list (fold 1), its T words, and what fold 2 changes; and the generic words."""
    import json

    L = {f: json.loads((OUT / f"fold{f}" / "word_lists.json").read_text()) for f in (1, 2)}
    info = {f: json.loads((RESULTS.parent / "data" / "readings" / "heldout" / f"attention13_{h}" / "info.json")
                          .read_text()) for f, h in ((1, "A"), (2, "B"))}
    rows = []
    for t in TASK_ORDER:
        w1, w2 = L[1]["tasks"][t]["words"], L[2]["tasks"][t]["words"]
        tw = {f: sorted({w for d in info[f][t]["found"].values() for w in d}) for f in (1, 2)}
        t_cell = ", ".join(map(tex_word, tw[1])) + (
            "" if tw[1] == tw[2] else "; fold 2: " + ", ".join(map(tex_word, tw[2])))
        diff = (("$+$" + ", ".join(tex_word(w) for w in w2 if w not in w1)) if set(w2) - set(w1) else "") + (
            ("; $-$" + ", ".join(tex_word(w) for w in w1 if w not in w2)) if set(w1) - set(w2) else "")
        rows.append(rt.tex_name("simple", t) + " & " + t_cell + " & " + ", ".join(map(tex_word, w1)) + " & "
                    + (diff.lstrip("; ") or "none") + r" \\")
    text = "\n".join([r"\begin{xltabular}{\textwidth}{@{}>{\raggedright\arraybackslash}p{1.9cm}"
                      r">{\raggedright\arraybackslash}p{2.3cm}>{\raggedright\arraybackslash}X"
                      r">{\raggedright\arraybackslash}p{2.9cm}@{}}",
                      r"\caption{Task-word lists (fold 1, by share) and the words of each list that occur in the "
                      r"instructions, which T measures attention to. Last column: fold 2's list against fold 1's.}"
                      r"\label{tab:lists} \\", r"\toprule",
                      r"Task & T words & 30-word list & Fold 2 \\", r"\midrule", r"\endfirsthead",
                      r"\toprule Task & T words & 30-word list & Fold 2 \\ \midrule", r"\endhead",
                      *rows, r"\bottomrule", r"\end{xltabular}"])
    (rt.TABLES / "e13_lists.tex").write_text("% Generated by scripts/e13_report.py.\n" + text + "\n")
    assert L[1]["generic"] == L[2]["generic"]
    gen = ", ".join(tex_word(w) for w, _ in L[1]["generic"])
    (rt.TABLES / "e13_generic.tex").write_text("% Generated by scripts/e13_report.py.\n" + gen + "\n")


# --------------------------------------------------------------------------- figures

LINE = {  # condition -> (style, width, marker, direct label)
    "DBT": ("-", 2.0, "o", "DBT"), "DBT_nogen": ((0, (4, 2)), 1.4, "o", "DBT $-$ gen."),
    "causal": ("-", 2.0, "s", "Causal"), "causal_nogen": ((0, (4, 2)), 1.4, "s", "Causal $-$ gen."),
    "fvsum": (":", 1.4, "D", "FV summed"), "fvsum_nogen": ((0, (1, 1.5)), 1.8, "D", "FV summed $-$ gen."),
    "yang": ("-", 1.4, "^", "Yang terms"), "random": ("-", 1.2, "v", "random"),
}


def figure_size(s: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), sharey=True)
    for ax, scope, title in ((axes[0], "task", "(a) One set per task"), (axes[1], "shared", "(b) One set for all 14 tasks")):
        g = s[s.scope == scope]
        ends = []
        for cond, (ls, lw, mk, lab) in LINE.items():
            r = g[g.condition == cond].sort_values("n")
            if r.empty:
                continue
            col = COLOUR[CONDITIONS[cond][1]]
            ax.plot(r.n, r.check1_task, color=col, ls=ls, lw=lw, marker=mk, ms=4.5, mec=rt.SURFACE, mew=0.8, zorder=3)
            ends.append((r.check1_task.iloc[-1], lab, rt.INK_2, 1.0, r.n.iloc[-1]))
        # direct labels at the right end, nudged apart
        ends.sort()
        ys, last = [], -1.0
        for y, lab, colr, a, x in ends:
            y = max(y, last + 0.028)
            ys.append((y, lab, colr, a, x))
            last = y
        for y, lab, colr, a, x in ys:
            ax.annotate(lab, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=5.8,
                        color=colr, alpha=a, annotation_clip=False)
        ax.set_xticks(SIZES)
        ax.set_xlim(14, SIZES[-1] + 6)
        ax.set_ylim(-0.02, 0.75)
        ax.set_xlabel("heads in the set (N)")
        ax.set_title(title, loc="left")
        rt.style_axes(ax)
    axes[0].set_ylabel("check 1 accuracy, mean of 14 tasks")
    handles = [plt.Line2D([], [], color=COLOUR[f], lw=2, marker=m, ms=4) for f, m in
               (("filter", "o"), ("causal", "s"), ("yang", "^"), ("random", "v"))]
    labels = ["decoding + attention filters", "Davidson et al. (causal)", "Yang terms (baseline)", "random heads"]
    handles += [plt.Line2D([], [], color=rt.INK_2, lw=1.4, ls=(0, (4, 2))),
                plt.Line2D([], [], color=rt.ORANGE, lw=1.4, ls=":", marker="D", ms=3.5)]
    labels += ["dashed: generic heads removed", "dotted: Davidson et al.'s summed vector"]
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=6.6)
    fig.tight_layout(rect=(0, 0, 0.97, 0.85), w_pad=3.5)
    fig.savefig(rt.FIGURES / "e13_size.pdf")
    if os.environ.get("SYNTH_PNG"):
        fig.savefig(Path(os.environ["SYNTH_PNG"]) / "e13_size.png", dpi=200)
    plt.close(fig)


def main() -> None:
    curves, base = load()
    curves.to_csv(OUT / "accuracy_alpha.csv", index=False)
    cf = crossfit(curves)
    per, s = summarise(cf)
    per.to_csv(OUT / "per_task.csv", index=False)
    s.to_csv(OUT / "summary.csv", index=False)
    # Tables and figure: only sets run on all 14 tasks in both folds (per-task Yang terms from N = 100 on
    # is built only where it fills N heads, so it has no 14-task mean there).
    full = s[(s.tasks == 14) & (s.folds == 2)]
    tex_main(full, "check1_task", "e13_main")
    tex_main(full, "check1_control", "e13_control")
    tex_groups(per[per.set.isin(full.set)])
    tex_tasks(per, base, N_DETAIL)
    tex_lists()
    figure_size(full)
    pd.set_option("display.width", 220)
    print(s.pivot_table(index="condition", columns=["scope", "n"], values="check1_task").round(3).to_string())
    print("unpatched", round(base.accuracy.mean(), 3))
    print("incomplete sets:", s[(s.tasks < 14) | (s.folds < 2)].set.tolist())


if __name__ == "__main__":
    main()
