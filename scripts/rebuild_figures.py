"""Rebuilt design (paper/head-sets-simple-design.tex, Secs. 4-6): figures and tables of stages 1-3.

Reads the cross-fitted accuracies of ``scripts/rebuild_report.py`` (``results/rebuild/additions/
s1_per_task.csv``, ``s2_per_task.csv``), the sets (``fold{1,2}/heads.json``), the selection
(``selected.json``, and the screen scores of ``scripts/rebuild_select.py``) and the generic heads
(``fold{1,2}/generic.json``). A mean over tasks is given only where every task and both folds have the
set (``--`` otherwise); random sets are averaged over their three draws first. Stage 3 (Sec. 6) reads the
same reports of its tests under ``results/rebuild/scenarios/`` and, for the wording groups, the patching
files themselves (each fold and task at its cross-fitted alpha). Writes
``paper/figures/design_*.pdf`` and ``paper/tables/design_*.tex``.

    .venv/bin/python scripts/rebuild_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import results_tex as rt  # noqa: E402  (palette, rcParams, task names, table helpers)
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from lth import simple  # noqa: E402
from rebuild_select import screen_scores  # noqa: E402
from rebuild_sets import FEATURES, SCREEN_N, SIZES  # noqa: E402

DIR = rt.RESULTS / "rebuild" / "additions"
N_HEADS, N_LAYERS = 32, 32
TYPES = {"word relations": ("antonym", "synonym"), "morphology": ("singular-plural", "present-past"),
         "facts": ("country-capital", "country-currency", "landmark-country", "park-country", "product-company"),
         "sequences": ("next_item", "prev_item"), "translations": ("english-french", "english-german", "english-spanish")}
# Rule -> (label, colour, marker). Colours: the validated categorical order (dataviz validator, light:
# all adjacent pairs pass), lexical blue and causal orange as in the project's earlier figures.
RULES = {"yangL": ("Yang terms (Logit)", rt.BLUE, "o"), "causal": ("Causal", rt.ORANGE, "v"),
         "selected": ("Selected", rt.AQUA, "D"), "yangJ": ("Yang terms (J)", rt.YELLOW, "s"),
         "D": ("D", rt.MAGENTA, "^"), "causalnogen": ("Causal $-$ generic", rt.GREEN, "P")}
LADDER = ("yangL", "yangJ", "D", "selected", "causal", "causalnogen")  # order of Table 3
SCOPES = {"per task": "", "shared": "shared_"}
FEATURE_NAMES = {c: name for c, (name, _) in FEATURES.items()}
CONTROL_STYLE = rt.CONTROL_STYLE


def plot_label(rule: str) -> str:
    return RULES[rule][0].replace("$-$", "−")


# ------------------------------------------------------------------ data


def cells(paths) -> pd.Series:
    """accuracy[(set, kind, fold, task)]; kind task / control / random (random: mean of the draws)."""
    pt = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    is_random = pt.set.str.match(r"^random_.+_\d+$")
    pt["base"] = pt.set.where(~is_random, pt.set.str.replace(r"^random_(.+)_\d+$", r"\1", regex=True))
    pt["kind"] = np.where(is_random, "random", pt.source)
    return pt.groupby(["base", "kind", "fold", "task"]).accuracy.mean()


ACC = cells([DIR / f"{s}_per_task.csv" for s in ("s1", "s2")])
# Stage 3 (Sec. 6): test -> (results folder, report stage); a test enters once scripts/rebuild_report.py has run.
SCEN = rt.RESULTS / "rebuild" / "scenarios"
S3 = {"wt": (SCEN / "wordings_transfer", "s3wt"), "sum": (SCEN / "summed", "s3sum")}
ACC3 = {k: cells([d / f"{st}_per_task.csv"]) for k, (d, st) in S3.items() if (d / f"{st}_per_task.csv").exists()}
S3_RULES = ("yangJ", "D", "selected", "causal", "causalnogen")  # the rules carried into Sec. 6


def mean_over(name: str, kind: str = "task", tasks=simple.TASKS, acc=None):
    """Mean over tasks and folds, or None unless every task of ``tasks`` has the set in both folds."""
    try:
        s = (ACC if acc is None else acc).loc[(name, kind)]
    except KeyError:
        return None
    s = s[s.index.get_level_values("task").isin(tasks)]
    return float(s.mean()) if len(s) == 2 * len(tasks) else None


def per_task(name: str, task: str, kind: str = "task"):
    try:
        s = ACC.loc[(name, kind)]
    except KeyError:
        return None
    s = s[s.index.get_level_values("task") == task]
    return float(s.mean()) if len(s) == 2 else None


def heads() -> dict[int, dict[str, dict[str, set]]]:
    """sets[fold][task][name] = set of (layer, head)."""
    out = {}
    for fold in (1, 2):
        h = json.loads((DIR / f"fold{fold}" / "heads.json").read_text())["simple"]
        out[fold] = {t: {n: set(map(tuple, hs)) for _, n, hs in h[t]["sets"]} for t in simple.TASKS}
    return out


SETS = heads()


def per_layer(s: set) -> np.ndarray:
    c = np.zeros(N_LAYERS)
    for l, _ in s:
        c[l] += 1
    return c


def chance_overlap(a: set, b: set) -> float:
    """Expected shared heads of two random sets with these numbers of heads per layer."""
    return float((per_layer(a) * per_layer(b)).sum() / N_HEADS)


def set_stat(rule: str, scope: str, n: int, fn):
    """Mean of fn(sets of the task-fold) over tasks and folds (shared: the one set per fold); fn may
    return a number or an array."""
    name = f"{SCOPES[scope]}{rule}{n}"
    vals = []
    tasks = simple.TASKS[:1] if scope == "shared" else simple.TASKS
    for fold in (1, 2):
        for t in tasks:
            sets = SETS[fold][t]
            if name not in sets:
                return None
            vals.append(fn(sets, name))
    m = np.mean(vals, axis=0)
    return float(m) if np.ndim(m) == 0 else m


# ------------------------------------------------------------------ tables


def write(name: str, text: str) -> None:
    (rt.TABLES / f"{name}.tex").write_text(f"% Generated by scripts/rebuild_figures.py.\n{text}")


def table_own() -> None:
    yang = [len(SETS[f][t]["yang_own"]) for f in (1, 2) for t in simple.TASKS]
    rows = [
        f"Lexical task heads & per task & {min(yang)}--{max(yang)} (median {int(np.median(yang))}) & head by head, "
        f"$\\alpha$ cross-fitted & {rt.num(mean_over('yang_own'))} & {rt.num(mean_over('yang_own', 'control'))} & "
        f"{rt.num(mean_over('yang_own', 'random'))} \\\\",
        f"Instruction function vector & shared & 20 & summed after layer 11, $\\alpha=1$ & "
        f"{rt.num(mean_over('shared_davidson_own'))} & {rt.num(mean_over('shared_davidson_own', 'control'))} & "
        f"{rt.num(mean_over('shared_davidson_own', 'random'))} \\\\"]
    write("design_own", rt.tabular("@{}llllrrr@{}", ["Method & Scope & Heads & Intervention & Task & Control & Random \\\\"],
                                   rows))


def table_ladder() -> None:
    """Task-mean accuracy of every rule at every size, both scopes."""
    header = ["$N$ & " + " & ".join(RULES[r][0] for r in LADDER) + " \\\\"]
    rows = []
    for scope, prefix in SCOPES.items():
        rows.append(f"\\multicolumn{{{len(LADDER) + 1}}}{{@{{}}l}}{{\\textit{{{scope.capitalize()}}}}} \\\\")
        for n in SIZES:
            vals = [mean_over(f"{prefix}{r}{n}") for r in LADDER]
            best = max(v for v in vals if v is not None)
            cells_ = [("\\textbf{" + rt.num(v) + "}") if v is not None and v == best else rt.num(v) for v in vals]
            rows.append(f"{n} & " + " & ".join(cells_) + " \\\\")
        if scope == "per task":
            rows.append("\\midrule")
    write("design_ladder", rt.tabular("@{}r" + "r" * len(LADDER) + "@{}", header, rows))


def table_controls() -> None:
    """Random sets and control means of every rule at every size."""
    header = ["& & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{RULES[r][0]}}}" for r in LADDER) + " \\\\",
              " ".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(LADDER))),
              "Scope & $N$ & " + " & ".join("rand. & ctrl." for _ in LADDER) + " \\\\"]
    rows = []
    for scope, prefix in SCOPES.items():
        for n in SIZES:
            vals = []
            for r in LADDER:
                vals += [rt.num(mean_over(f"{prefix}{r}{n}", "random")), rt.num(mean_over(f"{prefix}{r}{n}", "control"))]
            rows.append(f"{scope if n == SIZES[0] else ''} & {n} & " + " & ".join(vals) + " \\\\")
        if scope == "per task":
            rows.append("\\midrule")
    write("design_controls", rt.tabular("@{}lr" + "rr" * len(LADDER) + "@{}", header, rows))


def table_types(n: int = SCREEN_N) -> None:
    header = ["Rule & " + " & ".join(TYPES) + " & all \\\\"]
    rows = []
    for scope, prefix in SCOPES.items():
        rows.append(f"\\multicolumn{{{len(TYPES) + 2}}}{{@{{}}l}}{{\\textit{{{scope.capitalize()}}}}} \\\\")
        for r in LADDER:
            name = f"{prefix}{r}{n}"
            rows.append(f"{RULES[r][0]} & " + " & ".join(rt.num(mean_over(name, tasks=ts)) for ts in TYPES.values())
                        + f" & {rt.num(mean_over(name))} \\\\")
        if scope == "per task":
            rows.append("\\midrule")
    write("design_types", rt.tabular("@{}l" + "r" * (len(TYPES) + 1) + "@{}", header, rows))


def table_overlap() -> None:
    """Share of heads in common with Causal of the same scope and size, and layer-matched chance;
    share of a rule's heads in its random sets."""
    sizes = (20, 40, 80, 120)
    header = ["& & \\multicolumn{4}{c}{with Causal (chance)} & \\multicolumn{4}{c}{with its random sets} \\\\",
              "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}",
              "Scope & Rule & " + " & ".join(f"$N={n}$" for n in sizes) + " & " + " & ".join(f"{n}" for n in sizes) + " \\\\"]
    rows = []
    for scope, prefix in SCOPES.items():
        for r in LADDER:
            with_causal, with_random = [], []
            for n in sizes:
                ref = f"{prefix}causal{n}"
                if r == "causal":
                    with_causal.append("--")
                else:
                    ov = set_stat(r, scope, n, lambda s, nm: len(s[nm] & s[ref]) / n)
                    ch = set_stat(r, scope, n, lambda s, nm: chance_overlap(s[nm], s[ref]) / n)
                    with_causal.append("--" if ov is None else f"{rt.num(ov)} ({rt.num(ch)})")
                rv = set_stat(r, scope, n, lambda s, nm: np.mean([len(s[nm] & s[f'random_{nm}_{i}']) for i in range(3)]) / n)
                with_random.append(rt.num(rv))
            rows.append(f"{scope if r == LADDER[0] else ''} & {RULES[r][0]} & " + " & ".join(with_causal + with_random) + " \\\\")
        if scope == "per task":
            rows.append("\\midrule")
    write("design_overlap", rt.tabular("@{}ll" + "r" * 8 + "@{}", header, rows))


def table_yang_tasks() -> None:
    """Per-task accuracy of Yang terms (Logit and J) where both folds fill the size; tasks that fill."""
    header = ["& \\multicolumn{" + str(len(SIZES)) + "}{c}{Yang terms (Logit)} & \\multicolumn{" + str(len(SIZES))
              + "}{c}{Yang terms (J)} \\\\",
              f"\\cmidrule(lr){{2-{1 + len(SIZES)}}} \\cmidrule(lr){{{2 + len(SIZES)}-{1 + 2 * len(SIZES)}}}",
              "Task & " + " & ".join(str(n) for n in SIZES) + " & " + " & ".join(str(n) for n in SIZES) + " \\\\"]
    rows = []
    for t in simple.TASKS:
        vals = [rt.num(per_task(f"yang{rd}{n}", t)) for rd in "LJ" for n in SIZES]
        rows.append(f"{rt.tex_name('simple', t)} & " + " & ".join(vals) + " \\\\")
    rows.append("\\midrule")
    fills = [str(sum(per_task(f"yang{rd}{n}", t) is not None for t in simple.TASKS)) for rd in "LJ" for n in SIZES]
    rows.append("tasks filled & " + " & ".join(fills) + " \\\\")
    write("design_yang_tasks", rt.tabular("@{}l" + "r" * 2 * len(SIZES) + "@{}", header, rows))


def table_selected(scores) -> None:
    sel = json.loads((DIR / "selected.json").read_text())
    header = ["Scope & Fold & Rank & Combination & Features & Other fold & Own fold \\\\"]
    rows = []
    for scope in SCOPES:
        for fold in (1, 2):
            other = 3 - fold
            ranked = sorted(scores[other][scope], key=lambda c: (-scores[other][scope][c], len(c)))[:5]
            for k, c in enumerate(ranked, 1):
                mark = (lambda x: f"\\textbf{{{x}}}") if c == sel[str(fold)][scope]["combo"] else (lambda x: x)
                names = ", ".join(FEATURE_NAMES[f] if f != "D" else "D" for f in c)
                rows.append(f"{scope if fold == 1 and k == 1 else ''} & {fold if k == 1 else ''} & {k} & {mark(c)} & {names} & "
                            f"{rt.num(scores[other][scope][c])} & {rt.num(scores[fold][scope][c])} \\\\")
            if not (scope == "shared" and fold == 2):
                rows.append("\\midrule" if fold == 2 else "\\addlinespace")
    write("design_selected", rt.tabular("@{}lrrl>{\\raggedright\\arraybackslash}p{5.6cm}rr@{}", header, rows))


# ------------------------------------------------------------------ figures


def axes_style(ax, ylabel=True) -> None:
    ax.set_xticks(SIZES)
    ax.set_xlim(4, 126)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("heads in the set ($N$)")
    if ylabel:
        ax.set_ylabel("accuracy")
    ax.spines[["top", "right"]].set_visible(False)


def curve(ax, name_of, kind="task", acc=None, **kw) -> None:
    xs, ys = [], []
    for n in SIZES:
        v = mean_over(name_of(n), kind, acc=acc)
        if v is not None:
            xs.append(n)
            ys.append(v)
    if xs:
        ax.plot(xs, ys, **kw)


def fig_published() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
    for ax, (scope, prefix) in zip(axes, SCOPES.items()):
        for r in ("yangL", "causal"):
            label, col, mk = RULES[r]
            curve(ax, lambda n: f"{prefix}{r}{n}", color=col, marker=mk, ms=4, label=plot_label(r))
            curve(ax, lambda n: f"{prefix}{r}{n}", "random", color=col, alpha=0.45, lw=0.9, marker=mk, ms=3.5,
                  mfc="none", label=f"{plot_label(r)}, random sets")
            curve(ax, lambda n: f"{prefix}{r}{n}", "control", color=col, alpha=0.45, lw=0.9, ls=CONTROL_STYLE,
                  label=f"{plot_label(r)}, control mean")
        ax.set_title(f"({'ab'[list(SCOPES).index(scope)]}) {'One set per task' if scope == 'per task' else 'One set for all 14 tasks'}",
                     loc="left")
        axes_style(ax, ylabel=ax is axes[0])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01), handlelength=2.2)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(rt.FIGURES / "design_published.pdf")
    plt.close(fig)


def fig_ladder() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)
    for ax, (scope, prefix) in zip(axes, SCOPES.items()):
        for r in LADDER:
            label, col, mk = RULES[r]
            curve(ax, lambda n: f"{prefix}{r}{n}", color=col, marker=mk, ms=4, label=plot_label(r),
                  lw=1.8 if r == "selected" else 1.2, zorder=3 if r == "selected" else 2)
        ax.set_title(f"({'ab'[list(SCOPES).index(scope)]}) {'One set per task' if scope == 'per task' else 'One set for all 14 tasks'}",
                     loc="left")
        axes_style(ax, ylabel=ax is axes[0])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01), handlelength=1.8,
               columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(rt.FIGURES / "design_ladder.pdf")
    plt.close(fig)


def fig_controls() -> None:
    rules = ("yangJ", "D", "selected", "causalnogen")
    fig, axes = plt.subplots(2, len(rules), figsize=(6.5, 3.6), sharex=True, sharey=True)
    for i, (scope, prefix) in enumerate(SCOPES.items()):
        for j, r in enumerate(rules):
            ax = axes[i, j]
            label, col, mk = RULES[r]
            curve(ax, lambda n: f"{prefix}{r}{n}", color=col, marker=mk, ms=3.5, label="own task means")
            curve(ax, lambda n: f"{prefix}{r}{n}", "random", color=rt.GREY, marker=mk, ms=3.5, mfc="none", lw=1.0,
                  label="random sets, same layers")
            curve(ax, lambda n: f"{prefix}{r}{n}", "control", color=rt.GREY, lw=1.0, ls=CONTROL_STYLE,
                  label="control means")
            axes_style(ax, ylabel=j == 0)
            ax.set_xticks((10, 40, 80, 120))
            if i == 0:
                ax.set_title(plot_label(r), loc="left")
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(f"{scope}\naccuracy")
    handles = [Line2D([], [], color=rt.INK_2, marker="o", ms=3.5, label="set, own task means"),
               Line2D([], [], color=rt.GREY, marker="o", mfc="none", ms=3.5, label="random sets, same layers"),
               Line2D([], [], color=rt.GREY, ls=CONTROL_STYLE, label="set, control means")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(rt.FIGURES / "design_controls.pdf")
    plt.close(fig)


def feature_effects():
    """Accuracy of each feature alone at 40 heads, and the mean change when it is added to a combination
    that lacks it (mean over tasks and folds, cross-fitted alpha)."""
    from rebuild_sets import COMBOS
    order = list(FEATURES)
    out = {}
    for scope, prefix in SCOPES.items():
        acc = {c: mean_over(f"{prefix}screen_{c}") for c in COMBOS}
        alone = {f: acc[f] for f in order}
        added = {}
        for f in order:
            d = [acc["".join(sorted(c + f, key=order.index))] - acc[c] for c in COMBOS if f not in c]
            added[f] = (float(np.mean(d)), float(np.mean([x > 0 for x in d])))
        out[scope] = (alone, added)
    return out


def fig_features(effects) -> None:
    sel = json.loads((DIR / "selected.json").read_text())
    order = list(FEATURES)
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.0), sharey=True)
    y = np.arange(len(order))[::-1]
    for j, scope in enumerate(SCOPES):
        alone, added = effects[scope]
        chosen = set(sel["1"][scope]["combo"]) | set(sel["2"][scope]["combo"])
        cols = [rt.AQUA if f in chosen else rt.GREY for f in order]
        ax = axes[0, j]
        ax.barh(y, [alone[f] for f in order], color=cols, height=0.62)
        ax.set_xlim(0, 0.5)
        ax.set_title(f"({'ab'[j]}) {scope}: feature alone, $N={SCREEN_N}$", loc="left")
        ax.set_xlabel("accuracy")
        ax = axes[1, j]
        ax.barh(y, [added[f][0] for f in order], color=cols, height=0.62)
        ax.axvline(0, color=rt.AXIS_LINE, lw=0.8)
        ax.set_xlim(-0.2, 0.32)
        ax.set_title(f"({'cd'[j]}) {scope}: feature added", loc="left")
        ax.set_xlabel("mean change in accuracy")
        for a in axes[:, j]:
            a.set_yticks(y)
            a.set_yticklabels([f"{f}  {FEATURE_NAMES[f]}" if f != "D" else "D  decodes a list word" for f in order])
            a.grid(axis="y", visible=False)
            a.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=rt.AQUA, label="feature of the selected combination"),
               plt.Rectangle((0, 0), 1, 1, color=rt.GREY, label="other feature")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(rt.FIGURES / "design_features.pdf")
    plt.close(fig)


def task_best(top_k: int = 10):
    """Per scope and task: the best of the 511 combinations at 40 heads (accuracy on the test half with
    cross-fitted alpha, mean over the two folds; ties to fewer features), the share of the task's
    ``top_k`` best combinations that contain each feature, and the accuracies of the best and of Selected.
    Descriptive: the best is chosen on the accuracies it is shown with, so it is not held out."""
    from rebuild_sets import COMBOS
    sel = json.loads((DIR / "selected.json").read_text())
    out = {}
    for scope, prefix in SCOPES.items():
        rows = {}
        for t in simple.TASKS:
            acc = {c: per_task(f"{prefix}screen_{c}", t) for c in COMBOS}
            ranked = sorted(COMBOS, key=lambda c: (-acc[c], len(c), COMBOS.index(c)))
            share = {f: np.mean([f in c for c in ranked[:top_k]]) for f in FEATURES}
            rows[t] = dict(best=ranked[0], acc=acc[ranked[0]], share=share,
                           selected=per_task(f"{prefix}selected{SCREEN_N}", t))
        out[scope] = dict(tasks=rows, selected=sel["1"][scope]["combo"], selected_folds=sel["2"][scope]["combo"])
    return out


def fig_task_best(best, top_k: int = 10) -> None:
    cmap = LinearSegmentedColormap.from_list("blue", [rt.SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    order = list(FEATURES)
    tasks = [t for ts in TYPES.values() for t in ts]
    breaks = np.cumsum([len(ts) for ts in TYPES.values()])[:-1]  # rows after which a new task type starts
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 5.0), sharey=True)
    for j, (ax, scope) in enumerate(zip(axes, SCOPES)):
        b = best[scope]
        assert b["selected"] == b["selected_folds"], "both folds must select the same combination"
        grid = np.array([[b["tasks"][t]["share"][f] for f in order] for t in tasks])
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                       extent=(-0.5, len(order) - 0.5, len(tasks) - 0.5, -0.5))
        for i, t in enumerate(tasks):
            xs = [k for k, f in enumerate(order) if f in b["tasks"][t]["best"]]
            ax.scatter(xs, [i] * len(xs), s=22, color=rt.INK, edgecolor=rt.SURFACE, lw=1.0, zorder=3)
            ax.text(len(order) - 0.1, i, f"{b['tasks'][t]['acc']:.2f}", va="center", ha="left", fontsize=6.8,
                    color=rt.INK)
            ax.text(len(order) + 1.05, i, f"{b['tasks'][t]['selected']:.2f}", va="center", ha="left",
                    fontsize=6.8, color=rt.INK_2)
        # Selected: one combination for every task, below the tasks
        ys = len(tasks) + 0.4
        xs = [k for k, f in enumerate(order) if f in b["selected"]]
        ax.scatter(xs, [ys] * len(xs), s=22, color=rt.INK, edgecolor=rt.SURFACE, lw=1.0, zorder=3, clip_on=False)
        ax.text(len(order) - 0.1, ys, b["selected"], va="center", ha="left", fontsize=6.8, color=rt.INK_2)
        for y in breaks:
            ax.axhline(y - 0.5, color=rt.SURFACE, lw=2.2)
        ax.axhline(len(tasks) - 0.5, color=rt.AXIS_LINE, lw=0.8)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_ylim(ys + 0.6, -0.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{f} {FEATURE_NAMES[f]}" if f != "D" else "D decodes a list word" for f in order],
                           rotation=60, ha="left", rotation_mode="anchor")
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_yticks(list(range(len(tasks))) + [ys])
        ax.set_yticklabels([rt.plot_name("simple", t) for t in tasks] + ["Selected"])
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        ax.spines[:].set_visible(False)
        ax.text(len(order) - 0.1, -0.85, "best", ha="left", va="bottom", fontsize=6.8, color=rt.INK)
        ax.text(len(order) + 1.05, -0.85, "Sel.", ha="left", va="bottom", fontsize=6.8, color=rt.INK_2)
        ax.set_title(f"({'ab'[j]}) {'One set per task' if scope == 'per task' else 'One set for all 14 tasks'}, "
                     f"$N={SCREEN_N}$", loc="left", pad=86)
    handles = [Line2D([], [], ls="", marker="o", ms=4.5, color=rt.INK, mec=rt.SURFACE,
                      label="feature of the task's best combination")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.0))
    cax = fig.add_axes((0.56, 0.035, 0.3, 0.016))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"share of the task's {top_k} best combinations with the feature", fontsize=7)
    cax.xaxis.set_label_position("top")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=0)
    fig.subplots_adjust(left=0.155, right=0.875, top=0.735, bottom=0.12, wspace=0.62)
    fig.savefig(rt.FIGURES / "design_task_best.pdf")
    plt.close(fig)


def fig_screen(scores) -> tuple[float, float]:
    sel = json.loads((DIR / "selected.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharex=True, sharey=True)
    corr = []
    for ax, scope in zip(axes, SCOPES):
        combos = sorted(scores[1][scope])
        x = np.array([scores[1][scope][c] for c in combos])
        y = np.array([scores[2][scope][c] for c in combos])
        has_d = np.array(["D" in c for c in combos])
        corr.append(float(np.corrcoef(x, y)[0, 1]))
        ax.plot([0, 0.7], [0, 0.7], color=rt.AXIS_LINE, lw=0.8, zorder=1)
        ax.scatter(x[~has_d], y[~has_d], s=9, color=rt.GREY, alpha=0.6, lw=0, label="without D", zorder=2)
        ax.scatter(x[has_d], y[has_d], s=9, color=rt.MAGENTA, alpha=0.6, lw=0, label="with D", zorder=2)
        for c in {sel["1"][scope]["combo"], sel["2"][scope]["combo"]}:
            ax.scatter([scores[1][scope][c]], [scores[2][scope][c]], s=46, facecolor="none", edgecolor=rt.AQUA, lw=1.6,
                       zorder=4)
            ax.annotate(f"Selected: {c}", (scores[1][scope][c], scores[2][scope][c]), xytext=(-8, 8),
                        textcoords="offset points", ha="right", fontsize=7, color=rt.INK)
        ax.set_title(f"({'ab'[list(SCOPES).index(scope)]}) {scope}, all 511 combinations at $N={SCREEN_N}$", loc="left")
        ax.set_xlabel("fold 1 (tested on half B)")
        ax.set_xlim(0, 0.7)
        ax.set_ylim(0, 0.7)
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("fold 2 (tested on half A)")
    axes[0].legend(loc="upper left", markerscale=1.6)
    fig.tight_layout()
    fig.savefig(rt.FIGURES / "design_screen.pdf")
    plt.close(fig)
    return tuple(corr)


def fig_layers(n: int = SCREEN_N) -> None:
    cmap = LinearSegmentedColormap.from_list("blue", [rt.SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 3.6), sharex=True, gridspec_kw={"hspace": 0.45})
    vmax = 0
    grids = {}
    for scope, prefix in SCOPES.items():
        g = np.full((len(LADDER), N_LAYERS), np.nan)
        for i, r in enumerate(LADDER):
            v = set_stat(r, scope, n, lambda s, nm: per_layer(s[nm]))
            if v is not None:
                g[i] = v
        grids[scope] = g
        vmax = max(vmax, np.nanmax(g))
    for ax, scope in zip(axes, SCOPES):
        g = grids[scope]
        im = ax.imshow(np.nan_to_num(g, nan=0), aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        for i in np.flatnonzero(np.isnan(g).all(1)):
            ax.text(N_LAYERS / 2, i, "does not fill the set on every task", ha="center", va="center", fontsize=6.5,
                    color=rt.INK_2)
        ax.set_yticks(range(len(LADDER)))
        ax.set_yticklabels([plot_label(r) for r in LADDER])
        ax.set_title(f"({'ab'[list(SCOPES).index(scope)]}) {scope}, $N={n}$: heads per layer"
                     + (" (mean over tasks and folds)" if scope == "per task" else " (mean over folds)"), loc="left")
        ax.grid(False)
        ax.set_xticks(range(0, N_LAYERS, 2))
    axes[1].set_xlabel("layer")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("heads")
    cb.outline.set_visible(False)
    fig.savefig(rt.FIGURES / "design_layers.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_types() -> None:
    fig, axes = plt.subplots(2, len(TYPES), figsize=(6.5, 3.4), sharex=True, sharey=True)
    for i, (scope, prefix) in enumerate(SCOPES.items()):
        for j, (tname, ts) in enumerate(TYPES.items()):
            ax = axes[i, j]
            for r in LADDER:
                label, col, mk = RULES[r]
                xs, ys = [], []
                for n in SIZES:
                    v = mean_over(f"{prefix}{r}{n}", tasks=ts)
                    if v is not None:
                        xs.append(n)
                        ys.append(v)
                ax.plot(xs, ys, color=col, marker=mk, ms=2.8, lw=1.0, label=plot_label(r))
            ax.set_xticks((10, 60, 120))
            ax.set_ylim(0, 1)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(tname, loc="left")
            if j == 0:
                ax.set_ylabel(f"{scope}\naccuracy")
            if i == 1:
                ax.set_xlabel("$N$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01), handlelength=1.6,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(rt.FIGURES / "design_types.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ stage 3 (Sec. 6)

S3_SCENARIO = {"wt": "Unseen wordings, transfer", "sum": "Summed vector"}
# Wording groups (Sec. 6.1): Yang et al.'s cloze template and four question-answer templates (Sec. 5), and the
# six groups of Q wordings, ordered by form.
GROUP_FORMS = (("cloze", ("yang-cloze", "cloze-named", "cloze-unnamed")),
               ("question–answer", ("yang-qa", "qa-named", "qa-unnamed")),
               ("label", ("label-named", "label-unnamed")))
GROUP_LABEL = {"yang-cloze": "Yang's template (Sec. 5)", "yang-qa": "Yang's four templates (Sec. 5)",
               "cloze-named": "names the task", "cloze-unnamed": "does not", "qa-named": "names the task",
               "qa-unnamed": "does not", "label-named": "names the task", "label-unnamed": "does not"}
GROUPS = tuple(g for _, gs in GROUP_FORMS for g in gs)


def template_rows(folder: Path, parts: str, report: list[Path], n: int, skip: str | None = None) -> pd.DataFrame:
    """Task-mean rows (one per template mean) of the S3 rules at ``n`` heads, at each fold and task's
    cross-fitted alpha (from the report). Rows repeated by a later run of the same set are dropped."""
    names = rf"^(shared_)?({'|'.join(S3_RULES)}){n}$"
    raw = pd.concat([pd.read_csv(p).assign(fold=f) for f in (1, 2) for p in sorted((folder / f"fold{f}").glob(parts))
                     if skip is None or skip not in p.name], ignore_index=True)
    raw = raw[(raw.source == "task") & raw.heads.str.match(names)]
    alpha = pd.concat([pd.read_csv(p) for p in report])
    alpha = alpha[alpha.source == "task"][["fold", "task", "set", "alpha"]].rename(columns={"set": "heads"}).drop_duplicates()
    x = raw.merge(alpha, on=["fold", "task", "heads", "alpha"])
    return x.drop_duplicates(["fold", "task", "heads", "alpha", "mean_wording"])


def by_group(n: int = SCREEN_N) -> tuple[dict[str, dict[str, float]], list[str]]:
    """acc[set][group]: mean over the group's wordings, then over tasks and folds, on the tasks that have a mean
    of every group in both folds (a Q wording gives no mean without 10 correct prompts); and the tasks left out."""
    s5 = template_rows(DIR, "patching_simple.part-s*.csv", [DIR / f"{s}_per_task.csv" for s in ("s1", "s2")], n, skip="s1sum")
    s5["group"] = np.where(s5.mean_wording == "inst0", "yang-cloze", "yang-qa")
    d, st = S3["wt"]
    q = template_rows(d, f"patching_simple.part-{st}-*.csv", [d / f"{st}_per_task.csv"], n, skip="sel40")
    q["group"] = q.mean_wording.map(simple.wording_group)
    x = pd.concat([s5, q])
    cell = x.groupby(["heads", "group", "fold", "task"]).accuracy.mean()
    have = cell.groupby(["fold", "task"]).apply(lambda c: c.index.get_level_values("group").nunique())
    left_out = sorted({t for (f, t), k in have.items() if k < len(GROUPS)})
    cell = cell[~cell.index.get_level_values("task").isin(left_out)]
    n_tasks = len(simple.TASKS) - len(left_out)
    g = cell.groupby(["heads", "group"])
    full = g.size() == 2 * n_tasks
    return {h: {grp: (float(v) if full[(h, grp)] else None) for (h2, grp), v in g.mean().items() if h2 == h}
            for h in x.heads.unique()}, left_out


def table_s3_acc(key: str, name: str) -> None:
    """Task-mean accuracy of the S3 rules at every size in one test, both scopes (as Table design_ladder)."""
    acc = ACC3[key]
    header = ["$N$ & " + " & ".join(RULES[r][0] for r in S3_RULES) + " \\\\"]
    rows = []
    for scope, prefix in SCOPES.items():
        rows.append(f"\\multicolumn{{{len(S3_RULES) + 1}}}{{@{{}}l}}{{\\textit{{{scope.capitalize()}}}}} \\\\")
        for n in SIZES:
            vals = [mean_over(f"{prefix}{r}{n}", acc=acc) for r in S3_RULES]
            best = max(v for v in vals if v is not None)
            rows.append(f"{n} & " + " & ".join(("\\textbf{" + rt.num(v) + "}") if v is not None and v == best else rt.num(v)
                                              for v in vals) + " \\\\")
        if scope == "per task":
            rows.append("\\midrule")
    write(name, rt.tabular("@{}r" + "r" * len(S3_RULES) + "@{}", header, rows))


def short_label(rule: str) -> str:
    """Header of a rule spanning two narrow columns."""
    return "Causal $-$ gen." if rule == "causalnogen" else RULES[rule][0]


def table_summed() -> None:
    """Head by head (Sec. 5) against the summed vector, same sets and template means, every size."""
    header = ["& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{short_label(r)}}}" for r in S3_RULES) + " \\\\",
              " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(S3_RULES))),
              "$N$ & " + " & ".join("heads & sum" for _ in S3_RULES) + " \\\\"]
    rows = []
    for scope, prefix in SCOPES.items():
        rows.append(f"\\multicolumn{{{2 * len(S3_RULES) + 1}}}{{@{{}}l}}{{\\textit{{{scope.capitalize()}}}}} \\\\")
        for n in SIZES:
            vals = []
            for r in S3_RULES:
                vals += [rt.num(mean_over(f"{prefix}{r}{n}")), rt.num(mean_over(f"{prefix}{r}{n}", acc=ACC3["sum"]))]
            rows.append(f"{n} & " + " & ".join(vals) + " \\\\")
        if scope == "per task":
            rows.append("\\midrule")
    write("design_summed", rt.tabular("@{}r" + "rr" * len(S3_RULES) + "@{}", header, rows))


def table_s3_controls() -> None:
    """Random sets and control means of the S3 rules in every stage-3 test, every size."""
    header = ["& & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{short_label(r)}}}" for r in S3_RULES) + " \\\\",
              " ".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(S3_RULES))),
              "Scope & $N$ & " + " & ".join("rand. & ctrl." for _ in S3_RULES) + " \\\\"]
    rows = []
    keys = [k for k in S3_SCENARIO if k in ACC3]
    for i, key in enumerate(keys):
        rows.append(f"\\multicolumn{{{2 * len(S3_RULES) + 2}}}{{@{{}}l}}{{\\textit{{{S3_SCENARIO[key]}}}}} \\\\")
        for scope, prefix in SCOPES.items():
            for n in SIZES:
                vals = []
                for r in S3_RULES:
                    vals += [rt.num(mean_over(f"{prefix}{r}{n}", "random", acc=ACC3[key])),
                             rt.num(mean_over(f"{prefix}{r}{n}", "control", acc=ACC3[key]))]
                rows.append(f"{scope if n == SIZES[0] else ''} & {n} & " + " & ".join(vals) + " \\\\")
            if scope == "per task":
                rows.append("\\addlinespace")
        if i < len(keys) - 1:
            rows.append("\\midrule")
    write("design_s3_controls", rt.tabular("@{}lr" + "rr" * len(S3_RULES) + "@{}", header, rows))


def table_s3_types(key: str, name: str, n: int = SCREEN_N) -> None:
    header = ["Rule & " + " & ".join(TYPES) + " & all \\\\"]
    rows = []
    for scope, prefix in SCOPES.items():
        rows.append(f"\\multicolumn{{{len(TYPES) + 2}}}{{@{{}}l}}{{\\textit{{{scope.capitalize()}}}}} \\\\")
        for r in S3_RULES:
            nm = f"{prefix}{r}{n}"
            rows.append(f"{RULES[r][0]} & " + " & ".join(rt.num(mean_over(nm, tasks=ts, acc=ACC3[key])) for ts in TYPES.values())
                        + f" & {rt.num(mean_over(nm, acc=ACC3[key]))} \\\\")
        if scope == "per task":
            rows.append("\\midrule")
    write(name, rt.tabular("@{}l" + "r" * (len(TYPES) + 1) + "@{}", header, rows))


def table_groups(groups, n: int = SCREEN_N) -> None:
    header = ["& & " + " & ".join(f"\\multicolumn{{{len(gs)}}}{{c}}{{{form.replace('–', '--')}}}" for form, gs in GROUP_FORMS) + " \\\\",
              " ".join(f"\\cmidrule(lr){{{3 + a}-{2 + a + len(gs)}}}"
                       for a, (_, gs) in zip(np.cumsum([0] + [len(gs) for _, gs in GROUP_FORMS]), GROUP_FORMS)),
              "Scope & Rule & " + " & ".join({"yang-cloze": "Yang", "yang-qa": "Yang"}.get(g, "named" if g.endswith("-named") else "not")
                                             for g in GROUPS) + " \\\\"]
    rows = []
    for scope, prefix in SCOPES.items():
        for r in S3_RULES:
            v = groups.get(f"{prefix}{r}{n}", {})
            rows.append(f"{scope if r == S3_RULES[0] else ''} & {RULES[r][0]} & " + " & ".join(rt.num(v.get(g)) for g in GROUPS) + " \\\\")
        if scope == "per task":
            rows.append("\\midrule")
    write("design_wordings_groups", rt.tabular("@{}ll" + "r" * len(GROUPS) + "@{}", header, rows))


def panel_title(scope: str) -> str:
    return f"({'ab'[list(SCOPES).index(scope)]}) {'One set per task' if scope == 'per task' else 'One set for all 14 tasks'}"


def fig_wordings_transfer() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
    for ax, (scope, prefix) in zip(axes, SCOPES.items()):
        for r in S3_RULES:
            label, col, mk = RULES[r]
            curve(ax, lambda n: f"{prefix}{r}{n}", color=col, lw=0.9, alpha=0.4, zorder=1)
            curve(ax, lambda n: f"{prefix}{r}{n}", acc=ACC3["wt"], color=col, marker=mk, ms=4, label=plot_label(r),
                  lw=1.8 if r == "selected" else 1.2, zorder=3 if r == "selected" else 2)
        ax.set_title(panel_title(scope), loc="left")
        axes_style(ax, ylabel=ax is axes[0])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.055), handlelength=1.8, columnspacing=1.2)
    fig.legend([Line2D([], [], color=rt.INK_2, marker="o", ms=3.5, lw=1.2), Line2D([], [], color=rt.INK_2, lw=0.9, alpha=0.4)],
               ["means of the 15 unseen Q wordings", "means of Yang's five templates (Sec. 5)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01), handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(rt.FIGURES / "design_wordings_transfer.pdf")
    plt.close(fig)


def fig_groups(groups, n: int = SCREEN_N) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.3), sharey=True)
    ys, labels, y = [], [], 0.0
    form_y = []
    for form, gs in GROUP_FORMS:
        form_y.append((form, y))
        for g in gs:
            ys.append(y)
            labels.append(GROUP_LABEL[g])
            y += 1
        y += 0.6
    ypos = dict(zip(GROUPS, ys))
    for ax, (scope, prefix) in zip(axes, SCOPES.items()):
        for i, r in enumerate(S3_RULES):
            label, col, mk = RULES[r]
            v = groups.get(f"{prefix}{r}{n}", {})
            pts = [(v[g], ypos[g] + (i - 2) * 0.13) for g in GROUPS if v.get(g) is not None]
            if pts:
                ax.scatter(*zip(*pts), color=col, marker=mk, s=16, zorder=3, label=plot_label(r), lw=0)
        for g in ("yang-cloze", "yang-qa"):
            ax.axhspan(ypos[g] - 0.45, ypos[g] + 0.45, color=rt.GRIDLINE, alpha=0.6, lw=0, zorder=0)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        ax.set_ylim(ys[-1] + 0.6, -0.6 - 0.9)
        for form, fy in form_y if ax is axes[0] else ():
            ax.text(-0.01, fy - 0.75, form, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=7.5, color=rt.INK, fontweight="bold")
        ax.set_xlim(0, 0.75)
        ax.set_xlabel("accuracy")
        ax.grid(axis="y", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(panel_title(scope) + f", $N={n}$", loc="left")
    handles, hl = axes[1].get_legend_handles_labels()
    fig.legend(handles, hl, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(rt.FIGURES / "design_wordings_groups.pdf")
    plt.close(fig)


def fig_summed() -> None:
    fig, axes = plt.subplots(2, len(S3_RULES), figsize=(6.5, 3.6), sharex=True, sharey=True)
    for i, (scope, prefix) in enumerate(SCOPES.items()):
        for j, r in enumerate(S3_RULES):
            ax = axes[i, j]
            label, col, mk = RULES[r]
            curve(ax, lambda n: f"{prefix}{r}{n}", color=col, marker=mk, ms=3.5)
            curve(ax, lambda n: f"{prefix}{r}{n}", acc=ACC3["sum"], color=col, marker=mk, ms=3.5, mfc="none", ls=(0, (4, 2)),
                  lw=1.0)
            axes_style(ax, ylabel=j == 0)
            ax.set_xticks((10, 40, 80, 120))
            ax.set_xlabel("" if i == 0 else "heads ($N$)")
            if i == 0:
                ax.set_title(plot_label(r), loc="left")
            if j == 0:
                ax.set_ylabel(f"{scope}\naccuracy")
            if r == "yangJ" and scope == "per task":
                ax.text(0.5, 0.55, "fills 10 and 20\nheads only", transform=ax.transAxes, ha="center", fontsize=6.5,
                        color=rt.INK_2)
    handles = [Line2D([], [], color=rt.INK_2, marker="o", ms=3.5, label="head by head"),
               Line2D([], [], color=rt.INK_2, marker="o", mfc="none", ms=3.5, ls=(0, (4, 2)), lw=1.0,
                      label="summed after layer 11")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(rt.FIGURES / "design_summed.pdf")
    plt.close(fig)


def alpha2_share() -> dict[str, float]:
    """Share of task-folds whose cross-fitted alpha is 2, over the S3 rules' sets (task means)."""
    out = {}
    for key, paths in {"Sec. 5": [DIR / f"{s}_per_task.csv" for s in ("s1", "s2")],
                       **{k: [d / f"{st}_per_task.csv"] for k, (d, st) in S3.items()}}.items():
        if not all(p.exists() for p in paths):
            continue
        pt = pd.concat([pd.read_csv(p) for p in paths])
        q = pt[(pt.source == "task") & pt.set.str.match(rf"^(shared_)?({'|'.join(S3_RULES)})\d+$")].drop_duplicates(["fold", "task", "set"])
        out[key] = round(float((q.alpha == 2).mean()), 2)
    return out


def stage3() -> None:
    if "wt" in ACC3:
        table_s3_acc("wt", "design_wordings_transfer")
        table_s3_types("wt", "design_wordings_types")
        groups, left_out = by_group()
        print("wording groups: tasks left out", left_out)
        table_groups(groups)
        fig_wordings_transfer()
        fig_groups(groups)
    if "sum" in ACC3:
        table_summed()
        table_s3_types("sum", "design_summed_types")
        fig_summed()
    if ACC3:
        table_s3_controls()
        print("alpha = 2 share:", alpha2_share())


def main() -> None:
    scores = {fold: screen_scores(fold) for fold in (1, 2)}
    table_own()
    table_ladder()
    table_controls()
    table_types()
    table_overlap()
    table_yang_tasks()
    table_selected(scores)
    fig_published()
    fig_ladder()
    fig_controls()
    effects = feature_effects()
    fig_features(effects)
    corr = fig_screen(scores)
    fig_layers()
    fig_types()
    best = task_best()
    fig_task_best(best)
    stage3()
    for scope, b in best.items():
        print(scope, "best per task:", {t: (r["best"], round(r["acc"], 3)) for t, r in b["tasks"].items()})
    print("screen fold-1 vs fold-2 correlation: per task %.2f, shared %.2f" % corr)
    for scope, (alone, added) in effects.items():
        print(scope, "alone:", {f: round(v, 3) for f, v in alone.items()})
        print(scope, "added:", {f: (round(m, 3), round(s, 2)) for f, (m, s) in added.items()})
    gen = {f: json.loads((DIR / f"fold{f}" / "generic.json").read_text())["generic"] for f in (1, 2)}
    print("generic heads:", gen, "common:", sorted(set(gen[1]) & set(gen[2])))


if __name__ == "__main__":
    main()
