"""Rebuilt design (paper/head-sets-simple-design.tex, Secs. 4-6): figures and tables of stages 1-3.

Reads the cross-fitted accuracies of ``scripts/rebuild_report.py`` (``results/rebuild/additions/
s1_per_task.csv``, ``s2_per_task.csv``), the sets (``fold{1,2}/heads.json``), the selection
(``selected.json``, and the screen scores of ``scripts/rebuild_select.py``) and the generic heads
(``fold{1,2}/generic.json``). A mean over tasks is given only where every task and both folds have the
set (``--`` otherwise); random sets are averaged over their three draws first. Stage 3 (Sec. 6) reads the
same reports of its tests under ``results/rebuild/scenarios/`` and, for the wording groups, the patching
files themselves (each fold and task at its cross-fitted alpha). Writes
``paper/figures/design_*.pdf`` and ``paper/tables/design_*.tex``.

With a two-stage screen (``scripts/rebuild_quick_screen.sh``), only the shortlist is patched in full: the
analyses over all combinations (each feature's contribution, fold 1 against fold 2, the best combination of
each task) then read the quick pass (``scripts/rebuild_shortlist.py``: first-token accuracy at alpha 2 on 10
test items per task, mean over the template means) instead of the cross-fitted accuracies; the tables of the
five best combinations rank each fold's shortlist.

    .venv/bin/python scripts/rebuild_figures.py
"""

from __future__ import annotations

import json
import re
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
from rebuild_shortlist import has_quick, quick_per_task, quick_scores  # noqa: E402
from rebuild_sets import FEATURES, KINDS, SCREEN_N, SIZES, rankdata  # noqa: E402

DIR = rt.RESULTS / "rebuild" / "additions"
N_HEADS, N_LAYERS = 32, 32
TYPES = {"word relations": ("antonym", "synonym"), "morphology": ("singular-plural", "present-past"),
         "facts": ("country-capital", "country-currency", "landmark-country", "park-country", "product-company"),
         "sequences": ("next_item", "prev_item"), "translations": ("english-french", "english-german", "english-spanish")}
# Rule -> (label, colour, marker). Colours: the validated categorical order (dataviz validator, light:
# all adjacent pairs pass), lexical blue and causal orange as in the project's earlier figures.
# Plum for T / LRP-T: against each of the six colours above, CVD separation >= 12.8 and normal-vision >= 23 (dataviz
# validator, light surface); the categorical slots 7-8 (violet, red) fail against blue and orange.
PLUM = "#9b2d6f"
RULES = {"yangL": ("Yang terms (Logit)", rt.BLUE, "o"), "causal": ("Causal", rt.ORANGE, "v"),
         "selected": ("Selected", rt.AQUA, "D"), "yangJ": ("Yang terms (J)", rt.YELLOW, "s"),
         "D": ("D", rt.MAGENTA, "^"), "causalnogen": ("Causal $-$ generic", rt.GREEN, "P"),
         "causalzs": ("Causal (zero-shot)", rt.ORANGE, "v"),
         "causalzsnogen": ("Causal (zero-shot) $-$ generic", rt.GREEN, "P"),
         "T": ("T", PLUM, "X"), "lrpT": ("LRP-T", PLUM, "X")}
LADDER = ("yangL", "yangJ", "D", "T", "lrpT", "selected", "causal", "causalnogen", "causalzs", "causalzsnogen")
# Step 6 (Sec. 5.6): the causal score measured in the zero-shot prompt; drawn dashed, in its counterpart's colour.
ZERO_SHOT = ("causalzs", "causalzsnogen")
# Step 7 (Sec. 5.7): attention to the task words weighted by its AttnLRP relevance (LRP-T, solid) and plain (T, the
# step-3 rule of that segment, dashed), one hue.
DASHED = ZERO_SHOT + ("T",)
# Two-level table headers: rule -> (group, column); consecutive rules of a group share a spanning label.
HEADER = {"yangL": ("Yang terms", "Logit"), "yangJ": ("Yang terms", "J"), "D": ("", "D"), "selected": ("", "Selected"),
          "causal": ("Causal", "as ranked"), "causalnogen": ("Causal", "$-$ generic"),
          "causalzs": ("Causal (zero-shot)", "as ranked"), "causalzsnogen": ("Causal (zero-shot)", "$-$ generic"),
          "T": ("Task words", "T"), "lrpT": ("Task words", "LRP-T")}


def line_style(rule: str) -> str:
    return "--" if rule in DASHED else "-"


def group_header(rules, lead: int, width: int = 1) -> list[str]:
    """Header rows of a table whose first ``lead`` columns are labels and whose rules take ``width`` columns each:
    the group labels spanning their rules (with rules under them), then the column labels."""
    top, rules_, col = [], [], lead + 1
    i = 0
    while i < len(rules):
        g = HEADER[rules[i]][0]
        j = i
        while j + 1 < len(rules) and g and HEADER[rules[j + 1]][0] == g:
            j += 1
        span = (j - i + 1) * width
        top.append(f"\\multicolumn{{{span}}}{{c}}{{{g}}}" if g else (" & ".join([""] * span) if span > 1 else ""))
        if g:
            rules_.append(f"\\cmidrule(lr){{{col}-{col + span - 1}}}")
        col += span
        i = j + 1
    return [" & " * (lead - 1) + " & " + " & ".join(top) + " \\\\", " ".join(rules_)]
SCOPES = {"per task": "", "shared": "shared_"}
FEATURE_NAMES = {c: name for c, (name, _) in FEATURES.items()}
CONTROL_STYLE = rt.CONTROL_STYLE
QUICK = has_quick()  # the screen ran in two stages: the analyses over all combinations read the quick pass
SCREEN_METRIC = "first-token accuracy, quick pass" if QUICK else "accuracy"


def plot_label(rule: str) -> str:
    return RULES[rule][0].replace("$-$", "−")


# ------------------------------------------------------------------ data


def cells(src) -> pd.Series:
    """accuracy[(set, kind, fold, task)]; kind task / control / random (random: mean of the draws). ``src``: the
    per-task report files, or their rows."""
    pt = src if isinstance(src, pd.DataFrame) else pd.concat([pd.read_csv(p) for p in src], ignore_index=True)
    is_random = pt.set.str.match(r"^random_.+_\d+$")
    pt["base"] = pt.set.where(~is_random, pt.set.str.replace(r"^random_(.+)_\d+$", r"\1", regex=True))
    pt["kind"] = np.where(is_random, "random", pt.source)
    return pt.groupby(["base", "kind", "fold", "task"]).accuracy.mean()


# Shared Selected (user 2026-11-23): at a size with its own shared screen (scripts/rebuild_screen_n.py), each
# fold's pick at that size; at 40 heads the stage-2 pick, which a size without a screen uses too. Per task: the
# 40-head pick at every size. The same in stage 3: the transfer tests patch every size's stage-2 pick
# (results/rebuild/transfer_n), and the re-identified tests choose at each size on their pool
# (results/rebuild/{wordings,examples}_n<N>). Their rows and sets replace those of the 40-head choice.
R = rt.RESULTS / "rebuild"
SIZE_SCREEN = {n: R / f"shared_n{n}" for n in SIZES if (R / f"shared_n{n}" / "sel_per_task.csv").exists()}
SIZED = re.compile(r"^(random_)?shared_selected(\d+)(_\d+)?$")


def sized(name: str) -> int | None:
    m = SIZED.match(name)
    return int(m.group(2)) if m else None


def with_sized(pt: pd.DataFrame, files) -> pd.DataFrame:
    """``pt`` with the shared Selected rows of every size that ``files`` report replaced by theirs."""
    new = [pd.read_csv(p) for p in files if p.exists()]
    if not new:
        return pt
    new = pd.concat(new, ignore_index=True)
    sizes = set(new.set.map(sized))
    assert None not in sizes, "only shared Selected rows"
    return pd.concat([pt[~pt.set.map(sized).isin(sizes)], new], ignore_index=True)


# Stages: s1 (Sec. 4), s2 (Sec. 5), zs (Causal (zero-shot), Sec. 5.6; scripts/rebuild_causal_zs.sh), lrp (T and
# LRP-T, Sec. 5.7; scripts/rebuild_lrp.sh).
ACC = cells(with_sized(pd.concat([pd.read_csv(DIR / f"{s}_per_task.csv") for s in ("s1", "s2", "zs", "lrp")
                                  if (DIR / f"{s}_per_task.csv").exists()], ignore_index=True),
                       [d / "sel_per_task.csv" for d in SIZE_SCREEN.values()]))
# Stage 3 (Sec. 6): test -> (results folder, report stage); a test enters once scripts/rebuild_report.py has run.
SCEN = rt.RESULTS / "rebuild" / "scenarios"
WPOOL, EPOOL = rt.RESULTS / "rebuild" / "wordings", rt.RESULTS / "rebuild" / "examples"  # re-identified (Secs. 6.1-6.2)
S3 = {"wt": (SCEN / "wordings_transfer", "s3wt"), "wq": (WPOOL, "s3w-test"), "sum": (SCEN / "summed", "s3sum"),
      "et": (SCEN / "examples_transfer", "s3et"), "eq": (EPOOL, "s3e-test"), "esum": (EPOOL, "s3e-sum")}
SIZED_S3 = {"wt": [R / "transfer_n" / "wt_per_task.csv"], "et": [R / "transfer_n" / "et_per_task.csv"],
            "sum": [R / "transfer_n" / "tsum_per_task.csv"],
            "wq": [R / f"wordings_n{n}" / "test_per_task.csv" for n in SIZES],
            "eq": [R / f"examples_n{n}" / "test_per_task.csv" for n in SIZES],
            "esum": [R / f"examples_n{n}" / "sum_per_task.csv" for n in SIZES]}
ACC3 = {k: cells(with_sized(pd.read_csv(d / f"{st}_per_task.csv"), SIZED_S3[k]))
        for k, (d, st) in S3.items() if (d / f"{st}_per_task.csv").exists()}
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


def heads(base: Path = DIR) -> dict[int, dict[str, dict[str, set]]]:
    """sets[fold][task][name] = set of (layer, head)."""
    out = {}
    for fold in (1, 2):
        h = json.loads((base / f"fold{fold}" / "heads.json").read_text())["simple"]
        out[fold] = {t: {n: set(map(tuple, hs)) for _, n, hs in h[t]["sets"]} for t in simple.TASKS}
    return out


def with_sized_sets(sets, prefix: str):
    """``sets`` with the shared Selected set and its random sets of every size chosen in ``results/rebuild/
    <prefix><N>`` in place of the 40-head choice's."""
    for n in SIZES:
        d = R / f"{prefix}{n}"
        if not (d / "selected.json").exists():
            continue
        for f, by_task in heads(d).items():
            for t, s in by_task.items():
                new = {k: v for k, v in s.items() if sized(k) == n}
                assert len(new) == 4 and all(k in sets[f][t] for k in new), (prefix, n, f, t)
                sets[f][t].update(new)
    return sets


SETS = with_sized_sets(heads(), "shared_n")


def per_layer(s: set) -> np.ndarray:
    c = np.zeros(N_LAYERS)
    for l, _ in s:
        c[l] += 1
    return c


def chance_overlap(a: set, b: set) -> float:
    """Expected shared heads of two random sets with these numbers of heads per layer."""
    return float((per_layer(a) * per_layer(b)).sum() / N_HEADS)


def set_stat(rule: str, scope: str, n: int, fn, sets_of=None):
    """Mean of fn(sets of the task-fold) over tasks and folds (shared: the one set per fold); fn may
    return a number or an array. ``sets_of(fold, task)``: the sets fn reads (default: those of Sec. 5)."""
    name = f"{SCOPES[scope]}{rule}{n}"
    vals = []
    tasks = simple.TASKS[:1] if scope == "shared" else simple.TASKS
    for fold in (1, 2):
        for t in tasks:
            sets = SETS[fold][t] if sets_of is None else sets_of(fold, t)
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
    if mean_over("shared_davidson_zs") is not None:  # the same, ranked by the zero-shot score (Sec. 5.6)
        rows.append(
            f"\\quad zero-shot score (Sec.~\\ref{{sec:causal-zs}}) & shared & 20 & summed after layer 11, "
            f"$\\alpha=1$ & {rt.num(mean_over('shared_davidson_zs'))} & {rt.num(mean_over('shared_davidson_zs', 'control'))} & "
            f"{rt.num(mean_over('shared_davidson_zs', 'random'))} \\\\")
    write("design_own", rt.tabular("@{}llllrrr@{}", ["Method & Scope & Heads & Intervention & Task & Control & Random \\\\"],
                                   rows))


def table_ladder() -> None:
    """Task-mean accuracy of every rule at every size, both scopes."""
    header = group_header(LADDER, 1) + ["$N$ & " + " & ".join(HEADER[r][1] for r in LADDER) + " \\\\"]
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
    """Random sets and control means of every rule at every size; the rules in two blocks of four, one under the
    other, so that the table fits the page."""
    half = (len(LADDER) + 1) // 2

    def block_header(rules) -> list[str]:
        return ["& & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{RULES[r][0]}}}" for r in rules) + " \\\\",
                " ".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(rules))),
                "Scope & $N$ & " + " & ".join("rand. & ctrl." for _ in rules) + " \\\\"]

    rows = []
    for b, rules in enumerate((LADDER[:half], LADDER[half:])):
        if b:
            rows += ["\\midrule[\\heavyrulewidth]", *block_header(rules), "\\midrule"]
        for scope, prefix in SCOPES.items():
            for n in SIZES:
                vals = []
                for r in rules:
                    vals += [rt.num(mean_over(f"{prefix}{r}{n}", "random")), rt.num(mean_over(f"{prefix}{r}{n}", "control"))]
                rows.append(f"{scope if n == SIZES[0] else ''} & {n} & " + " & ".join(vals) + " \\\\")
            if scope == "per task":
                rows.append("\\addlinespace")
    write("design_controls", rt.tabular("@{}lr" + "rr" * half + "@{}", block_header(LADDER[:half]), rows))


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


CONTEXTS = {"equiprobable": "Equiprobable sequences", "wikitext": "WikiText-103 text",
            "other_task": "Other tasks' instructions", "causal": "Causal: mean over the three",
            "frame": "Frame only, \\p{ Q: x\\nl A:}", "zero": "Zero-shot prompt \\p{ x:}"}


def context_score(fv, f: int, t: str, c: str) -> np.ndarray:
    cie, ctx = fv[f]
    if c == "causal":
        return np.mean([cie[f"{t}/{k}"] for k in KINDS], axis=0).ravel()
    return (cie if c in KINDS else ctx)[f"{t}/{c}"].ravel()


def table_causal_kinds() -> None:
    """Sec. 5.6: the share of the top N heads by the causal score that the two halves of the items have in common,
    by the prompt the score is measured in (fold 1's score on half A against fold 2's on half B, both directions
    for two different prompts), and the size of the scores (mean of the 20 largest)."""
    root = rt.RESULTS / "heldout"
    if not all((root / f"fold{f}" / "fv" / "cie_context.npz").exists() for f in (1, 2)):
        return
    fv = {f: (np.load(root / f"fold{f}" / "fv" / "cie.npz"), np.load(root / f"fold{f}" / "fv" / "cie_context.npz"))
          for f in (1, 2)}
    sc = {(f, t, c): context_score(fv, f, t, c) for f in (1, 2) for t in simple.TASKS for c in CONTEXTS}

    def tops(f, c, n, shared):
        if shared:
            r = np.mean([rankdata(-sc[(f, t, c)]) for t in simple.TASKS], axis=0)
            return {None: set(np.argsort(r, kind="stable")[:n])}
        return {t: set(np.argsort(-sc[(f, t, c)], kind="stable")[:n]) for t in simple.TASKS}

    def share(pairs, n, shared):
        vals = []
        for a, b in pairs:
            ta, tb = tops(1, a, n, shared), tops(2, b, n, shared)
            vals += [len(ta[k] & tb[k]) / n for k in ta]
        return float(np.mean(vals))

    cols = [(False, 20), (False, 40), (True, 20), (True, 40)]
    header = ["& \\multicolumn{2}{c}{Per task} & \\multicolumn{2}{c}{Shared} & \\\\",
              "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
              "Prompt the score is measured in & $N=20$ & $N=40$ & $N=20$ & $N=40$ & Top-20 effect \\\\"]
    rows = ["\\multicolumn{6}{@{}l}{\\textit{Same prompt on both halves}} \\\\"]
    for c, label in CONTEXTS.items():
        eff = np.mean([np.sort(sc[(f, t, c)])[-20:].mean() for f in (1, 2) for t in simple.TASKS])
        rows.append(f"{label} & " + " & ".join(rt.num(share([(c, c)], n, sh)) for sh, n in cols) + f" & {eff:.4f} \\\\")
        if c == "other_task":
            rows.append("\\addlinespace")
    rows += ["\\midrule", "\\multicolumn{6}{@{}l}{\\textit{Different prompts on the two halves}} \\\\"]
    pairs = {"Equiprobable -- WikiText": [("equiprobable", "wikitext")],
             "Other tasks' instructions -- equiprobable or WikiText": [("other_task", "equiprobable"), ("other_task", "wikitext")],
             "Frame only -- Causal": [("frame", "causal")], "Zero-shot -- Causal": [("zero", "causal")],
             "Zero-shot -- frame only": [("zero", "frame")]}
    for label, ps in pairs.items():
        both = ps + [(b, a) for a, b in ps]
        rows.append(f"{label} & " + " & ".join(rt.num(share(both, n, sh)) for sh, n in cols) + " & \\\\")
    write("design_causal_kinds", rt.tabular("@{}lrrrrr@{}", header, rows))


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
            pool = sel[str(fold)][scope].get("candidates") or scores[other][scope]  # the shortlist of a two-stage screen
            ranked = sorted(pool, key=lambda c: (-scores[other][scope][c], len(c)))[:5]
            for k, c in enumerate(ranked, 1):
                mark = (lambda x: f"\\textbf{{{x}}}") if c == sel[str(fold)][scope]["combo"] else (lambda x: x)
                names = ", ".join(FEATURE_NAMES[f] if f != "D" else "D" for f in c)
                rows.append(f"{scope if fold == 1 and k == 1 else ''} & {fold if k == 1 else ''} & {k} & {mark(c)} & {names} & "
                            f"{rt.num(scores[other][scope][c])} & {rt.num(scores[fold][scope][c])} \\\\")
            if not (scope == "shared" and fold == 2):
                rows.append("\\midrule" if fold == 2 else "\\addlinespace")
    write("design_selected", rt.tabular("@{}lrrl>{\\raggedright\\arraybackslash}p{5.6cm}rr@{}", header, rows))


def table_selected_sizes() -> None:
    """Shared Selected at every size: each fold's combination, its accuracy on its own test half (mean over the
    14 tasks), the mean of the folds, its control means and random sets."""
    sel40 = json.loads((DIR / "selected.json").read_text())
    header = ["$N$ & Fold 1 & Acc. & Fold 2 & Acc. & Task & Control & Random \\\\"]
    rows = []
    for n in SIZES:
        name = f"shared_selected{n}"
        if n in SIZE_SCREEN:
            sel = json.loads((SIZE_SCREEN[n] / "selected.json").read_text())
            combos = [sel[str(f)]["combo"] for f in (1, 2)]
        else:
            combos = [sel40[str(f)]["shared"]["combo"] + ("" if n == SCREEN_N else "$^\\dagger$") for f in (1, 2)]
        acc = [float(ACC.loc[(name, "task", f)].mean()) for f in (1, 2)]
        rows.append(f"{n} & {combos[0]} & {rt.num(acc[0])} & {combos[1]} & {rt.num(acc[1])} & {rt.num(mean_over(name))} & "
                    f"{rt.num(mean_over(name, 'control'))} & {rt.num(mean_over(name, 'random'))} \\\\")
    write("design_selected_sizes", rt.tabular("@{}rlrlrrrr@{}", header, rows))


# ------------------------------------------------------------------ figures


def axes_style(ax, ylabel=True, ymax: float = 0.75) -> None:
    ax.set_xticks(SIZES)
    ax.set_xlim(4, 126)
    ax.set_ylim(0, ymax)
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
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.95), sharey=True)
    for ax, (scope, prefix) in zip(axes, SCOPES.items()):
        for r in LADDER:
            label, col, mk = RULES[r]
            curve(ax, lambda n: f"{prefix}{r}{n}", color=col, marker=mk, ms=4, label=plot_label(r), ls=line_style(r),
                  lw=1.8 if r == "selected" else 1.2, zorder=3 if r == "selected" else 2)
        ax.set_title(f"({'ab'[list(SCOPES).index(scope)]}) {'One set per task' if scope == 'per task' else 'One set for all 14 tasks'}",
                     loc="left")
        axes_style(ax, ylabel=ax is axes[0])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), handlelength=2.4,
               columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(rt.FIGURES / "design_ladder.pdf")
    plt.close(fig)


def fig_controls() -> None:
    rules = ("yangJ", "D", "lrpT", "selected", "causalnogen")
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
    that lacks it (mean over tasks and folds, cross-fitted alpha; with a two-stage screen, the quick pass)."""
    from rebuild_sets import COMBOS
    order = list(FEATURES)
    quick = {f: quick_scores(f, "additions") for f in (1, 2)} if QUICK else None
    out = {}
    for scope, prefix in SCOPES.items():
        if QUICK:
            acc = {c: float(np.mean([quick[f][scope][c] for f in (1, 2)])) for c in COMBOS}
        else:
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
        ax.set_xlim(0, max(0.5, 1.08 * max(alone.values())) if QUICK else 0.5)
        ax.set_title(f"({'ab'[j]}) {scope}: feature alone, $N={SCREEN_N}$", loc="left")
        ax.set_xlabel(SCREEN_METRIC)
        ax = axes[1, j]
        ax.barh(y, [added[f][0] for f in order], color=cols, height=0.62)
        ax.axvline(0, color=rt.AXIS_LINE, lw=0.8)
        changes = [added[f][0] for f in order]
        ax.set_xlim(*((min(-0.2, 1.1 * min(changes)), max(0.32, 1.1 * max(changes))) if QUICK else (-0.2, 0.32)))
        ax.set_title(f"({'cd'[j]}) {scope}: feature added", loc="left")
        ax.set_xlabel("mean change, quick pass" if QUICK else "mean change in accuracy")
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
    Descriptive: the best is chosen on the accuracies it is shown with, so it is not held out. With a
    two-stage screen, every accuracy (Selected's too) is the quick pass's, mean over the two folds."""
    from rebuild_sets import COMBOS
    sel = json.loads((DIR / "selected.json").read_text())
    quick = {f: quick_per_task(f, "additions") for f in (1, 2)} if QUICK else None
    out = {}
    for scope, prefix in SCOPES.items():
        rows = {}
        for t in simple.TASKS:
            if QUICK:
                acc = {c: float(np.mean([quick[f][scope][c][t] for f in (1, 2)])) for c in COMBOS}
                selected = float(np.mean([quick[f][scope][sel[str(f)][scope]["combo"]][t] for f in (1, 2)]))
            else:
                acc = {c: per_task(f"{prefix}screen_{c}", t) for c in COMBOS}
                selected = per_task(f"{prefix}selected{SCREEN_N}", t)
            ranked = sorted(COMBOS, key=lambda c: (-acc[c], len(c), COMBOS.index(c)))
            share = {f: np.mean([f in c for c in ranked[:top_k]]) for f in FEATURES}
            rows[t] = dict(best=ranked[0], acc=acc[ranked[0]], share=share, selected=selected)
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
                     f"$N={SCREEN_N}$" + (", quick pass" if QUICK else ""), loc="left", pad=86)
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
    lim = max(0.7, 1.05 * max(v for f in (1, 2) for s in SCOPES for v in scores[f][s].values()))
    for ax, scope in zip(axes, SCOPES):
        combos = sorted(scores[1][scope])
        x = np.array([scores[1][scope][c] for c in combos])
        y = np.array([scores[2][scope][c] for c in combos])
        has_d = np.array(["D" in c for c in combos])
        corr.append(float(np.corrcoef(x, y)[0, 1]))
        ax.plot([0, lim], [0, lim], color=rt.AXIS_LINE, lw=0.8, zorder=1)
        ax.scatter(x[~has_d], y[~has_d], s=9, color=rt.GREY, alpha=0.6, lw=0, label="without D", zorder=2)
        ax.scatter(x[has_d], y[has_d], s=9, color=rt.MAGENTA, alpha=0.6, lw=0, label="with D", zorder=2)
        for c in {sel["1"][scope]["combo"], sel["2"][scope]["combo"]}:
            ax.scatter([scores[1][scope][c]], [scores[2][scope][c]], s=46, facecolor="none", edgecolor=rt.AQUA, lw=1.6,
                       zorder=4)
            ax.annotate(f"Selected: {c}", (scores[1][scope][c], scores[2][scope][c]), xytext=(-8, 8),
                        textcoords="offset points", ha="right", fontsize=7, color=rt.INK)
        ax.set_title(f"({'ab'[list(SCOPES).index(scope)]}) {scope}, all {len(combos)} combinations at $N={SCREEN_N}$"
                     + ("\nquick pass: first token, $\\alpha=2$, 10 items" if QUICK else ""), loc="left")
        ax.set_xlabel("fold 1 (tested on half B)")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
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
    fig, axes = plt.subplots(2, len(TYPES), figsize=(6.5, 3.7), sharex=True, sharey=True)
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
                ax.plot(xs, ys, color=col, marker=mk, ms=2.8, lw=1.0, ls=line_style(r), label=plot_label(r))
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
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), handlelength=2.2,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(rt.FIGURES / "design_types.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ stage 3 (Sec. 6)

S3_SCENARIO = {"wt": "Unseen wordings, transfer", "wq": "Unseen wordings, re-identified",
               "sum": "Summed vector, sets of Sec. 5 with the means of Yang et al.'s five templates",
               "et": "Example-based prompts, transfer", "eq": "Example-based prompts, re-identified",
               "esum": "Summed vector, sets re-identified on example-based prompts"}
# Wording groups (Sec. 6.1): Yang et al.'s cloze template and four question-answer templates (Sec. 5), and the
# six groups of test wordings (key Q), ordered by form.
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
    of every group in both folds (a test wording gives no mean without 10 correct prompts); and the tasks left out."""
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


ABBREV = {"yangJ": "Yang", "D": "D", "selected": "Sel.", "causal": "Causal", "causalnogen": "C.$-$gen."}  # narrow tables


def short_label(rule: str) -> str:
    """Header of a rule spanning two narrow columns."""
    return "Causal $-$ gen." if rule == "causalnogen" else RULES[rule][0]


def table_pair(key_a: str | None, key_b: str, cols: tuple[str, str], name: str) -> None:
    """Two tests side by side for every S3 rule at every size (key None: the sets of Sec. 5 as in Table 3)."""
    acc_a = None if key_a is None else ACC3[key_a]
    header = ["& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{short_label(r)}}}" for r in S3_RULES) + " \\\\",
              " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(S3_RULES))),
              "$N$ & " + " & ".join(" & ".join(cols) for _ in S3_RULES) + " \\\\"]
    rows = []
    for scope, prefix in SCOPES.items():
        rows.append(f"\\multicolumn{{{2 * len(S3_RULES) + 1}}}{{@{{}}l}}{{\\textit{{{scope.capitalize()}}}}} \\\\")
        for n in SIZES:
            vals = []
            for r in S3_RULES:
                vals += [rt.num(mean_over(f"{prefix}{r}{n}", acc=acc_a)), rt.num(mean_over(f"{prefix}{r}{n}", acc=ACC3[key_b]))]
            rows.append(f"{n} & " + " & ".join(vals) + " \\\\")
        if scope == "per task":
            rows.append("\\midrule")
    write(name, rt.tabular("@{}r" + "rr" * len(S3_RULES) + "@{}", header, rows))


def table_summed_pick40() -> None:
    """Shared Selected at every size against the 40-head pick (the stage-2 combination, its ranking cut at every
    size), each patched head by head and as one summed vector with Yang's five templates' means; the 40-head
    pick also with its random sets and control means as one summed vector. The 40-head pick's rows are those
    of the stage-2 and summed reports before the size picks replace them (``with_sized``)."""
    acc40 = cells([DIR / f"{s}_per_task.csv" for s in ("s1", "s2")])
    d, st = S3["sum"]
    sum40 = cells([d / f"{st}_per_task.csv"])
    sel40 = json.loads((DIR / "selected.json").read_text())
    pick40 = sorted({sel40[str(f)]["shared"]["combo"] for f in (1, 2)})
    assert len(pick40) == 1, pick40
    header = ["& \\multicolumn{3}{c}{Selected at $N$} & \\multicolumn{4}{c}{" + pick40[0] + " (40-head pick)} \\\\",
              "\\cmidrule(lr){2-4} \\cmidrule(lr){5-8}",
              "$N$ & Folds 1 / 2 & heads & sum & heads & sum & rand. & ctrl. \\\\"]
    rows = []
    for n in SIZES:
        name = f"shared_selected{n}"
        if n in SIZE_SCREEN:
            sel = json.loads((SIZE_SCREEN[n] / "selected.json").read_text())
            combos = [sel[str(f)]["combo"] for f in (1, 2)]
        else:
            combos = [sel40[str(f)]["shared"]["combo"] for f in (1, 2)]
        picks = combos[0] if combos[0] == combos[1] else " / ".join(combos)
        rows.append(f"{n} & {picks} & {rt.num(mean_over(name))} & {rt.num(mean_over(name, acc=ACC3['sum']))} & "
                    f"{rt.num(mean_over(name, acc=acc40))} & {rt.num(mean_over(name, acc=sum40))} & "
                    f"{rt.num(mean_over(name, 'random', acc=sum40))} & {rt.num(mean_over(name, 'control', acc=sum40))} \\\\")
    write("design_summed_pick40", rt.tabular("@{}rlrrrrrr@{}", header, rows))


def table_s3_controls(keys: tuple[str, ...], name: str) -> None:
    """Random sets and control means of the S3 rules in the given stage-3 tests, every size."""
    header = ["& & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{short_label(r)}}}" for r in S3_RULES) + " \\\\",
              " ".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(S3_RULES))),
              "Scope & $N$ & " + " & ".join("rand. & ctrl." for _ in S3_RULES) + " \\\\"]
    rows = []
    keys = [k for k in keys if k in ACC3]
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
    write(name, rt.tabular("@{}lr" + "rr" * len(S3_RULES) + "@{}", header, rows))


def table_s3_types(keys: tuple[str, ...], name: str, n: int = SCREEN_N) -> None:
    """Accuracy by task type at ``n`` heads in one or more tests (a block per test when several)."""
    header = ["Rule & " + " & ".join(TYPES) + " & all \\\\"]
    rows = []
    for i, key in enumerate(keys):
        if len(keys) > 1:
            rows.append(f"\\multicolumn{{{len(TYPES) + 2}}}{{@{{}}l}}{{\\textbf{{{S3_SCENARIO[key]}}}}} \\\\")
        for scope, prefix in SCOPES.items():
            rows.append(f"\\multicolumn{{{len(TYPES) + 2}}}{{@{{}}l}}{{\\textit{{{scope.capitalize()}}}}} \\\\")
            for r in S3_RULES:
                nm = f"{prefix}{r}{n}"
                rows.append(f"{RULES[r][0]} & " + " & ".join(rt.num(mean_over(nm, tasks=ts, acc=ACC3[key])) for ts in TYPES.values())
                            + f" & {rt.num(mean_over(nm, acc=ACC3[key]))} \\\\")
            if scope == "per task":
                rows.append("\\addlinespace" if len(keys) > 1 else "\\midrule")
        if i < len(keys) - 1:
            rows.append("\\midrule")
    write(name, rt.tabular("@{}l" + "r" * (len(TYPES) + 1) + "@{}", header, rows))


def table_candidates() -> None:
    """Sec. 6.1, re-identified: the candidate combinations (the ten best of Sec. 5.4 of each fold and scope) at 40
    heads with the Q-wording means, as identified in Sec. 5 and re-identified; which fold selected each."""
    sel = json.loads((WPOOL / "selected.json").read_text())
    header = ["Scope & Combination & Selected in fold & Sec.~5 sets & re-identified \\\\"]
    rows = []
    for scope, prefix in (("per task", "screen_"), ("shared", "shared_screen_")):
        cands = sorted({c for f in "12" for c in sel[f][scope]["candidates"]})
        chosen = {c: [f for f in "12" if sel[f][scope]["combo"] == c] for c in cands}
        vals = {c: (mean_over(prefix + c, acc=ACC3["wt"]), mean_over(prefix + c, acc=ACC3["wq"])) for c in cands}
        for k, c in enumerate(sorted(cands, key=lambda c: -vals[c][1])):
            mark = (lambda x: f"\\textbf{{{x}}}") if chosen[c] else (lambda x: x)
            rows.append(f"{scope if k == 0 else ''} & {mark(c)} & {', '.join(chosen[c])} & {rt.num(vals[c][0])} & "
                        f"{rt.num(vals[c][1])} \\\\")
        if scope == "per task":
            rows.append("\\midrule")
    write("design_wordings_candidates", rt.tabular("@{}llcrr@{}", header, rows))


def table_selected_examples() -> None:
    """Sec. 6.2, re-identified: the five best of the 127 example combinations per fold and scope (as Table 4)."""
    from rebuild_sets_s3 import EX_FEATURES
    sel = json.loads((EPOOL / "selected.json").read_text())
    scores = {fold: screen_scores(fold, "examples") for fold in (1, 2)}
    header = ["Scope & Fold & Rank & Combination & Features & Other fold & Own fold \\\\"]
    rows = []
    for scope in SCOPES:
        for fold in (1, 2):
            other = 3 - fold
            pool = sel[str(fold)][scope].get("candidates") or scores[other][scope]  # the shortlist of a two-stage screen
            ranked = sorted(pool, key=lambda c: (-scores[other][scope][c], len(c)))[:5]
            for k, c in enumerate(ranked, 1):
                mark = (lambda x: f"\\textbf{{{x}}}") if c == sel[str(fold)][scope]["combo"] else (lambda x: x)
                names = ", ".join(EX_FEATURES[f][0] if f != "D" else "D" for f in c)
                rows.append(f"{scope if fold == 1 and k == 1 else ''} & {fold if k == 1 else ''} & {k} & {mark(c)} & {names} & "
                            f"{rt.num(scores[other][scope][c])} & {rt.num(scores[fold][scope][c])} \\\\")
            if not (scope == "shared" and fold == 2):
                rows.append("\\midrule" if fold == 2 else "\\addlinespace")
    write("design_examples_selected", rt.tabular("@{}lrrl>{\\raggedright\\arraybackslash}p{5.2cm}rr@{}", header, rows))


def table_control_tasks(key: str = "eq", n: int = SCREEN_N) -> None:
    """Control means per task at ``n`` heads in one test, both scopes."""
    header = ["& \\multicolumn{" + str(len(S3_RULES)) + "}{c}{One set per task} & \\multicolumn{" + str(len(S3_RULES))
              + "}{c}{One set for all tasks} \\\\",
              f"\\cmidrule(lr){{2-{1 + len(S3_RULES)}}} \\cmidrule(lr){{{2 + len(S3_RULES)}-{1 + 2 * len(S3_RULES)}}}",
              "Task & " + " & ".join(ABBREV[r] for r in S3_RULES) + " & " + " & ".join(ABBREV[r] for r in S3_RULES) + " \\\\"]
    rows = []
    acc = ACC3[key]
    for t in simple.TASKS + ("all",):
        ts = simple.TASKS if t == "all" else (t,)
        if t == "all":
            rows.append("\\midrule")
        vals = [rt.num(mean_over(f"{p}{r}{n}", "control", tasks=ts, acc=acc)) for p in SCOPES.values() for r in S3_RULES]
        rows.append(f"{'mean' if t == 'all' else rt.tex_name('simple', t)} & " + " & ".join(vals) + " \\\\")
    write(f"design_control_tasks_{key}", rt.tabular("@{}l" + "r" * 2 * len(S3_RULES) + "@{}", header, rows))


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


def fig_pair(key_a: str | None, key_b: str, labels: tuple[str, str], name: str, ymax: float = 0.75) -> None:
    """Every S3 rule in test ``key_b`` (markers) over test ``key_a`` (thin lines; None: Sec. 5)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
    for ax, (scope, prefix) in zip(axes, SCOPES.items()):
        for r in S3_RULES:
            label, col, mk = RULES[r]
            curve(ax, lambda n: f"{prefix}{r}{n}", acc=None if key_a is None else ACC3[key_a], color=col, lw=0.9, alpha=0.4,
                  zorder=1)
            curve(ax, lambda n: f"{prefix}{r}{n}", acc=ACC3[key_b], color=col, marker=mk, ms=4, label=plot_label(r),
                  lw=1.8 if r == "selected" else 1.2, zorder=3 if r == "selected" else 2)
        ax.set_title(panel_title(scope), loc="left")
        axes_style(ax, ylabel=ax is axes[0], ymax=ymax)
    handles, hl = axes[1].get_legend_handles_labels()
    fig.legend(handles, hl, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.055), handlelength=1.8, columnspacing=1.2)
    fig.legend([Line2D([], [], color=rt.INK_2, marker="o", ms=3.5, lw=1.2), Line2D([], [], color=rt.INK_2, lw=0.9, alpha=0.4)],
               list(labels), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01), handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(rt.FIGURES / f"{name}.pdf")
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


def fig_summed(key_heads: str | None = None, key_sum: str = "sum", name: str = "design_summed", ymax: float = 0.75) -> None:
    fig, axes = plt.subplots(2, len(S3_RULES), figsize=(6.5, 3.6), sharex=True, sharey=True)
    for i, (scope, prefix) in enumerate(SCOPES.items()):
        for j, r in enumerate(S3_RULES):
            ax = axes[i, j]
            label, col, mk = RULES[r]
            curve(ax, lambda n: f"{prefix}{r}{n}", acc=None if key_heads is None else ACC3[key_heads], color=col, marker=mk, ms=3.5)
            curve(ax, lambda n: f"{prefix}{r}{n}", acc=ACC3[key_sum], color=col, marker=mk, ms=3.5, mfc="none", ls=(0, (4, 2)),
                  lw=1.0)
            axes_style(ax, ylabel=j == 0, ymax=ymax)
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
    fig.savefig(rt.FIGURES / f"{name}.pdf")
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
    q_words = "means of the 15 test wordings"
    if "wt" in ACC3:
        table_s3_acc("wt", "design_wordings_transfer")
        groups, left_out = by_group()
        print("wording groups: tasks left out", left_out)
        table_groups(groups)
        fig_pair(None, "wt", (q_words, "means of Yang's five templates (Sec. 5)"), "design_wordings_transfer")
        fig_groups(groups)
    if "wq" in ACC3:
        table_s3_acc("wq", "design_wordings_reid")
        table_candidates()
        fig_pair("wt", "wq", ("re-identified on Yang's five + 15 identification wordings", "identified on Yang's five (transfer)"),
                 "design_wordings_reid")
    table_s3_types(tuple(k for k in ("wt", "wq") if k in ACC3), "design_wordings_types")
    if "sum" in ACC3:
        table_pair(None, "sum", ("heads", "sum"), "design_summed")
        table_summed_pick40()
        table_s3_types(("sum",), "design_summed_types")
        fig_summed()
    if "et" in ACC3 and "eq" in ACC3:
        table_pair("et", "eq", ("tr.", "re-id."), "design_examples")
        table_selected_examples()
        table_s3_types(("et", "eq"), "design_examples_types")
        table_control_tasks("eq")
        fig_pair("et", "eq", ("re-identified on the 3 identification formats", "identified on Yang's five templates (transfer)"),
                 "design_examples", ymax=1.0)
    if "esum" in ACC3:
        table_pair("eq", "esum", ("heads", "sum"), "design_summed_examples")
        table_s3_types(("esum",), "design_summed_examples_types")
        fig_summed("eq", "esum", "design_summed_examples", ymax=1.0)
    if ACC3:
        table_s3_controls(("wt", "wq", "sum"), "design_s3_controls")
        table_s3_controls(("et", "eq", "esum"), "design_s3_controls_examples")
        print("alpha = 2 share:", alpha2_share())


# ------------------------------------------------------------------ overlap between conditions (Sec. 7)
#
# A condition is a rule, the prompts its sets are identified on (pool) and a scope; each has one set per fold and
# size (and task, per task). For two conditions A, B of the same size, s = |A & B| / N (mean over folds and, per
# task, tasks), c = the share two random sets with the same heads per layer would have (layer-matched chance), and
# a condition's stability rho = its fold-1 set against its fold-2 set (identified on the two disjoint item halves),
# with its own chance. Normalized overlap q = (s - c) / sqrt((rho_A - c_A)(rho_B - c_B)): 0 = no more than random
# sets in the same layers, 1 = as alike as each condition is with itself on the other half of the items.

OV_POOLS = {"I": "templates", "W": "wordings", "E": "examples"}  # Sec. 5 sets; re-identified (Secs. 6.1, 6.2)
OV_SIZES = (20, 40, 80, 120)
OV_DIR = rt.RESULTS / "rebuild" / "overlap"
T0 = simple.TASKS[0]  # a shared set is the same for every task
_OVS = {}


def ov_sets():
    if not _OVS:
        _OVS.update(I=SETS, W=with_sized_sets(heads(WPOOL), "wordings_n"), E=with_sized_sets(heads(EPOOL), "examples_n"))
    return _OVS


def ov_get(cond, fold: int, task: str, n: int):
    pool, scope, rule = cond
    return ov_sets()[pool][fold][task].get(f"{SCOPES[scope]}{rule}{n}")


def ov_share(a, b, n: int, units):
    """Mean share in common and layer-matched chance over ``units`` = [((fold_a, task_a), (fold_b, task_b))];
    None if a set is missing in any unit."""
    vals = []
    for (fa, ta), (fb, tb) in units:
        A, B = ov_get(a, fa, ta, n), ov_get(b, fb, tb, n)
        if A is None or B is None:
            return None
        assert len(A) == len(B) == n
        vals.append((len(A & B) / n, chance_overlap(A, B) / n))
    return tuple(np.mean(vals, axis=0))


def ov_units(scope: str, task: str | None = None, task_b: str | None = None, stability: bool = False):
    """Always one set from each fold, so that two conditions built from the same scores of a fold (Causal and
    Causal - generic, or D and a combination with D) do not share that fold's noise: a comparison averages fold 1
    against fold 2 and the reverse, stability is fold 1 against fold 2. Per task every task unless one is given."""
    tasks = [T0] if scope == "shared" and task is None else ([task] if task else simple.TASKS)
    folds = [(1, 2)] if stability else [(1, 2), (2, 1)]
    return [((fa, t), (fb, task_b or t)) for fa, fb in folds for t in tasks]


def ov_q(s, ra, rb):
    if s is None or ra is None or rb is None:
        return None
    den = (ra[0] - ra[1]) * (rb[0] - rb[1])
    return float((s[0] - s[1]) / np.sqrt(den)) if den > 0 else None


def ov_conditions(scope: str) -> list[tuple[str, str, str]]:
    """Conditions with a set at some size of OV_SIZES on every task and fold, in display order."""
    out = []
    for p in OV_POOLS:
        for r in LADDER:
            c = (p, scope, r)
            if any(ov_share(c, c, n, ov_units(scope)) is not None for n in OV_SIZES):
                out.append(c)
    return out


def overlap_rows() -> pd.DataFrame:
    """Every comparison at every size of SIZES; written to results/rebuild/overlap/overlap.csv."""
    rows = []
    stab = {}
    for scope in SCOPES:
        conds = ov_conditions(scope)
        for n in SIZES:
            for c in conds:
                rho = ov_share(c, c, n, ov_units(scope, stability=True))
                stab[(c, n)] = rho
                rnd = None
                if rho is not None:
                    pre = f"random_{SCOPES[scope]}{c[2]}{n}"
                    rnd = float(np.mean([len(ov_get(c, f, t, n) & ov_sets()[c[0]][f][t][f"{pre}_{i}"]) / n
                                         for f in (1, 2) for t in ([T0] if scope == "shared" else simple.TASKS)
                                         for i in range(3)]))
                rows.append(dict(kind="stability", scope=scope, n=n, pool_a=c[0], rule_a=c[2],
                                 share=None if rho is None else rho[0], chance=None if rho is None else rho[1],
                                 random_share=rnd))
            for i, a in enumerate(conds):  # comparisons 2 (same pool) and 3 (same rule), and the other pairs
                for b in conds[i + 1:]:
                    s = ov_share(a, b, n, ov_units(scope))
                    kind = "rules" if a[0] == b[0] else ("pools" if a[2] == b[2] else "other")
                    rows.append(dict(kind=kind, scope=scope, n=n, pool_a=a[0], rule_a=a[2], pool_b=b[0], rule_b=b[2],
                                     share=None if s is None else s[0], chance=None if s is None else s[1],
                                     q=ov_q(s, stab[(a, n)], stab[(b, n)])))
    for n in SIZES:  # comparison 4: a task's own set against the set for all tasks, same rule and pool
        for p in OV_POOLS:
            for r in LADDER:
                a, b = (p, "per task", r), (p, "shared", r)
                s = ov_share(a, b, n, ov_units("per task"))
                rows.append(dict(kind="scope", scope="per task", n=n, pool_a=p, rule_a=r, pool_b=p, rule_b=r,
                                 share=None if s is None else s[0], chance=None if s is None else s[1],
                                 q=ov_q(s, stab.get((a, n)), stab.get((b, n)))))
    for n in SIZES:  # comparison 5: task against task, same rule and pool, per task
        for p in OV_POOLS:
            for r in LADDER:
                c = (p, "per task", r)
                for t in simple.TASKS:  # each task's own stability, reported; not used as its ceiling
                    s = ov_share(c, c, n, ov_units("per task", t, stability=True))
                    rows.append(dict(kind="task_stability", scope="per task", n=n, pool_a=p, rule_a=r, task_a=t,
                                     share=None if s is None else s[0], chance=None if s is None else s[1]))
                # Ceiling: the rule's stability over all 14 tasks, as for comparisons 2-4; a single task's stability
                # can lie near chance (per-task Causal on English-German), which would make its ratio meaningless.
                for i, t in enumerate(simple.TASKS):
                    for t2 in simple.TASKS[i + 1:]:
                        s = ov_share(c, c, n, ov_units("per task", t, t2))
                        rows.append(dict(kind="tasks", scope="per task", n=n, pool_a=p, rule_a=r, pool_b=p, rule_b=r,
                                         task_a=t, task_b=t2, share=None if s is None else s[0],
                                         chance=None if s is None else s[1], q=ov_q(s, stab.get((c, n)), stab.get((c, n))),
                                         same_type=any(t in ts and t2 in ts for ts in TYPES.values())))
    df = pd.DataFrame(rows)
    OV_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OV_DIR / "overlap.csv", index=False)
    return df


def ov_cell(df, **kw):
    m = df
    for k, v in kw.items():
        m = m[m[k] == v]
    assert len(m) <= 1, kw
    return None if m.empty else m.iloc[0]


def ov_label(c) -> str:
    return plot_label(c[2]).replace("Yang terms ", "Yang ")


OV_CMAP = LinearSegmentedColormap.from_list("blue", [rt.SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def ov_short(v) -> str:
    """Two decimals without the leading zero, typeset minus: cells of the overlap figures."""
    if v is None or pd.isna(v):
        return "--"
    t = f"{v:.2f}"
    t = "0.00" if t == "-0.00" else t
    return t.replace("0.", ".", 1).replace("-", "\u2212")


def ov_colorbar(fig, rect, label: str) -> None:
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=OV_CMAP, norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=6.5)
    cax.xaxis.set_label_position("top")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=0)


def fig_overlap(df: pd.DataFrame, n: int) -> None:
    """Normalized overlap of every pair of conditions (lower triangle) and their stability (diagonal), per scope;
    conditions without a set of this size on every task and fold are left out."""
    stab = df[(df.kind == "stability") & (df.n == n)]
    pairs = df[df.kind.isin(["rules", "pools", "other"]) & (df.n == n)]
    shown = {scope: [c for c in ov_conditions(scope)
                     if (r := ov_cell(stab, scope=scope, pool_a=c[0], rule_a=c[2])) is not None and pd.notna(r.share)]
             for scope in SCOPES}
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 8.7),
                             gridspec_kw={"hspace": 0.36, "height_ratios": [len(shown[s]) for s in SCOPES]})
    for j, (ax, scope) in enumerate(zip(axes, SCOPES)):
        conds = shown[scope]
        k = len(conds)
        grid = np.full((k, k), np.nan)
        for a in range(k):
            for b in range(a):
                ca, cb = conds[b], conds[a]  # pairs are stored in display order
                r = ov_cell(pairs, scope=scope, pool_a=ca[0], rule_a=ca[2], pool_b=cb[0], rule_b=cb[2])
                grid[a, b] = r.q if r is not None and pd.notna(r.q) else np.nan
        ax.imshow(np.clip(np.nan_to_num(grid, nan=0), 0, 1), cmap=OV_CMAP, vmin=0, vmax=1, interpolation="nearest")
        for a in range(k):
            for b in range(a + 1, k):
                ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, color=rt.SURFACE, zorder=2))
            for b in range(a):
                v = grid[a, b]
                ax.text(b, a, ov_short(v), ha="center", va="center", fontsize=5.8, zorder=3,
                        color=rt.SURFACE if (not np.isnan(v) and v > 0.55) else rt.INK,
                        fontweight="bold" if conds[a][2] == conds[b][2] else "normal")
            st = ov_cell(stab, scope=scope, pool_a=conds[a][0], rule_a=conds[a][2])
            ax.add_patch(plt.Rectangle((a - 0.5, a - 0.5), 1, 1, facecolor=rt.GRIDLINE, edgecolor="none", zorder=2))
            ax.text(a, a, ov_short(st.share), ha="center", va="center", fontsize=5.8, fontstyle="italic", color=rt.INK_2,
                    zorder=3)
        starts = [i for i in range(k) if i == 0 or conds[i][0] != conds[i - 1][0]]
        for i in starts[1:]:
            ax.plot([i - 0.5, i - 0.5], [i - 0.5, k - 0.5], color=rt.INK_2, lw=0.8, zorder=4)
            ax.plot([-0.5, i - 0.5], [i - 0.5, i - 0.5], color=rt.INK_2, lw=0.8, zorder=4)
        labels = [ov_label(c) for c in conds]
        ax.set_xticks(range(k))
        ax.set_xticklabels(labels, rotation=60, ha="right", rotation_mode="anchor", fontsize=6.5)
        ax.set_yticks(range(k))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.tick_params(length=0)
        for s0, s1 in zip(starts, starts[1:] + [k]):
            mid = (s0 + s1 - 1) / 2
            ax.annotate(OV_POOLS[conds[s0][0]], (0, mid), xycoords=("axes fraction", "data"), xytext=(-62, 0),
                        textcoords="offset points", rotation=90, ha="center", va="center", fontsize=7, color=rt.INK)
            ax.annotate(OV_POOLS[conds[s0][0]], (mid, 0), xycoords=("data", "axes fraction"), xytext=(0, -58),
                        textcoords="offset points", ha="center", va="center", fontsize=7, color=rt.INK)
        ax.set_xlim(-0.5, k - 0.5)
        ax.set_ylim(k - 0.5, -0.5)
        ax.grid(False)
        ax.spines[:].set_visible(False)
        ax.set_title(f"({'ab'[j]}) {'One set per task' if scope == 'per task' else 'One set for all 14 tasks'}, $N={n}$",
                     loc="left")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.115)
    fig.canvas.draw()  # the equal-aspect axes take their final box here
    pos = axes[1].get_position()  # the colorbar goes into the empty upper triangle of panel (b)
    ov_colorbar(fig, (pos.x1 - 0.33, pos.y1 - 0.035, 0.3, 0.008),
                "below the diagonal: normalized overlap\n(0 = layer-matched chance, 1 = stability)")
    fig.savefig(rt.FIGURES / f"design_overlap_{n}.pdf")
    plt.close(fig)


def fig_overlap_tasks(df: pd.DataFrame, n: int = SCREEN_N, pool: str = "I") -> None:
    """Task against task, normalized overlap, per rule that fills the size per task; diagonal: each task's own
    stability."""
    tasks = [t for ts in TYPES.values() for t in ts]
    sub = df[(df.kind == "tasks") & (df.n == n) & (df.pool_a == pool)]
    rules = [r for r in ("D", "selected", "causal", "causalnogen") if sub[sub.rule_a == r].q.notna().any()]  # 2x2 panels
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.8), gridspec_kw={"hspace": 0.1, "wspace": 0.06})
    breaks = np.cumsum([len(ts) for ts in TYPES.values()])[:-1]
    for j, (ax, r) in enumerate(zip(axes.flat, rules)):
        g = np.full((len(tasks), len(tasks)), np.nan)
        for _, row in sub[sub.rule_a == r].iterrows():
            a, b = sorted((tasks.index(row.task_a), tasks.index(row.task_b)), reverse=True)
            g[a, b] = row.q
        ax.imshow(np.clip(np.nan_to_num(g, nan=0), 0, 1), cmap=OV_CMAP, vmin=0, vmax=1, interpolation="nearest")
        own = df[(df.kind == "task_stability") & (df.n == n) & (df.pool_a == pool) & (df.rule_a == r)].set_index("task_a")
        for a in range(len(tasks)):
            for b in range(a + 1, len(tasks)):
                ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, color=rt.SURFACE, zorder=2))
            ax.add_patch(plt.Rectangle((a - 0.5, a - 0.5), 1, 1, facecolor=rt.GRIDLINE, edgecolor="none", zorder=2))
            ax.text(a, a, ov_short(own.loc[tasks[a], "share"]), ha="center", va="center", fontsize=4.8,
                    fontstyle="italic", color=rt.INK_2, zorder=3)
        for y in breaks:
            ax.plot([y - 0.5, y - 0.5], [y - 0.5, len(tasks) - 0.5], color=rt.INK_2, lw=0.6, zorder=3)
            ax.plot([-0.5, y - 0.5], [y - 0.5, y - 0.5], color=rt.INK_2, lw=0.6, zorder=3)
        ax.set_xticks(range(len(tasks)))
        ax.set_yticks(range(len(tasks)))
        names = [rt.plot_name("simple", t) for t in tasks]
        ax.set_xticklabels(names if j >= 2 else [], rotation=60, ha="right", rotation_mode="anchor", fontsize=6)
        ax.set_yticklabels(names if j % 2 == 0 else [], fontsize=6)
        ax.tick_params(length=0)
        ax.grid(False)
        ax.spines[:].set_visible(False)
        ax.set_title(f"({'abcd'[j]}) {plot_label(r)}", loc="left", fontsize=8)
    for ax in list(axes.flat)[len(rules):]:
        ax.set_visible(False)
    ov_colorbar(fig, (0.75, 0.9, 0.22, 0.01), "normalized overlap")
    fig.subplots_adjust(left=0.17, right=0.99, top=0.965, bottom=0.135)
    fig.savefig(rt.FIGURES / "design_overlap_tasks.pdf")
    plt.close(fig)


def ov_fmt(r, what: str) -> str:
    if r is None or pd.isna(r.share):
        return "--"
    return f"{rt.num(r.share)} ({rt.num(r.chance)})" if what == "share" else rt.num(r.q)


def table_overlap_stability(df: pd.DataFrame) -> None:
    """Comparison 1: each condition's fold-1 set against its fold-2 set (chance), and its share in its random sets."""
    sub = df[df.kind == "stability"]
    header = ["Scope & Prompts & Rule & " + " & ".join(f"{n}" for n in SIZES) + " \\\\"]
    rows = []
    for block, col in (("Stability: fold 1 against fold 2 (chance)", "share"), ("Share in its own random sets", "random")):
        rows.append(f"\\multicolumn{{{len(SIZES) + 3}}}{{@{{}}l}}{{\\textit{{{block}}}}} \\\\")
        for scope in SCOPES:
            for i, c in enumerate(ov_conditions(scope)):
                cells_ = []
                for n in SIZES:
                    r = ov_cell(sub, scope=scope, n=n, pool_a=c[0], rule_a=c[2])
                    if col == "share":
                        cells_.append("--" if r is None or pd.isna(r.share) else f"{rt.num(r.share)} ({rt.num(r.chance)})")
                    else:
                        cells_.append(rt.num(None if r is None else r.random_share))
                first_of_pool = i == 0 or ov_conditions(scope)[i - 1][0] != c[0]
                rows.append(f"{scope if i == 0 else ''} & {OV_POOLS[c[0]] if first_of_pool else ''} & {ov_tex(c[2]) if c[2] in ZERO_SHOT else RULES[c[2]][0]} & "
                            + " & ".join(cells_) + " \\\\")
            rows.append("\\addlinespace")
        rows.pop()
        rows.append("\\midrule")
    rows.pop()
    write("design_overlap_stability", rt.tabular("@{}lll" + "r" * len(SIZES) + "@{}", header, rows))


def ov_table(df, kind: str, pairs, name: str) -> None:
    """Share in common (chance) and normalized overlap at OV_SIZES; ``pairs`` = [(scope, label_1, label_2, query)]."""
    header = ["& & & \\multicolumn{4}{c}{Share in common (chance)} & \\multicolumn{4}{c}{Normalized} \\\\",
              "\\cmidrule(lr){4-7} \\cmidrule(lr){8-11}",
              "Scope & & & " + " & ".join(f"$N={n}$" for n in OV_SIZES) + " & " + " & ".join(f"{n}" for n in OV_SIZES) + " \\\\"]
    rows = []
    last = None
    for scope, l1, l2, query in pairs:
        if last is not None and scope != last[0]:
            rows.append("\\midrule")
        elif last is not None and l1 != last[1]:
            rows.append("\\addlinespace")
        cells_ = [ov_cell(df[df.kind == kind], scope=scope, n=n, **query) for n in OV_SIZES]
        rows.append(f"{scope if last is None or scope != last[0] else ''} & {l1 if last is None or l1 != last[1] or scope != last[0] else ''}"
                    f" & {l2} & " + " & ".join(ov_fmt(r, "share") for r in cells_) + " & "
                    + " & ".join(ov_fmt(r, "q") for r in cells_) + " \\\\")
        last = (scope, l1)
    write(name, rt.tabular("@{}lll" + "r" * 8 + "@{}", header, rows))


def table_overlap_tasks(df: pd.DataFrame) -> None:
    """Comparison 5 summary: mean normalized overlap over pairs of tasks of the same type and of different types."""
    sub = df[df.kind == "tasks"]
    header = ["& & \\multicolumn{4}{c}{Same task type} & \\multicolumn{4}{c}{Different task types} \\\\",
              "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}",
              "Prompts & Rule & " + " & ".join(f"$N={n}$" for n in OV_SIZES) + " & " + " & ".join(f"{n}" for n in OV_SIZES) + " \\\\"]
    rows = []
    for p in OV_POOLS:
        for i, r in enumerate([r for r in LADDER if sub[(sub.pool_a == p) & (sub.rule_a == r) & sub.n.isin(OV_SIZES)].q.notna().any()]):
            cells_ = []
            for same in (True, False):
                for n in OV_SIZES:
                    m = sub[(sub.pool_a == p) & (sub.rule_a == r) & (sub.n == n) & (sub.same_type == same)]
                    cells_.append(rt.num(m.q.mean()) if len(m) and m.q.notna().all() else "--")
            rows.append(f"{OV_POOLS[p] if i == 0 else ''} & {RULES[r][0]} & " + " & ".join(cells_) + " \\\\")
        rows.append("\\addlinespace")
    rows.pop()
    write("design_overlap_tasks", rt.tabular("@{}ll" + "r" * 8 + "@{}", header, rows))


def table_overlap_task_stability(df: pd.DataFrame, n: int = SCREEN_N) -> None:
    """Each task's own stability at ``n`` heads, per prompts and rule (fold 1 against fold 2)."""
    sub = df[(df.kind == "task_stability") & (df.n == n)]
    rules = [r for r in ("D", "selected", "causal", "causalnogen")]
    header = [" & " + " & ".join(f"\\multicolumn{{{len(rules)}}}{{c}}{{{OV_POOLS[p]}}}" for p in OV_POOLS) + " \\\\",
              " ".join(f"\\cmidrule(lr){{{2 + i * len(rules)}-{1 + (i + 1) * len(rules)}}}" for i in range(len(OV_POOLS))),
              "Task & " + " & ".join(ABBREV[r] for _ in OV_POOLS for r in rules) + " \\\\"]
    rows = []
    for ts in TYPES.values():
        for t in ts:
            vals = [ov_cell(sub, pool_a=p, rule_a=r, task_a=t) for p in OV_POOLS for r in rules]
            rows.append(f"{rt.tex_name('simple', t)} & " + " & ".join(rt.num(None if v is None else v.share) for v in vals) + " \\\\")
    rows.append("\\midrule")
    rows.append("Chance & " + " & ".join(rt.num(sub[(sub.pool_a == p) & (sub.rule_a == r)].chance.mean())
                                         for p in OV_POOLS for r in rules) + " \\\\")
    write("design_overlap_task_stability", rt.tabular("@{}l" + "r" * (len(rules) * len(OV_POOLS)) + "@{}", header, rows))


def ov_tex(rule: str) -> str:
    """Rule name in the pair column of the rule tables, shortened to fit the page."""
    if rule in ZERO_SHOT:  # Causal (zero-shot) as Causal_0, so that the pair column fits the page
        return "Causal$_0$" + (" $-$ gen." if rule.endswith("nogen") else "")
    return RULES[rule][0].replace("Yang terms ", "Yang ")


LRP_RULES = ("T", "lrpT")


def zs_pair_shown(a: str, b: str) -> bool:
    """The rule tables compare a zero-shot rule (Sec. 5.6) only with the causal rankings, Selected and each other, and
    T and LRP-T (Sec. 5.7) only as LRP-T against T, D, Selected, Causal and Causal (zero-shot)."""
    if a in LRP_RULES or b in LRP_RULES:
        other = b if a == "lrpT" else a if b == "lrpT" else None
        return other in ("T", "D", "selected", "causal", "causalzs")
    if a not in ZERO_SHOT and b not in ZERO_SHOT:
        return True
    other = b if a in ZERO_SHOT else a
    return other in ("causal", "causalnogen", "selected") + ZERO_SHOT


def overlap() -> None:
    df = overlap_rows()
    table_overlap_stability(df)
    table_overlap_task_stability(df)
    for scope in SCOPES:  # one table per scope, each fits a page
        conds = ov_conditions(scope)
        rules_pairs = []
        for p in OV_POOLS:
            cs = [c for c in conds if c[0] == p]
            rules_pairs += [(scope, OV_POOLS[p], f"{ov_tex(a[2])} -- {ov_tex(b[2])}",
                             dict(pool_a=p, rule_a=a[2], pool_b=p, rule_b=b[2])) for i, a in enumerate(cs) for b in cs[i + 1:]
                            if zs_pair_shown(a[2], b[2])]
        ov_table(df, "rules", rules_pairs, f"design_overlap_rules_{scope.replace(' ', '')}")
    pool_pairs = []
    for scope in SCOPES:
        conds = ov_conditions(scope)
        for r in LADDER:
            cs = [c for c in conds if c[2] == r]
            pool_pairs += [(scope, RULES[r][0], f"{OV_POOLS[a[0]]} -- {OV_POOLS[b[0]]}",
                            dict(pool_a=a[0], rule_a=r, pool_b=b[0], rule_b=r)) for i, a in enumerate(cs) for b in cs[i + 1:]]
    ov_table(df, "pools", pool_pairs, "design_overlap_pools")
    scope_pairs = [("per task", OV_POOLS[p], RULES[r][0], dict(pool_a=p, rule_a=r))
                   for p in OV_POOLS for r in LADDER
                   if df[(df.kind == "scope") & (df.pool_a == p) & (df.rule_a == r) & df.n.isin(OV_SIZES)].share.notna().any()]
    ov_table(df, "scope", scope_pairs, "design_overlap_scope")
    table_overlap_tasks(df)
    for n in OV_SIZES:
        fig_overlap(df, n)
    fig_overlap_tasks(df)


def main() -> None:
    if "--overlap" in sys.argv:  # Sec. 7 only
        overlap()
        return
    scores = {fold: screen_scores(fold) for fold in (1, 2)}
    table_own()
    table_ladder()
    table_controls()
    table_types()
    table_causal_kinds()
    table_yang_tasks()
    table_selected(scores)
    table_selected_sizes()
    fig_published()
    fig_ladder()
    fig_controls()
    effects = feature_effects()
    fig_features(effects)
    corr = fig_screen({f: quick_scores(f, "additions") for f in (1, 2)} if QUICK else scores)
    fig_layers()
    fig_types()
    best = task_best()
    fig_task_best(best)
    if "--stage2" not in sys.argv:  # --stage2: Secs. 4-5 only (stage 3 being re-run)
        stage3()
        overlap()
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
