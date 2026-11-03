"""Tables and figures of the proposal's Results section, from results/ (no GPU).

Writes the tabulars ``paper/tables/*.tex``, each ``\\input`` inside a table float of
proposal-simple.tex or proposal-complex.tex, and the figures ``paper/figures/*.pdf``:
- ``e0_simple``, ``e0_complex``: strict accuracy per style and wording of the saved outputs
  (``answer_eval/accuracy.csv``, ``scripts/score_outputs.py``; chance from ``e0_accuracy.csv``);
  ``e0_simple_flexible``, ``e0_complex_flexible``: the same under the flexible score;
- ``heads_simple``, ``heads_complex``: head counts and overlaps (``measures.csv``,
  ``overlaps.csv``); ``head_counts_yang``: E1's Logit-head counts against those Yang et al.
  publish and against E1 in the setting of their code (``source_setting/heads.json``);
- ``effects_simple``, ``effects_simple_cross``, ``effects_complex``: the causal effect, the
  highest accuracy over alpha minus the unpatched accuracy, per head set and mean source
  (``patching_*.csv``). The accuracy at each alpha is first averaged over the source
  wordings of the mean (each wording is patched separately, as Yang et al.);
  ``effects_simple`` patches each style's heads with means of the same style,
  ``effects_simple_cross`` with means of the other style; ``effects_simple_wrong`` and
  ``effects_simple_wrong_cross`` the same for the second simple check (``simple-wrong``:
  wrongly answered 1-shot prompts patched, no control); ``effects_complex`` and
  ``effects_complex_wrong`` the two checks of the complex tasks' rule heads;
- ``heads_consensus``, ``effects_consensus``, ``effects_consensus_wrong``,
  ``effects_consensus_control`` (E1b, ``scripts/identify_consensus.py``): rule-style
  Logit-head and J-head sets of the heads that pass Definition 1 under at least 1, 3 or all 5
  instruction templates (``results/consensus_k1``, ``consensus_k3``, ``consensus``), against
  E1's; their causal effect with rule-style means in the two checks, and of the control mean
  in the first;
- ``layers_simple.pdf``, ``layers_complex.pdf``: lexical task heads per layer under each
  readout (``heads.json``), one figure per task family (Parts I and II of the proposal);
- ``copy_share_simple.pdf``, ``copy_share_complex.pdf``: copy share per task, style and
  readout (``measures.csv``);
- ``patching_simple.pdf``, ``patching_simple_wrong.pdf``, ``patching_complex.pdf``: accuracy
  against alpha;
- ``patching_consensus_k{1,3,5}.pdf``, ``patching_shared.pdf``, ``patching_noretrieval.pdf``: the
  same for the rule-style sets of E1b and its follow-ups (``results/consensus*``), both checks
  side by side; ``patching_pp_unshared.pdf``, ``patching_pp_split.pdf``: present--past, J-head
  subsets of its union set (``consensus_k1_shared``, ``_unshared``, ``pp_L21H9``, ``pp_3heads``).

Conditions without output yet (a causal check still to run) are marked in the tables and
left blank in the figures, so the script can be rerun as results arrive.

    .venv/bin/python scripts/results_tex.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import PAPER, READOUTS, RESULTS, simple  # noqa: E402

TABLES, FIGURES = PAPER / "tables", PAPER / "figures"

SIMPLE_FAMILIES = ("simple", "simple-wrong")  # the two simple-task causal checks (scripts/patch.py)
SIMPLE_NAMES = {"singular-plural": "singular--plural", "present-past": "present--past",
                "country-capital": "country--capital", "country-currency": "country--currency",
                "landmark-country": "landmark--country", "park-country": "park--country",
                "product-company": "product--company", "next_item": "next-item", "prev_item": "prev-item",
                "english-french": "English--French", "english-german": "English--German",
                "english-spanish": "English--Spanish"}
COMPLEX_NAMES = {"noun_class_agreement/easy": "Noun Class Agreement",
                 "lexical_category_inference/easy": "Lexical Category Inference",
                 "set_game/easy": "Set Game", "operator_func/easy": "Operator Function",
                 "tapatan/easy": "Tapatan"}
#: Complex rows: the five tasks, one family since the heads and checks were rebuilt under the
#: flexible score (2026-10-30), all read at the last prompt token.
COMPLEX_ROWS = [("complex", key) for key in COMPLEX_NAMES]
#: The two complex-task checks (scripts/patch.py): check 1, and check 2 with this family suffix.
WRONG = "-wrong"
STYLES = ("rule", "example", "combined")
SETS = ("logithead", "jhead", "jhead_only")
#: E1b: heads that pass Definition 1 under at least k of the five instruction templates.
CONSENSUS_K = (1, 3, 5)
MODEL_HEADS = 32 * 32  # Llama-3.1-8B-Instruct: 32 layers of 32 heads
#: Logit-head counts that Yang et al. publish for Llama-3.1-8B-Instruct, read off the bar
#: labels: Fig. 28 (instruction-based) and Fig. 20 (example-based), and the Instruct bars of
#: Fig. 79 and the "original" bars of Fig. 88, which name no style.
YANG_COUNTS = {
    "fig28": dict(zip(simple.TASKS, (23, 7, 26, 11, 31, 20, 25, 26, 35, 42, 29, 14, 11, 9))),
    "fig79": dict(zip(simple.TASKS, (17, 2, 18, 9, 27, 16, 21, 22, 28, 35, 25, 11, 9, 7))),
    "fig20": dict(zip(simple.TASKS, (12, 8, 24, 6, 15, 16, 14, 16, 20, 16, 10, 6, 7, 3))),
    "fig88": dict(zip(simple.TASKS, (9, 4, 15, 6, 13, 15, 12, 9, 16, 11, 6, 5, 6, 3))),
}
#: E1's Logit-head sets in the setting of Yang et al.'s code (attended padding, first-token
#: score), from the run of c86d907.
SOURCE_SETTING = RESULTS / "source_setting" / "heads.json"
# Tasks with a fixed label set, and their label fields (as scripts/e0_table.py).
LABEL_FIELDS = {"noun_class_agreement": "expected_label", "lexical_category_inference": "expected_answer"}

# The style of head_level_jlens/scripts/figures.py: blue for J-head, orange for Logit-head,
# grey for context; the third series (heads only J-head finds) takes the next slot, aqua.
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
# Present--past subsets of the J-head union set: blue (the union, as J-head), then magenta,
# green, yellow (dataviz validator: all pairs pass CVD and normal-vision separation).
MAGENTA, GREEN, YELLOW = "#e87ba4", "#008300", "#eda100"
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
GRIDLINE, AXIS_LINE = "#e1e0d9", "#c3c2b7"
SET_COLOUR = {"logithead": ORANGE, "jhead": BLUE, "jhead_only": AQUA}
SET_LABEL = {"logithead": "Logit-head", "jhead": "J-head", "jhead_only": "J-head only"}
CONTROL_STYLE = (0, (1, 1.5))

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "axes.edgecolor": AXIS_LINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.3,
        "lines.solid_capstyle": "round",
    }
)


def tex_name(family: str, task: str) -> str:
    if family in SIMPLE_FAMILIES:
        return SIMPLE_NAMES.get(task, task)
    return COMPLEX_NAMES[task]


def plot_name(family: str, task: str) -> str:
    return tex_name(family, task).replace("--", "–")


def num(x) -> str:
    """Two decimals, a typeset minus, ``--`` for a missing value."""
    if x is None or pd.isna(x):
        return "--"
    text = f"{x:.2f}"
    return "0.00" if text == "-0.00" else text.replace("-", "$-$")


def tabular(spec: str, header: list[str], rows: list[str]) -> str:
    return "\n".join([f"\\begin{{tabular}}{{{spec}}}", "\\toprule", *header, "\\midrule", *rows,
                      "\\bottomrule", "\\end{tabular}"]) + "\n"


def write(name: str, text: str) -> None:
    TABLES.mkdir(exist_ok=True)
    (TABLES / f"{name}.tex").write_text(f"% Generated by scripts/results_tex.py from results/.\n{text}")


# ------------------------------------------------------------------ data


def e0_strict() -> pd.DataFrame:
    """E0 accuracy: the strict score of the saved outputs (answer_eval/accuracy.csv), with
    chance from e0_accuracy.csv (it depends on the items only)."""
    acc = pd.read_csv(RESULTS / "answer_eval" / "accuracy.csv")
    acc = acc.rename(columns={"study_strict": "accuracy"})[["family", "task", "wording", "n", "accuracy",
                                                            "flexible"]]
    chance = pd.read_csv(RESULTS / "e0_accuracy.csv").drop_duplicates(["family", "task"])
    chance = chance[chance.family != "reasoning"][["family", "task", "chance"]]
    return acc.merge(chance, on=["family", "task"], how="left")


def missing(e0: pd.DataFrame, family: str, task: str) -> str:
    """Why a complex-task check has no rows: the second check of a task answered correctly on
    every item under its original rules prompt (flexible score) has nothing to patch."""
    orig = e0[(e0.family == "complex") & (e0.task == task) & (e0.wording == "orig")]
    if family.endswith(WRONG) and len(orig) and orig.flexible.iloc[0] == 1:
        return "no item answered wrongly"
    return "not yet run"


def load() -> dict:
    frames = [pd.read_csv(f) for f in sorted(RESULTS.glob("patching_*.csv"))]
    patching = pd.concat([f for f in frames if "mean_wording" in f], ignore_index=True)
    patching, cross = wording_curves(patching)
    return {"e0": e0_strict(), "measures": pd.read_csv(RESULTS / "measures.csv"),
            "overlaps": pd.read_csv(RESULTS / "overlaps.csv"), "patching": patching, "cross": cross,
            "heads": json.loads((RESULTS / "heads.json").read_text())}


def wording_curves(patching: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accuracy at each alpha averaged over the source wordings of the mean.

    Returns the table with each style's heads patched with means of the same style (the
    complex tasks have only those), and the simple rows patched with means of the other
    style (baseline rows included).
    """
    patching = patching.assign(mean_style=patching.mean_style.fillna(patching["style"]))
    base = patching[patching.source == "none"]
    keys = ["family", "task", "style", "heads", "source", "mean_style", "alpha"]
    curve = (patching[patching.source != "none"]
             .groupby(keys, as_index=False).agg(accuracy=("accuracy", "mean"), n_prompts=("n_prompts", "first")))
    same = curve[curve.mean_style == curve["style"]].drop(columns="mean_style")
    other = curve[curve.mean_style != curve["style"]].drop(columns="mean_style")
    return (pd.concat([base, same], ignore_index=True),
            pd.concat([base[base.family.isin(SIMPLE_FAMILIES)], other], ignore_index=True))


def baselines(patching: pd.DataFrame) -> dict:
    """``(family, task)`` -> (patched items, unpatched accuracy)."""
    rows = patching[patching.source == "none"]
    return {(r.family, r.task): (int(r.n_prompts), r.accuracy) for r in rows.itertuples()}


def effects(patching: pd.DataFrame) -> dict:
    """``(family, task, style, head set, source)`` -> highest accuracy over alpha minus baseline."""
    base = baselines(patching)
    rows = patching[patching.source != "none"]
    best = rows.groupby(["family", "task", "style", "heads", "source"]).accuracy.max()
    return {key: value - base[key[:2]][1] for key, value in best.items()}


def constant_answer(family: str, task: str) -> float | None:
    """Accuracy on the patched items of answering every one with the most frequent label,
    from the baseline answers of ``results/answers/patching_{family}.jsonl.gz``."""
    if task.split("/")[0] not in LABEL_FIELDS:
        return None
    with gzip.open(RESULTS / "answers" / f"patching_{family}.jsonl.gz", "rt") as f:
        lines = [json.loads(line) for line in f]
    labels = Counter(str(x["expected"]) for x in lines if x["source"] == "none" and x["task"] == task)
    return max(labels.values()) / sum(labels.values())


# ------------------------------------------------------------------ tables


def table_e0_simple(e0: pd.DataFrame, score: str = "accuracy", name: str = "e0_simple") -> None:
    e0 = e0[e0.family == "simple"]
    rows = []
    for task in simple.TASKS:
        t = e0[e0.task == task].set_index("wording")
        rule = t.loc[list(simple.WORDINGS["rule"]), score]
        shots = [num(t.loc[w, score]) for w in simple.WORDINGS["example"]]
        rows.append(" & ".join([tex_name("simple", task), str(int(t.n.iloc[0])), num(rule.mean()),
                                f"{num(rule.min())}--{num(rule.max())}", *shots,
                                num(t.loc[simple.ZERO, score])]) + r" \\")
    header = [rf" & & \multicolumn{{2}}{{c}}{{Rule ({len(simple.INSTRUCTIONS)} templates)}} & \multicolumn{{5}}{{c}}{{Example (shots)}} & \\",
              r"\cmidrule(lr){3-4}\cmidrule(lr){5-9}",
              "Task & Items & mean & range & " + " & ".join(str(n) for n in simple.SHOTS) + r" & Zero-shot \\"]
    write(name, tabular("@{}lrcccccccc@{}", header, rows))


def table_e0_complex(e0: pd.DataFrame, score: str = "accuracy", name: str = "e0_complex") -> None:
    rows = []
    for key in ["noun_class_agreement/easy", "lexical_category_inference/easy", "set_game/easy",
                "operator_func/easy", "tapatan/easy"]:
        t = e0[(e0.family == "complex") & (e0.task == key)].set_index("wording")
        # Wordings still running are left out of the mean and range, and shown as --.
        acc = t[score].reindex(["orig", *(f"w{i}" for i in range(1, 7)), "examples", "combined", "bare"])
        wordings = acc[[f"w{i}" for i in range(1, 7)]].dropna()
        span = f"{num(wordings.min())}--{num(wordings.max())}" if len(wordings) else "--"
        rows.append(" & ".join([COMPLEX_NAMES[key], str(int(t.n.iloc[0])), num(acc["orig"]),
                                num(wordings.mean()), span, num(acc["examples"]), num(acc["combined"]),
                                num(acc["bare"]), num(t.chance.iloc[0])]) + r" \\")
    header = [r" & & \multicolumn{3}{c}{Rule} & & & No task & \\", r"\cmidrule(lr){3-5}",
              r"Task & Items & orig. & w1--6 & range & Examples & Combined & info & Chance \\"]
    write(name, tabular("@{}lrccccccc@{}", header, rows))


def head_cells(measures, overlaps, family, task, style) -> list[str]:
    """|P+|, |H^L|, |H^J| and |H^L ∩ H^J| for one style; dashes when P+ is empty."""
    m = measures[(measures.family == family) & (measures.task == task) & (measures["style"] == style)]
    if m.empty or m.n_prompts.iloc[0] == 0:
        return ["0" if not m.empty else "--", "--", "--", "--"]
    n = m.set_index("readout").n_heads
    o = overlaps[(overlaps.family == family) & (overlaps.task == task) & (overlaps.a == f"{style}/logithead")
                 & (overlaps.b == f"{style}/jhead")]
    return [str(int(m.n_prompts.iloc[0])), str(int(n["logithead"])), str(int(n["jhead"])), str(int(o.n_both.iloc[0]))]


def shared_cells(overlaps, family, task) -> list[str]:
    """Heads shared by the rule and example sets, per readout."""
    cells = []
    for readout in READOUTS:
        o = overlaps[(overlaps.family == family) & (overlaps.task == task) & (overlaps.a == f"rule/{readout}")
                     & (overlaps.b == f"example/{readout}")]
        cells.append("--" if o.empty or o.n_b.iloc[0] == 0 else str(int(o.n_both.iloc[0])))
    return cells


def table_heads(measures: pd.DataFrame, overlaps: pd.DataFrame) -> None:
    rows = []
    for task in simple.TASKS:
        cells = [c for style in ("rule", "example") for c in head_cells(measures, overlaps, "simple", task, style)[1:]]
        rows.append(" & ".join([tex_name("simple", task), *cells, *shared_cells(overlaps, "simple", task)]) + r" \\")
    header = [r" & \multicolumn{3}{c}{Rule} & \multicolumn{3}{c}{Example} & \multicolumn{2}{c}{Rule $\cap$ example} \\",
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
              r"Task & L & J & L$\cap$J & L & J & L$\cap$J & L & J \\"]
    write("heads_simple", tabular("@{}lcccccccc@{}", header, rows))

    rows = []
    for family, task in COMPLEX_ROWS:
        cells = [c for style in STYLES for c in head_cells(measures, overlaps, family, task, style)]
        rows.append(" & ".join([tex_name(family, task), *cells, *shared_cells(overlaps, family, task)]) + r" \\")
    header = [r" & \multicolumn{4}{c}{Rule} & \multicolumn{4}{c}{Example} & \multicolumn{4}{c}{Combined}"
              r" & \multicolumn{2}{c}{Rule $\cap$ ex.} \\",
              r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}\cmidrule(lr){14-15}",
              r"Task & $|P^+|$ & L & J & L$\cap$J & $|P^+|$ & L & J & L$\cap$J & $|P^+|$ & L & J & L$\cap$J & L & J \\"]
    write("heads_complex", tabular("@{}l" + "c" * 14 + "@{}", header, rows))


def table_head_counts(heads: dict) -> None:
    """Logit-head counts of E1 against those Yang et al. publish (YANG_COUNTS), and against
    E1 in the setting of their code (SOURCE_SETTING)."""
    source = json.loads(SOURCE_SETTING.read_text())["simple"]
    rows, sums = [], Counter()
    for task in simple.TASKS:
        cells = []
        for style, figures in (("rule", ("fig28", "fig79")), ("example", ("fig20", "fig88"))):
            values = [YANG_COUNTS[f][task] for f in figures]
            values += [len(h[task][style]["logithead"]["heads"]) for h in (source, heads["simple"])]
            cells += values
            sums.update({(style, i): v for i, v in enumerate(values)})
        rows.append(" & ".join([tex_name("simple", task), *map(str, cells)]) + r" \\")
    rows.append(r"\midrule")
    rows.append(" & ".join(["Sum", *(str(sums[(s, i)]) for s in ("rule", "example") for i in range(4))]) + r" \\")
    header = [r" & \multicolumn{4}{c}{Rule} & \multicolumn{4}{c}{Example} \\",
              r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
              r"Task & Fig.~28 & Fig.~79 & Source & E1 & Fig.~20 & Fig.~88 & Source & E1 \\"]
    write("head_counts_yang", tabular("@{}lcccccccc@{}", header, rows))


def effect_rows(patching, effect, family, task, styles) -> list[str]:
    base = baselines(patching)
    if (family, task) not in base:
        return [f"{tex_name(family, task)} & \\multicolumn{{8}}{{l}}{{\\textit{{not yet run}}}} \\\\"]
    n, accuracy = base[(family, task)]
    rows, first = [], True
    for style in styles:
        cells = [num(effect.get((family, task, style, s, source))) for source in ("task", "control") for s in SETS]
        if all(c == "--" for c in cells):
            continue
        lead = [tex_name(family, task) if first else "", style, num(accuracy) if first else ""]
        rows.append(" & ".join(lead + cells) + r" \\")
        first = False
    return rows


def table_effects(patching: pd.DataFrame, cross: pd.DataFrame, e0: pd.DataFrame) -> None:
    group = r"\multicolumn{3}{c}{Task mean} & \multicolumn{3}{c}{Control mean} \\"
    sets = r"L & J & J only & L & J & J only \\"
    for family in SIMPLE_FAMILIES:
        stem = "effects_" + family.replace("-", "_")
        for name, frame, column in ((stem, patching, "Style"), (stem + "_cross", cross, "Heads")):
            effect = effects(frame)
            rows = []
            for task in simple.TASKS:
                rows += effect_rows(frame, effect, family, task, ("rule", "example"))
            header = [r" & & & " + group, r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}", f"Task & {column} & Baseline & " + sets]
            write(name, tabular("@{}llccccccc@{}", header, rows))

    # The complex tasks: rule heads only, so no style column; both checks have a control
    # (check 2 since 2026-10-31).
    effect, base = effects(patching), baselines(patching)
    for suffix, sources in (("", ("task", "control")), (WRONG, ("task", "control"))):
        rows = []
        for family, task in COMPLEX_ROWS:
            family += suffix
            if (family, task) not in base:
                rows.append(f"{tex_name(family, task)} & \\multicolumn{{{2 + 3 * len(sources)}}}{{l}}"
                            f"{{\\textit{{{missing(e0, family, task)}}}}} \\\\")
                continue
            n, accuracy = base[(family, task)]
            cells = [num(effect.get((family, task, "rule", s, source))) for source in sources for s in SETS]
            rows.append(" & ".join([tex_name(family, task), str(n), num(accuracy), *cells]) + r" \\")
        if len(sources) == 2:
            header = [r" & & & " + group, r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}", r"Task & Items & Baseline & " + sets]
        else:
            header = [r" & & & \multicolumn{3}{c}{Task mean} \\", r"\cmidrule(lr){4-6}",
                      r"Task & Items & Baseline & L & J & J only \\"]
        write("effects_complex" + suffix.replace("-", "_"), tabular("@{}lrc" + "c" * 3 * len(sources) + "@{}", header, rows))


def table_consensus(patching: pd.DataFrame) -> None:
    """E1b against E1: rule-style Logit-head and J-head sets from the heads that pass under at
    least k of the five instruction templates (k = 1, 3, 5), their sizes and causal effects
    with rule-style means."""
    dirs = {k: RESULTS / ("consensus" if k == 5 else f"consensus_k{k}") for k in CONSENSUS_K}
    if not all((d / "measures.csv").exists() for d in dirs.values()):
        return
    readouts = ("logithead", "jhead")
    sets = " & ".join(rf"\multicolumn{{4}}{{c}}{{{label}}}" for label in ("L", "J"))
    columns = "Task" + " & E1 & $k=1$ & $k=3$ & $k=5$" * 2

    def share(n: float, digits: int = 0) -> str:  # a head count and its share of the model's heads, in %
        return f"{n:.{digits}f} ({100 * n / MODEL_HEADS:.1f})"

    head_sets = {"E1": json.loads((RESULTS / "heads.json").read_text())["simple"],
                 **{k: json.loads((d / "heads.json").read_text())["simple"] for k, d in dirs.items()}}
    rows = []
    for task in simple.TASKS:
        cells = [share(len(head_sets[key][task]["rule"][readout]["heads"]))
                 for readout in readouts for key in ("E1", *CONSENSUS_K)]
        rows.append(" & ".join([tex_name("simple", task), *cells]) + r" \\")
    mean, distinct = [], []
    for readout in readouts:
        for key in ("E1", *CONSENSUS_K):
            per_task = [{(l, h) for l, h, _ in head_sets[key][task]["rule"][readout]["heads"]}
                        for task in simple.TASKS]
            mean.append(share(sum(map(len, per_task)) / len(per_task), 1))
            distinct.append(share(len(set().union(*per_task))))
    rows += [r"\midrule", " & ".join(["Mean per task", *mean]) + r" \\",
             " & ".join(["Distinct, all tasks", *distinct]) + r" \\"]
    header = [f" & {sets} \\\\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}", columns + r" \\"]
    write("heads_consensus", tabular("@{}lcccccccc@{}", header, rows))

    curves = {}
    for k, d in dirs.items():
        frames = [pd.read_csv(f) for f in sorted(d.glob("patching_*.csv"))]
        if not frames:
            return
        curves[k], _ = wording_curves(pd.concat(frames, ignore_index=True))
    eff = {"E1": effects(patching), **{k: effects(c) for k, c in curves.items()}}
    items = baselines(curves[1])
    for name, family, source in (("effects_consensus", "simple", "task"),
                                 ("effects_consensus_wrong", "simple-wrong", "task"),
                                 ("effects_consensus_control", "simple", "control")):
        wrong = family == "simple-wrong"
        rows = []
        for task in simple.TASKS:
            cells = [num(eff[key].get((family, task, "rule", readout, source)))
                     for readout in readouts for key in ("E1", *CONSENSUS_K)]
            lead = [tex_name("simple", task)] + ([str(items[(family, task)][0])] if wrong else [])
            rows.append(" & ".join(lead + cells) + r" \\")
        lead = " &" * (2 if wrong else 1)
        first = 3 if wrong else 2
        rules = rf"\cmidrule(lr){{{first}-{first + 3}}}\cmidrule(lr){{{first + 4}-{first + 7}}}"
        header = [f"{lead} {sets} \\\\", rules,
                  columns.replace("Task", "Task & Items" if wrong else "Task", 1) + r" \\"]
        write(name, tabular("@{}l" + ("r" if wrong else "") + "c" * 8 + "@{}", header, rows))


# ------------------------------------------------------------------ figures


def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure_layers(heads: dict, family: str) -> None:
    """Heads per layer, summed over the task and style sets of one family."""
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for readout in READOUTS:
        counts = [0] * 32
        for styles in heads.get(family, {}).values():
            for sets in styles.values():
                for layer, _, _ in sets[readout]["heads"]:
                    counts[layer] += 1
        ax.plot(range(32), counts, color=SET_COLOUR[readout], label=SET_LABEL[readout])
    ax.set_xlabel("layer")
    ax.set_xticks([0, 8, 16, 24, 31])
    style_axes(ax)
    ax.set_ylabel("heads")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / f"layers_{family}.pdf")
    plt.close(fig)


def figure_copy_share(measures: pd.DataFrame, family: str) -> None:
    tasks = simple.TASKS if family == "simple" else list(COMPLEX_NAMES)
    styles = STYLES[:2] if family == "simple" else STYLES
    fig, axes = plt.subplots(1, len(styles), figsize=(2.1 * len(styles), 0.6 + 0.2 * len(tasks)), sharey=True)
    for ax, style in zip(axes, styles):
        for pos, task in enumerate(tasks):
            m = measures[(measures.family == family) & (measures.task == task) & (measures["style"] == style)]
            for readout, dy in zip(READOUTS, (-0.17, 0.17)):
                value = m[m.readout == readout].copy_share
                if len(value) and pd.notna(value.iloc[0]):
                    ax.plot(value.iloc[0], pos + dy, "o", color=SET_COLOUR[readout], ms=4, mec=SURFACE, mew=0.6)
        ax.set_title(f"{style} prompts", loc="left")
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("copy share")
        ax.grid(axis="y", visible=False)
        style_axes(ax)
    axes[0].set_yticks(range(len(tasks)), [plot_name(family, t) for t in tasks])
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=SET_COLOUR[r], marker="o", lw=0, ms=4) for r in READOUTS]
    fig.legend(handles, [SET_LABEL[r] for r in READOUTS], loc="upper right", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.18 / (0.6 + 0.2 * len(tasks))))
    fig.savefig(FIGURES / f"copy_share_{family}.pdf")
    plt.close(fig)


def curve_panel(ax, patching, family, task, style, reference=None, sets=SETS) -> bool:
    """Accuracy against alpha for one condition; False if it has no patched rows."""
    rows = patching[(patching.family == family) & (patching.task == task) & (patching["style"] == style)]
    if rows.empty:
        return False
    base = patching[(patching.family == family) & (patching.task == task) & (patching.source == "none")]
    ax.axhline(base.accuracy.iloc[0], color=GREY, lw=0.9, ls=(0, (4, 2)), zorder=1)
    if reference is not None:
        ax.axhline(reference, color=INK_2, lw=0.8, ls=(0, (7, 2, 1.5, 2)), zorder=1)
    for s in sets:
        for source in ("task", "control"):
            r = rows[(rows.heads == s) & (rows.source == source)].sort_values("alpha")
            if len(r):
                ax.plot(r.alpha, r.accuracy, color=SET_COLOUR[s], lw=1.3 if source == "task" else 0.9,
                        ls="-" if source == "task" else CONTROL_STYLE, marker="o" if len(r) < 6 else None,
                        ms=2.5, zorder=3 if source == "task" else 2)
    return True


def curve_legend(fig, constant: bool = False, control: bool = True, sets=SETS) -> None:
    handles = [plt.Line2D([], [], color=SET_COLOUR[s]) for s in sets]
    labels = [f"{SET_LABEL[s]}, task mean" for s in sets]
    if control:
        handles.append(plt.Line2D([], [], color=INK_2, ls=CONTROL_STYLE, lw=0.9))
        labels.append("control mean (same colours)")
    handles.append(plt.Line2D([], [], color=GREY, ls=(0, (4, 2)), lw=0.9))
    labels.append("unpatched")
    if constant:
        handles.append(plt.Line2D([], [], color=INK_2, ls=(0, (7, 2, 1.5, 2)), lw=0.8))
        labels.append("constant answer")
    fig.legend(handles, labels, loc="upper center", ncol=3)


def empty_panel(ax, text: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=INK_2, fontsize=7)


def figure_patching(patching: pd.DataFrame, e0: pd.DataFrame) -> None:
    for family in SIMPLE_FAMILIES:
        if not (patching.family == family).any():
            continue
        fig, axes = plt.subplots(7, 4, figsize=(6.3, 8.4), sharex=True, sharey=True)
        for i, task in enumerate(simple.TASKS):
            for j, style in enumerate(("rule", "example")):
                ax = axes[i // 2, 2 * (i % 2) + j]
                curve_panel(ax, patching, family, task, style)
                ax.set_title(f"{plot_name(family, task)}, {style}", loc="left", fontsize=7)
                ax.set_ylim(-0.03, 1.03)
                style_axes(ax)
        for ax in axes[-1]:
            ax.set_xlabel(r"$\alpha$")
            ax.set_xticks([-10, -5, 0, 5, 10])
        for ax in axes[:, 0]:
            ax.set_ylabel("accuracy")
        curve_legend(fig, control=family == "simple")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(FIGURES / f"patching_{family.replace('-', '_')}.pdf")
        plt.close(fig)

    # The two checks head the columns and tasks the rows (rule heads only); empty panels
    # keep their own y axis, so the axes are not shared.
    checks = (("", "no task information"), (WRONG, "rules prompt, answered wrongly"))
    fig, axes = plt.subplots(len(COMPLEX_ROWS), len(checks), figsize=(6.3, 9.0))
    ran = set(zip(patching.family, patching.task))
    for i, (found, task) in enumerate(COMPLEX_ROWS):
        for j, (suffix, _) in enumerate(checks):
            ax, family = axes[i, j], found + suffix
            reference = constant_answer(family, task)
            if (family, task) not in ran:
                empty_panel(ax, missing(e0, family, task))
            elif not curve_panel(ax, patching, family, task, "rule", reference):
                empty_panel(ax, "no head set")
            else:
                ax.set_xticks([-10, -5, 0, 5, 10])
                ax.set_ylim(-0.03, 1.03)
                ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"] if j == 0 else [])
                style_axes(ax)
        axes[i, 0].set_ylabel(COMPLEX_NAMES[task].replace(" ", "\n", 1), fontsize=7)
    for ax, (_, title) in zip(axes[0], checks):
        ax.set_title(f"patched: {title}", loc="left")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\alpha$")
    fig.supylabel("accuracy", fontsize=8, color=INK_2)
    curve_legend(fig, constant=True)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(FIGURES / "patching_complex.pdf")
    plt.close(fig)


def rule_curves(directory: Path) -> pd.DataFrame | None:
    """The rule-style sets of one E1b result folder, patched with rule-style means."""
    frames = [pd.read_csv(f) for f in sorted(directory.glob("patching_*.csv"))]
    if not frames:
        return None
    curves, _ = wording_curves(pd.concat(frames, ignore_index=True))
    return curves


def figure_rule_sets(directory: Path, name: str, sets) -> None:
    """Accuracy against alpha for the rule-style sets of E1b or a follow-up: each task's two
    checks side by side (first check with its control mean, second without)."""
    curves = rule_curves(directory)
    if curves is None:
        return
    fig, axes = plt.subplots(7, 4, figsize=(6.3, 8.4), sharex=True, sharey=True)
    for i, task in enumerate(simple.TASKS):
        for j, (family, check) in enumerate((("simple", "check 1"), ("simple-wrong", "check 2"))):
            ax = axes[i // 2, 2 * (i % 2) + j]
            if not curve_panel(ax, curves, family, task, "rule", sets=sets):
                empty_panel(ax, "no head set")
            ax.set_title(f"{plot_name('simple', task)}, {check}", loc="left", fontsize=7)
            ax.set_ylim(-0.03, 1.03)
            style_axes(ax)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\alpha$")
        ax.set_xticks([-10, -5, 0, 5, 10])
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")
    curve_legend(fig, sets=sets)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES / f"patching_{name}.pdf")
    plt.close(fig)


def figure_present_past(name: str, series) -> None:
    """Present--past, J-head subsets of the union set with rule-style means: one curve per
    subset ``(label, result folder, colour)``, the two checks side by side."""
    loaded = [(label, rule_curves(RESULTS / folder), colour) for label, folder, colour in series]
    if any(c is None for _, c, _ in loaded):
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3), sharey=True)
    checks = (("simple", "check 1: zero-shot prompts"), ("simple-wrong", "check 2: 1-shot prompts answered wrongly"))
    for ax, (family, title) in zip(axes, checks):
        base = None
        for label, curves, colour in loaded:
            rows = curves[(curves.family == family) & (curves.task == "present-past") & (curves.heads == "jhead")]
            base = curves[(curves.family == family) & (curves.task == "present-past") & (curves.source == "none")]
            for source in ("task", "control"):
                r = rows[rows.source == source].sort_values("alpha")
                if len(r):
                    ax.plot(r.alpha, r.accuracy, color=colour, lw=1.3 if source == "task" else 0.9,
                            ls="-" if source == "task" else CONTROL_STYLE, zorder=3 if source == "task" else 2)
        ax.axhline(base.accuracy.iloc[0], color=GREY, lw=0.9, ls=(0, (4, 2)), zorder=1)
        ax.set_title(title, loc="left", fontsize=7)
        ax.set_xlabel(r"$\alpha$")
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_ylim(-0.03, 1.03)
        style_axes(ax)
    axes[0].set_ylabel("accuracy")
    handles = [plt.Line2D([], [], color=c) for _, _, c in loaded]
    labels = [label for label, _, _ in loaded]
    handles += [plt.Line2D([], [], color=INK_2, ls=CONTROL_STYLE, lw=0.9), plt.Line2D([], [], color=GREY, ls=(0, (4, 2)), lw=0.9)]
    labels += ["control mean (same colours)", "unpatched"]
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    fig.savefig(FIGURES / f"patching_{name}.pdf")
    plt.close(fig)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    data = load()
    table_e0_simple(data["e0"])
    table_e0_simple(data["e0"], "flexible", "e0_simple_flexible")
    table_e0_complex(data["e0"])
    table_e0_complex(data["e0"], "flexible", "e0_complex_flexible")
    table_heads(data["measures"], data["overlaps"])
    table_head_counts(data["heads"])
    table_effects(data["patching"], data["cross"], data["e0"])
    table_consensus(data["patching"])
    FIGURES.mkdir(exist_ok=True)
    for family in ("simple", "complex"):
        figure_layers(data["heads"], family)
        figure_copy_share(data["measures"], family)
    figure_patching(data["patching"], data["e0"])
    readouts = ("logithead", "jhead")
    for k in CONSENSUS_K:
        figure_rule_sets(RESULTS / ("consensus" if k == 5 else f"consensus_k{k}"), f"consensus_k{k}", readouts)
    figure_rule_sets(RESULTS / "consensus_k1_shared", "shared", ("jhead",))
    figure_rule_sets(RESULTS / "consensus_k1_noretrieval", "noretrieval", readouts)
    figure_present_past("pp_unshared", (("union, 19 heads", "consensus_k1", BLUE),
                                        ("shared, 15 heads", "consensus_k1_shared", YELLOW),
                                        ("unshared, 4 heads", "consensus_k1_unshared", MAGENTA)))
    figure_present_past("pp_split", (("unshared, 4 heads", "consensus_k1_unshared", MAGENTA),
                                     ("L21H9", "pp_L21H9", GREEN),
                                     ("L19H10, L19H24, L20H28", "pp_3heads", YELLOW)))
    print(f"wrote {TABLES} and {FIGURES}")


if __name__ == "__main__":
    main()
