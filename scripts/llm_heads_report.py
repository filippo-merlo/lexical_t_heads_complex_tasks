"""E5 (paper/proposal-llm-heads.tex): accuracies of the LLM head sets beside E4's, no statistical tests.

Reads ``results/llm_heads/fold{1,2}/patching_{simple,simple-wrong}.part-*.csv``
(``scripts/llm_heads_patch.sh``) and, for the references, E4's ``results/heldout`` (both read with
``heldout_report.load``: task accuracy = mean over the five rule means of the choosing half).
Accuracy at the **cross-fitted alpha** as E4: in fold 1 at the alpha with the highest task
accuracy for the same task and set in fold 2, and the reverse (ties: smaller |alpha|, then the
positive one); a random draw ``random_{set}_{i}`` at the alpha of its set, E4's ``random{20,40}_i``
at that of ``cc20``/``cc40``. Tables and figure give the mean of the two folds.

Also: E4's ``fv20``, patched again here, against E4's own rows (must agree), and, for every
set at its cross-fitted alpha, the share of the patched items given the most frequent reply
(constant-answer check, ``answers/``).

Writes, per task, ``results/llm_heads/{accuracy_alpha,accuracy_crossfit,constant_answers}_{task}.csv``,
``table_{task}.md`` and the figure ``paper/figures/llm_heads_{task}.pdf`` (and a PNG beside the CSVs).
Tasks after the pilot have the ``all`` condition only; table and figure show the conditions present.

    .venv/bin/python scripts/llm_heads_report.py --task english-german
    .venv/bin/python scripts/llm_heads_report.py --task country-capital
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import heldout_report as hr  # noqa: E402
import results_tex as rt  # noqa: E402
from lth import RESULTS  # noqa: E402

OUT = RESULTS / "llm_heads"
SIZES = (20, 30, 40, 60, 80, "Nt")  # Nt: the size of the fold's full CC set (60, 80, Nt added 2026-11-07)
LLM = (("allJ", "all, J-head"), ("allL", "all, Logit-head"), ("batJ", "batched, J-head"), ("batL", "batched, Logit-head"))
COLOUR = {"J": rt.BLUE, "L": rt.ORANGE, "cc": rt.AQUA, "fv": rt.INK_2}
STYLE = {"all": "-", "bat": (0, (4, 2))}


def refs_of(n) -> tuple[str, str]:
    """The CC and FV references of a size (at N_t: E4's full CC set and FV-N_t)."""
    return ("cc", "fvNt") if n == "Nt" else (f"cc{n}", f"fv{n}")


def base_of(name: str) -> str:
    """The set whose cross-fitted alpha a set takes (itself, unless it is a random draw)."""
    if name.startswith("random_"):
        return name[len("random_"):].rsplit("_", 1)[0]
    m = re.match(r"random(20|40|Nt)_\d$", name)
    return hr.RANDOM_OF[m.group(1)] if m else name


def crossfit(c: pd.DataFrame) -> pd.DataFrame:
    best = hr.best_alpha(c)
    other = best.assign(fold=3 - best.fold).rename(columns={"set": "base"})
    sets = c[["fold", "family", "task", "set"]].drop_duplicates()
    sets = sets.assign(base=sets.set.map(base_of))
    chosen = sets.merge(other, on=["fold", "family", "task", "base"], how="inner")
    return chosen.merge(c, on=["fold", "family", "task", "set", "alpha"], how="inner")


def constant_answers(cf: pd.DataFrame) -> pd.DataFrame:
    """Per set at its cross-fitted alpha: replies, share of the most frequent one, accuracy (task means)."""
    want = {(r.fold, r.family, r.set, float(r.alpha)) for r in cf[cf.source == "task"].itertuples()}
    rows = {}
    for fold in (1, 2):
        for path in glob.glob(str(OUT / f"fold{fold}" / "answers" / "*.jsonl.gz")):
            with gzip.open(path, "rt") as f:
                for line in f:
                    a = json.loads(line)
                    if a["source"] != "task" or a["alpha"] == "":
                        continue
                    key = (fold, a["family"], a["heads"], float(a["alpha"]))
                    if key in want:
                        rows.setdefault(key, []).append(a["predicted"].strip().lower())
    out = []
    for (fold, family, s, alpha), replies in sorted(rows.items()):
        reply, n = Counter(replies).most_common(1)[0]
        out.append(dict(fold=fold, family=family, set=s, alpha=alpha, replies=len(replies),
                        distinct=len(set(replies)), top_reply=reply, top_share=round(n / len(replies), 3)))
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", default="english-german")
    args = parser.parse_args()
    task = args.task

    hr.HELD = OUT
    mine, base = hr.load()
    hr.HELD = RESULTS / "heldout"
    e4, e4_base = hr.load()
    mine, base = mine[mine.task == task], base[base.task == task]
    e4, e4_base = e4[e4.task == task], e4_base[e4_base.task == task]

    # check: E4's fv20 patched again here must give E4's accuracies
    a = mine[mine.set == "fv20"].merge(e4[e4.set == "fv20"], on=["fold", "family", "task", "set", "source", "alpha"],
                                       suffixes=("", "_e4"))
    diff = (a.accuracy - a.accuracy_e4).abs()
    b = base.merge(e4_base, on=["fold", "family", "task"], suffixes=("", "_e4"))
    print(f"fv20 again vs E4: {len(a)} points, max |difference| {diff.max():.4f}, "
          f"{(diff > 0).sum()} differ; unpatched: max |difference| "
          f"{(b.accuracy - b.accuracy_e4).abs().max():.4f}", flush=True)

    refs = ["e1", "cc", "cc20", "cc40", "fvNt", "fv40"] + [f"random{n}_{i}" for n in (20, 40) for i in range(5)]
    c = pd.concat([mine, e4[e4.set.isin(refs)]], ignore_index=True)
    cf = crossfit(c)
    c.to_csv(OUT / f"accuracy_alpha_{task}.csv", index=False)
    cf.to_csv(OUT / f"accuracy_crossfit_{task}.csv", index=False)
    ca = constant_answers(cf[~cf.set.str.startswith("random")])
    ca.to_csv(OUT / f"constant_answers_{task}.csv", index=False)

    # table: mean over folds; random draws of an LLM set averaged over the five draws
    s = cf.assign(set=cf.set.str.replace(r"^(random_?\w+?)_\d$", r"\1", regex=True))
    t = s.groupby(["family", "set", "source"]).accuracy.mean().unstack(["family", "source"])
    n_heads = cf.groupby("set").n_heads.agg(["min", "max"])
    alphas = cf[cf.source == "task"].groupby(["set", "family", "fold"]).alpha.first().unstack(["family", "fold"])
    heads = {f: {n: {tuple(h) for h in hs}
                 for _, n, hs in json.loads((OUT / f"fold{f}" / "heads.json").read_text())["simple"][task]["sets"]}
             for f in (1, 2)}
    order = []
    for n in SIZES:
        for name, _ in LLM:
            order.append(f"{name}{n}")
            if any(heads[f].get(f"{name}{n}") != heads[f].get(f"{name}{n}raw") for f in (1, 2)):
                order.append(f"{name}{n}raw")
        order += [f"random_{name}{n}" for name, _ in LLM]
        order += list(refs_of(n)) + ([f"random{n}"] if n in (20, 40) else [])
    order += ["e1"]
    lines = ["| set | heads | check 1 | ctrl | check 2 | ctrl | alpha c1 (f1/f2) | alpha c2 (f1/f2) |",
             "|---|---|---|---|---|---|---|---|"]
    for name in order:
        if name not in t.index:
            continue
        row = t.loc[name]
        get = lambda fam, src: row.get((fam, src), float("nan"))  # noqa: E731
        al = lambda fam: "/".join(  # noqa: E731
            "%g" % alphas.loc[name, (fam, f)] if (fam, f) in alphas.columns and pd.notna(alphas.loc[name, (fam, f)])
            else "-" for f in (1, 2)) if name in alphas.index else "-"
        key = name if name in n_heads.index else name + "_0"
        lo, hi = n_heads.loc[key] if key in n_heads.index else (float("nan"), float("nan"))
        nh = f"{lo:.0f}" if lo == hi else f"{lo:.0f}-{hi:.0f}"
        lines.append(f"| {name} | {nh} | {get('simple', 'task'):.2f} | {get('simple', 'control'):.2f} | "
                     f"{get('simple-wrong', 'task'):.2f} | {get('simple-wrong', 'control'):.2f} | {al('simple')} | "
                     f"{al('simple-wrong')} |")
    unp = base.groupby("family").accuracy.mean()
    lines.append(f"| unpatched | 0 | {unp.get('simple', float('nan')):.2f} | | {unp.get('simple-wrong', float('nan')):.2f} | | | |")
    (OUT / f"table_{task}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nconstant-answer check (task means, cross-fitted alpha): largest share of one reply",
          ca.top_share.max() if len(ca) else "-")
    print(ca.sort_values("top_share", ascending=False).head(8).to_string(index=False))

    figure(task, c, cf, base)


def figure(task: str, c: pd.DataFrame, cf: pd.DataFrame, base: pd.DataFrame) -> None:
    """Accuracy against alpha, mean of the folds: one row per size, columns check 1 and 2; the
    LLM sets (colour = readout, line = condition), CC and FV of the same size (task means); random
    draws of the LLM sets of that size as a grey band (check 1), unpatched dotted."""
    m = c.groupby(["family", "set", "source", "alpha"], as_index=False).accuracy.mean()
    llm = [(name, label) for name, label in LLM if m.set.str.match(rf"{name}\d").any()]
    nt = sorted(cf[cf.set == "cc"].n_heads.unique())  # size of the full CC set in each fold
    nt = f"{nt[0]:.0f}" if len(nt) == 1 else f"{nt[0]:.0f}-{nt[-1]:.0f}"
    fig, axes = plt.subplots(len(SIZES), 2, figsize=(6.3, 1.45 * len(SIZES) + 0.9), sharex=True, sharey=True)
    for j, family in enumerate(hr.CHECKS):
        for i, n in enumerate(SIZES):
            ax = axes[i, j]
            r = m[(m.family == family) & m.set.str.match(rf"random_(all|bat)[JL]{n}_\d$") & (m.source == "task")]
            if len(r):
                band = r.groupby("alpha").accuracy.agg(["min", "max"])
                ax.fill_between(band.index, band["min"], band["max"], color=rt.GREY, alpha=0.3, lw=0, zorder=1)
            for name, _ in llm:
                d = m[(m.family == family) & (m.set == f"{name}{n}") & (m.source == "task")]
                ax.plot(d.alpha, d.accuracy, color=COLOUR[name[-1]], ls=STYLE[name[:3]], lw=1.2, zorder=3)
            for ref, s_ref in zip(("cc", "fv"), refs_of(n)):
                d = m[(m.family == family) & (m.set == s_ref) & (m.source == "task")]
                ax.plot(d.alpha, d.accuracy, color=COLOUR[ref], lw=1.0, zorder=2)
            ax.axhline(base[base.family == family].accuracy.mean(), color=rt.INK_2, lw=0.7, ls=(0, (1, 2)), zorder=2)
            size = f"{nt} (full CC set's size)" if n == "Nt" else n
            ax.set_title(f"{hr.CHECKS[family]}, {size} heads", loc="left", fontsize=7)
            ax.set_ylim(-0.03, 1.03)
            rt.style_axes(ax)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\alpha$")
        ax.set_xticks([-10, -5, 0, 5, 10])
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")
    handles = [plt.Line2D([], [], color=COLOUR[name[-1]], ls=STYLE[name[:3]], lw=1.2) for name, _ in llm]
    labels = [f"LLM, {label}" for _, label in llm]
    handles += [plt.Line2D([], [], color=COLOUR["cc"]), plt.Line2D([], [], color=COLOUR["fv"]),
                plt.Rectangle((0, 0), 1, 1, color=rt.GREY, alpha=0.3, lw=0),
                plt.Line2D([], [], color=rt.INK_2, lw=0.7, ls=(0, (1, 2)))]
    labels += [f"CC, same size (at {nt}: the full CC set)", "FV, same size", "random, LLM sets' layers (range, check 1)", "unpatched"]
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (1.45 * len(SIZES) + 0.9)))
    fig.savefig(rt.FIGURES / f"llm_heads_{task}.pdf")
    fig.savefig(OUT / f"llm_heads_{task}.png", dpi=150)
    plt.close(fig)
    print(f"wrote {rt.FIGURES / f'llm_heads_{task}.pdf'}", flush=True)


if __name__ == "__main__":
    main()
