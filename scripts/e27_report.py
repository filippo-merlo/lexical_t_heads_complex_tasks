"""E27 report: do the causal heads found with instructions and with demonstrations overlap?

Davidson et al. (2025, Sec. 3.3, Fig. 3) take the 20 heads with the largest CIE averaged over
tasks, once with instruction prompts and once with shuffled 10-shot demonstrations, and find 4
of 20 shared on Llama-3.1-8B-Instruct; shared heads lie in later layers. Here the same
comparison on our 14 simple tasks:

- instruction CIE: ``scripts/davidson_fv.py`` (mean over its three kinds of uninformative
  instruction; all items: ``results/davidson_fv_all14_30``; held-out halves: E4's
  ``results/heldout/fold{1,2}/fv``, whose top N are the Causal sets of head-sets-simple);
- demonstration CIE: ``scripts/e27_demo_cie.py`` (``results/e27_demo_cie/{all,A,B}``).

For each item set (all items, half A, half B) and scope (one set over the tasks: CIE averaged
over tasks, Davidson et al.'s; per task: each task's own CIE), the top N heads of the two
rankings, N in 20-120: heads in both, against chance for two random sets of N heads (N^2/1024)
and for two random sets with the same number of heads in each layer (sum over layers of
n_inst(l) n_demo(l) / 32). Also the rank correlation of the two scores over all 1024 heads,
and, at 20 heads, the mean layer of the heads found only with instructions, only with
demonstrations, and with both.

Writes ``results/e27_demo_cie/{report.md,overlap.csv,layers.csv}`` and the figures
``e27_heads.pdf`` (Davidson et al.'s Fig. 3 layout: the top 20 on all items) and
``e27_overlap.pdf`` there.

    .venv/bin/python scripts/e27_report.py
"""

from __future__ import annotations

import json
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
from e13_sets import top  # noqa: E402
from e21_report import md  # noqa: E402
from lth import RESULTS, simple  # noqa: E402

OUT = RESULTS / "e27_demo_cie"
KINDS = ("equiprobable", "wikitext", "other_task")
INSTRUCTION = {"all": RESULTS / "davidson_fv_all14_30", "A": RESULTS / "heldout" / "fold1" / "fv",
               "B": RESULTS / "heldout" / "fold2" / "fv"}
SIZES = (20, 40, 60, 80, 100, 120)
N_LAYERS = N_HEADS = 32
DAVIDSON_SHARED = 4  # of 20, Llama-3.1-8B-Instruct (Sec. 3.3, Fig. 3D)


def scores(split: str) -> tuple[dict, dict]:
    """Per-task CIE ``[n_layers * n_heads]`` (instruction: mean over kinds), and ``"pooled"``: mean over tasks."""
    ic = np.load(INSTRUCTION[split] / "cie.npz")
    dc = np.load(OUT / split / "cie.npz")
    inst = {t: np.mean([ic[f"{t}/{k}"] for k in KINDS], axis=0).ravel() for t in simple.TASKS}
    demo = {t: dc[t].ravel() for t in simple.TASKS}
    for d in (inst, demo):
        d["pooled"] = np.mean([d[t] for t in simple.TASKS], axis=0)
    return inst, demo


def layers(heads) -> np.ndarray:
    return np.bincount([l for l, _ in heads], minlength=N_LAYERS)


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def check_causal_sets(inst_a: dict, inst_b: dict) -> int:
    """The held-out instruction rankings reproduce head-sets-simple's Causal sets (E13)."""
    n = 0
    for fold, inst in ((1, inst_a), (2, inst_b)):
        sets = json.loads((RESULTS / "e13_capped" / f"fold{fold}" / "heads.json").read_text())["simple"]
        for t in simple.TASKS:
            have = {name: hs for _, name, hs in sets[t]["sets"]}
            for size in SIZES:
                assert top(np.argsort(-inst[t], kind="stable"), size) == have[f"causal{size}"], (fold, t, size)
                assert top(np.argsort(-inst["pooled"], kind="stable"), size) == have[f"shared_causal{size}"]
                n += 2
    return n


def main() -> None:
    rows, lay, corr, both = [], [], [], {}
    for split in ("all", "A", "B"):
        inst, demo = scores(split)
        both[split] = (inst, demo)
        for scope in ["pooled"] + list(simple.TASKS):
            io = np.argsort(-inst[scope], kind="stable")
            do = np.argsort(-demo[scope], kind="stable")
            corr.append(dict(split=split, scope=scope, rank_corr=rank_corr(inst[scope], demo[scope])))
            for n in SIZES:
                a, b = top(io, n), top(do, n)
                shared = {tuple(h) for h in a} & {tuple(h) for h in b}
                rows.append(dict(split=split, scope=scope, n=n, shared=len(shared),
                                 chance=n * n / (N_LAYERS * N_HEADS),
                                 chance_layer=float((layers(a) * layers(b)).sum() / N_HEADS)))
                if n == 20:
                    ia, ib = {tuple(h) for h in a}, {tuple(h) for h in b}
                    lay.append(dict(split=split, scope=scope,
                                    instruction_only=np.mean([l for l, _ in ia - ib]) if ia - ib else np.nan,
                                    demonstration_only=np.mean([l for l, _ in ib - ia]) if ib - ia else np.nan,
                                    shared=np.mean([l for l, _ in shared]) if shared else np.nan,
                                    instruction_all=np.mean([l for l, _ in ia]),
                                    demonstration_all=np.mean([l for l, _ in ib])))
    ov, ly, cr = pd.DataFrame(rows), pd.DataFrame(lay), pd.DataFrame(corr)
    ov.to_csv(OUT / "overlap.csv", index=False)
    ly.to_csv(OUT / "layers.csv", index=False)
    cr.to_csv(OUT / "rank_corr.csv", index=False)
    n_checked = check_causal_sets(both["A"][0], both["B"][0])
    acc = {s: json.loads((OUT / s / "accuracy.json").read_text()) for s in ("all", "A", "B")}

    figure_heads(*both["all"])
    figure_overlap(ov)

    pooled = ov[ov.scope == "pooled"].pivot(index="n", columns="split", values="shared")
    pooled_ch = ov[ov.scope == "pooled"].groupby("n")[["chance", "chance_layer"]].mean()
    task = ov[ov.scope != "pooled"].groupby(["split", "n"])[["shared", "chance", "chance_layer"]].mean().reset_index()
    with (OUT / "report.md").open("w") as f:
        f.write("# E27: causal heads from instructions vs demonstrations (Davidson et al. Sec. 3.3)\n\n")
        f.write(f"Davidson et al., Llama-3.1-8B-Instruct: {DAVIDSON_SHARED} of the top 20 shared.\n\n")
        f.write("## 10-shot accuracy (first token; prompts tried until 125 correct)\n\n"
                + md(pd.DataFrame({s: {t: v["accuracy"] for t, v in a.items()} for s, a in acc.items()}).round(3))
                + "\n\n")
        f.write("## One set over the 14 tasks: heads in both top-N\n\n"
                + md(pooled.join(pooled_ch.round(1))) + "\n\n")
        f.write("## Per task: heads in both top-N, mean over the 14 tasks\n\n"
                + md(task.round(2).set_index("split")) + "\n\n")
        f.write("## Rank correlation of the two scores over the 1024 heads\n\n"
                + md(cr[cr.scope == "pooled"].round(3).set_index("split")) + "\n\nPer task (mean, min, max): "
                + "; ".join(f"{s}: {g.rank_corr.mean():.2f} ({g.rank_corr.min():.2f}, {g.rank_corr.max():.2f})"
                            for s, g in cr[cr.scope != "pooled"].groupby("split")) + "\n\n")
        f.write("## Mean layer of the top 20 (one set over the tasks)\n\n"
                + md(ly[ly.scope == "pooled"].round(1).set_index("split")) + "\n\n")
        f.write(f"Check: the held-out instruction rankings reproduce head-sets-simple's Causal sets ({n_checked} sets).\n")
    print((OUT / "report.md").read_text())


def figure_heads(inst: dict, demo: dict) -> None:
    a = {tuple(h) for h in top(np.argsort(-inst["pooled"], kind="stable"), 20)}
    b = {tuple(h) for h in top(np.argsort(-demo["pooled"], kind="stable"), 20)}
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for heads, colour, label in ((a - b, rt.ORANGE, "instruction only"), (b - a, rt.BLUE, "demonstration only"),
                                 (a & b, rt.GREEN, "both")):
        if heads:
            ax.scatter([l for l, _ in heads], [h for _, h in heads], s=22, color=colour, label=label, zorder=3)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_ylim(-0.5, N_HEADS - 0.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("head")
    ax.set_title(f"top 20 heads, {len(a & b)} in both", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "e27_heads.pdf")
    fig.savefig(OUT / "e27_heads.png", dpi=130)
    plt.close(fig)


def figure_overlap(ov: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    colour = {"all": rt.GREY, "A": rt.BLUE, "B": rt.AQUA}
    for ax, scope, title in ((axes[0], "pooled", "(a) One set over the 14 tasks"), (axes[1], "task", "(b) Per task, mean")):
        d = ov[ov.scope == "pooled"] if scope == "pooled" else ov[ov.scope != "pooled"]
        m = d.groupby(["split", "n"], as_index=False)[["shared", "chance", "chance_layer"]].mean()
        for split, g in m.groupby("split"):
            ax.plot(g.n, g.shared / g.n, "o-", ms=3, color=colour[split],
                    label={"all": "all items", "A": "half A", "B": "half B"}[split])
            ax.plot(g.n, g.chance_layer / g.n, ls=rt.CONTROL_STYLE, color=colour[split], lw=1)
        g = m.groupby("n").chance.mean()
        ax.plot(g.index, g / g.index, color="black", lw=0.8, ls="--", label="chance (uniform)")
        ax.set_title(title, loc="left")
        ax.set_xticks(SIZES)
        ax.set_xlabel("heads in the set ($N$)")
    axes[0].scatter([20], [DAVIDSON_SHARED / 20], marker="*", s=60, color=rt.ORANGE, zorder=4, label="Davidson et al.")
    axes[0].set_ylabel("share of the $N$ heads in both")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7, ncol=5, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "e27_overlap.pdf")
    fig.savefig(OUT / "e27_overlap.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
