"""Shared scope, every size: the combination chosen by the LRP-weighted screen (``scripts/rebuild_screen_n.sh`` with
POOL=lrp, ``results/rebuild/lrp_shared_n<N>/``) next to the combination chosen by the attention screen at the same
size (``results/rebuild/shared_n<N>/``; at 40 heads stage 2's shared pick, ``results/rebuild/additions/``), with
shared LRP-T, Causal and Causal - generic for reference. No GPU.

Accuracy: check 1 over the 14 tasks and both folds, each fold's pick on its own test half, alpha cross-fitted by
rule (``scripts/rebuild_report.py``'s ``sel`` / ``s1`` / ``s2`` / ``lrp`` summaries); control means and the
layer-matched random sets alongside. Heads in common: the two picks of the same fold and size, mean over folds.
Per task: the two picks' cross-fitted accuracy per task, mean over folds.

Writes ``results/rebuild/lrp_shared_compare.csv`` and ``results/rebuild/lrp_shared_per_task.csv`` and prints them.
Sizes whose LRP run is not finished are shown as "--".

    .venv/bin/python scripts/lrp_shared_compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rebuild_report  # noqa: E402
from lth import RESULTS, simple  # noqa: E402

R = RESULTS / "rebuild"
SIZES = (10, 20, 40, 60, 80, 100, 120)
REFS = {"LRP-T": ("lrp", "shared_lrpT"), "Causal": ("s1", "shared_causal"), "Causal - gen.": ("s2", "shared_causalnogen")}


def summary(directory: Path, stage: str) -> pd.DataFrame | None:
    path = directory / f"{stage}_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path).set_index("set")
    for c in ("task", "control", "random"):  # unfillable sets are written as "--"
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ours(n: int):
    """The attention screen's shared pick at n: (directory, sel-like stage, combos per fold)."""
    if n == 40:
        sel = json.loads((R / "additions" / "selected.json").read_text())
        return R / "additions", "s2", {f: sel[str(f)]["shared"]["combo"] for f in (1, 2)}
    sel = json.loads((R / f"shared_n{n}" / "selected.json").read_text())
    return R / f"shared_n{n}", "sel", {f: sel[str(f)]["combo"] for f in (1, 2)}


def heads_of(directory: Path, fold: int, name: str) -> set:
    sets = json.loads((directory / f"fold{fold}" / "heads.json").read_text())["simple"][simple.TASKS[0]]["sets"]
    (heads,) = [hs for _, nm, hs in sets if nm == name]
    return {tuple(h) for h in heads}


def general(n: int) -> tuple[float, float, float]:
    """The general combination (scripts/general_sets.py, DBTSML) at n heads: its own run (general_n, part gen) with
    control and random sets; else the full pass of the size's screen where it was shortlisted (accuracy only)."""
    g = summary(R / "general_n", "gen")
    if g is not None and f"shared_general{n}" in g.index:
        r = g.loc[f"shared_general{n}"]
        return r.task, r.control, r.random
    combo = json.loads((R / "general_n" / "selected.json").read_text())["1"]["combo"]
    path = R / f"shared_n{n}" / "summary.csv"
    if path.exists():
        s = pd.read_csv(path).set_index("combo")
        if combo in s.index:
            return float(s.loc[combo].accuracy), np.nan, np.nan
    return np.nan, np.nan, np.nan


def per_task(directory: Path, stage: str, name: str) -> dict[str, float]:
    rebuild_report.DIR = directory
    cf = rebuild_report.crossfit(rebuild_report.load(stage))
    return {t: float(np.mean([cf[(f, t, name, "task")][1] for f in (1, 2)])) for t in simple.TASKS}


def main() -> None:
    rows, tasks = [], []
    for n in SIZES:
        label = f"shared_selected{n}"
        odir, ostage, ocombo = ours(n)
        o = summary(odir, ostage).loc[label]
        row = dict(n=n, ours_combo="/".join(dict.fromkeys(ocombo.values())), ours=o.task, ours_control=o.control,
                   ours_random=o.random)
        ldir = R / f"lrp_shared_n{n}"
        l_sum = summary(ldir, "sel")
        if l_sum is not None and label in l_sum.index:
            sel = json.loads((ldir / "selected.json").read_text())
            l = l_sum.loc[label]
            assert l.tasks_folds == 28 and o.tasks_folds == 28, n
            common = [len(heads_of(ldir, f, label) & heads_of(odir, f, label)) for f in (1, 2)]
            row.update(lrp_combo="/".join(dict.fromkeys(sel[str(f)]["combo"] for f in (1, 2))), lrp=l.task,
                       lrp_control=l.control, lrp_random=l.random, common=float(np.mean(common)))
            pt_o, pt_l = per_task(odir, ostage, label), per_task(ldir, "sel", label)
            tasks += [dict(n=n, task=t, ours=round(pt_o[t], 3), lrp=round(pt_l[t], 3)) for t in simple.TASKS]
        for ref, (stage, prefix) in REFS.items():
            s = summary(R / "additions", stage)
            row[ref] = s.loc[f"{prefix}{n}"].task if s is not None and f"{prefix}{n}" in s.index else np.nan
        row["general"], row["general_control"], row["general_random"] = general(n)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(R / "lrp_shared_compare.csv", index=False)
    pd.set_option("display.width", 220)
    print("Shared scope: accuracy (control / random), each fold's pick on its own test half; combos fold 1/fold 2\n")
    for r in rows:
        ours_cell = f"{r['ours']:.2f} ({r['ours_control']:.2f}/{r['ours_random']:.2f}) {r['ours_combo']}"
        lrp_cell = (f"{r['lrp']:.2f} ({r['lrp_control']:.2f}/{r['lrp_random']:.2f}) {r['lrp_combo']}  common {r['common']:.1f}"
                    if "lrp" in r else "--")
        refs = "  ".join(f"{k} {r[k]:.2f}" for k in REFS)
        g = r["general"]
        refs += "  DBTSML " + ("--" if np.isnan(g) else f"{g:.2f}" + (
            "" if np.isnan(r["general_control"]) else f" ({r['general_control']:.2f}/{r['general_random']:.2f})"))
        print(f"{r['n']:>4}  attention: {ours_cell:34s} LRP: {lrp_cell:48s} {refs}")
    if tasks:
        t = pd.DataFrame(tasks)
        t.to_csv(R / "lrp_shared_per_task.csv", index=False)
        t["diff"] = t.lrp - t.ours
        print("\nPer task, LRP pick minus attention pick (mean over folds):")
        print(t.pivot(index="task", columns="n", values="diff").round(2).to_string())


if __name__ == "__main__":
    main()
