"""Rebuilt design (paper/head-sets-simple-design.tex), unseen tasks: accuracy of the shared 40-head sets on tasks
that took no part in identifying them (``scripts/rebuild_newtasks.sh``). No GPU.

Accuracy of a (fold, task, set, mean source, alpha) cell is the mean over the template means
(``rebuild_report.load``). alpha is fixed (1 or 2), not cross-fitted (user 2026-11-19). For every unseen
task, set and alpha: task means, control means (the control task's means of the same templates) and the
three random sets (task means, averaged over the draws); then means over the two folds and over the tasks.
Reference: the same sets on the 14 tasks they were identified on, at the same fixed alpha, from their
stage-1/2 runs (``results/rebuild/additions``; the screen sets have task means only).

Also, for every unseen task and fold: the unpatched accuracy of the patched items, the share of the most
common expected answer among them (majority-label rate: a set that always gives one label scores up to
this), and every condition in which one reply makes up at least half of the replies (from the answers files).

Writes ``results/rebuild/newtasks/per_task.csv``, ``summary.csv``, ``repeated.csv`` and prints the summary.

    .venv/bin/python scripts/rebuild_newtask_report.py
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import RESULTS, simple  # noqa: E402
import rebuild_report  # noqa: E402
from rebuild_newtask_sets import combos_used, selected  # noqa: E402
from rebuild_sets import FOLDS, N_RANDOM, SCREEN_N  # noqa: E402

DIR = RESULTS / "rebuild" / "newtasks"
ALPHAS = (1.0, 2.0)
REPEATED = 0.5


def rules() -> dict[str, str]:
    """label -> set name"""
    SELECTED = selected()
    out = {f"Combined ({SELECTED})": f"shared_selected{SCREEN_N}"}
    out.update({c: f"shared_screen_{c}" for c in combos_used() if c != SELECTED})
    out.update({"Causal": f"shared_causal{SCREEN_N}", "Causal - generic": f"shared_causalnogen{SCREEN_N}",
                "D": f"shared_D{SCREEN_N}"})
    return out


def main() -> None:
    rebuild_report.DIR = DIR
    acc = rebuild_report.load("nt")
    rebuild_report.DIR = RESULTS / "rebuild" / "additions"
    ref = rebuild_report.load("s1")
    ref.update(rebuild_report.load("s2"))  # also globs s2sel (Combined) and the screen
    tasks = simple.UNSEEN_TASKS
    rows = []
    for label, nm in rules().items():
        for alpha in ALPHAS:
            for fold in FOLDS:
                for t in tasks:
                    task = acc[(fold, t, nm, "task")][alpha]
                    control = acc[(fold, t, nm, "control")][alpha]
                    random = float(np.mean([acc[(fold, t, f"random_{nm}_{i}", "task")][alpha] for i in range(N_RANDOM)]))
                    rows.append(dict(rule=label, set=nm, alpha=alpha, fold=fold, task=t, accuracy=task,
                                     control=control, random=random))
    rounded = lambda d, k: {a: round(v, k) if isinstance(v, float) else v for a, v in d.items()}  # noqa: E731
    with (DIR / "per_task.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rounded(r, 4) for r in rows)

    summary = []  # unrounded; rounded only when written
    for label, nm in rules().items():
        for alpha in ALPHAS:
            sel = [r for r in rows if r["rule"] == label and r["alpha"] == alpha]
            assert len(sel) == len(FOLDS) * len(tasks)
            line = dict(rule=label, alpha=alpha)
            for t in tasks:
                line[t] = float(np.mean([r["accuracy"] for r in sel if r["task"] == t]))
            line["unseen mean"] = float(np.mean([r["accuracy"] for r in sel]))
            line["control"] = float(np.mean([r["control"] for r in sel]))
            line["random"] = float(np.mean([r["random"] for r in sel]))
            seen = [ref[(fold, t, nm, "task")][alpha] for fold in FOLDS for t in simple.TASKS]
            assert len(seen) == 2 * len(simple.TASKS)
            line["14 tasks (identified on)"] = float(np.mean(seen))
            summary.append(line)
    with (DIR / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(rounded(s, 3) for s in summary)

    # Unpatched baseline, majority label, repeated replies.
    repeated, base = [], []
    for fold in FOLDS:
        for t in tasks:
            path = DIR / f"fold{fold}" / "answers" / f"patching_simple.part-nt-{t}.jsonl.gz"
            lines = [json.loads(x) for x in gzip.open(path, "rt")]
            none = [x for x in lines if x["source"] == "none"]
            expected = Counter(x["expected"] for x in none)
            label, count = expected.most_common(1)[0]
            base.append(dict(fold=fold, task=t, items=len(none), unpatched=float(np.mean([x["correct"] for x in none])),
                             majority_label=label, majority_rate=count / len(none)))
            groups = defaultdict(list)
            for x in lines:
                if x["source"] != "none":
                    groups[(x["heads"], x["source"], x["mean_wording"], x["alpha"])].append(x)
            for (nm, source, wording, alpha), xs in groups.items():
                reply, n = Counter(x["predicted"].strip().lower() for x in xs).most_common(1)[0]
                if n / len(xs) >= REPEATED:
                    repeated.append(dict(fold=fold, task=t, set=nm, source=source, wording=wording, alpha=alpha,
                                         reply=reply, share=round(n / len(xs), 3),
                                         correct=round(np.mean([x["correct"] for x in xs if x["predicted"].strip().lower() == reply]), 3)))
    with (DIR / "repeated.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "task", "set", "source", "wording", "alpha", "reply", "share", "correct"])
        w.writeheader()
        w.writerows(repeated)

    print("baseline of the patched items:")
    for b in base:
        print(f"  fold {b['fold']} {b['task']:18s} items {b['items']:3d} unpatched {b['unpatched']:.3f} "
              f"majority '{b['majority_label']}' {b['majority_rate']:.3f}")
    print(f"\naccuracy at {SCREEN_N} heads, mean of the two folds (control, random: mean over the unseen tasks):")
    for s in summary:
        print(f"  {s['rule']:18s} a={s['alpha']:.0f}  " + "  ".join(f"{t}={s[t]:.3f}" for t in tasks)
              + f"  | unseen {s['unseen mean']:.3f}  ctrl {s['control']:.3f}  rand {s['random']:.3f}  | 14 tasks {s['14 tasks (identified on)']:.3f}")
    n_task = sum(r["source"] == "task" and not r["set"].startswith("random_") for r in repeated)
    print(f"\nconditions with one reply >= {REPEATED:.0%} of replies: {len(repeated)} "
          f"({n_task} with task means on a rule's own set); see repeated.csv")
    tables(summary, base)


def tables(summary: list[dict], base: list[dict]) -> None:
    """paper/tables/design_unseen.tex: the rules of Sec. 5 and the combinations that do better than Combined on
    the unseen tasks at both alphas; design_unseen_combos.tex: the other combinations."""
    import results_tex as rt

    tasks = simple.UNSEEN_TASKS
    by = {(s["rule"], s["alpha"]): s for s in summary}
    SELECTED = selected()
    selected_label = f"Combined ({SELECTED})"
    combos = [c for c in combos_used() if c != SELECTED]
    better = [c for c in combos if all(by[(c, a)]["unseen mean"] > by[(selected_label, a)]["unseen mean"] for a in ALPHAS)]
    others = [c for c in combos if c not in better]
    print(f"combinations above Combined on the unseen tasks at both alphas: {better}")
    short = {"person-occupation": "occupation", "person-sport": "sport", "person-instrument": "instrument",
             "capitalize": "capitalize"}
    label = lambda r: r.replace(" - ", " $-$ ")  # noqa: E731
    header = [" & & \\multicolumn{4}{c}{Unseen task} & & & & \\\\", "\\cmidrule(lr){3-6}",
              "& Rule & " + " & ".join(short[t] for t in tasks) + " & Mean & Control & Random & 14 tasks \\\\"]

    def block(names: list[str]) -> list[str]:
        out = []
        for alpha in ALPHAS:
            for i, r in enumerate(names):
                s = by[(r, alpha)]
                out.append((f"$\\alpha={alpha:.0f}$" if i == 0 else "") + f" & {label(r)} & "
                           + " & ".join(rt.num(s[t]) for t in tasks)
                           + f" & {rt.num(s['unseen mean'])} & {rt.num(s['control'])} & {rt.num(s['random'])}"
                           + f" & {rt.num(s['14 tasks (identified on)'])} \\\\")
            if alpha != ALPHAS[-1]:
                out.append("\\midrule")
        return out

    mean_base = lambda key, t: float(np.mean([b[key] for b in base if b["task"] == t]))  # noqa: E731
    top = [" & No patch & " + " & ".join(rt.num(mean_base("unpatched", t)) for t in tasks) + " & & & & \\\\",
           " & Most common answer & " + " & ".join(rt.num(mean_base("majority_rate", t)) for t in tasks) + " & & & & \\\\",
           "\\midrule"]
    main_rules = [selected_label, *better, "Causal", "Causal - generic", "D"]
    spec = "@{}ll" + "r" * (len(tasks) + 4) + "@{}"
    note = "% Generated by scripts/rebuild_newtask_report.py.\n"
    (rt.TABLES / "design_unseen.tex").write_text(note + rt.tabular(spec, header, top + block(main_rules)))
    (rt.TABLES / "design_unseen_combos.tex").write_text(note + rt.tabular(spec, header, block(others)))
    print(f"wrote {rt.TABLES / 'design_unseen.tex'} and design_unseen_combos.tex (others: {others})")


if __name__ == "__main__":
    main()
