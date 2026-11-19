"""Rebuilt design (paper/head-sets-simple-design.tex): Selected's combination of features chosen without the
task it is tested on. Per-task scope, 40 heads, from the stage-2 screen; no GPU.

Agreed with the user 2026-11-19. Two schemes, every task tested exactly once in each:
  * ``loto``  leave one task out: the combination is chosen on the other 13 tasks;
  * ``loty``  leave one type out: chosen on the tasks of the other four types, tested on every task of the
              held-out type (types as in ``rebuild_figures.TYPES``).
Reference ``all14``: the same choice on all 14 tasks (the tested task included), i.e. what holding out costs.

alpha is NOT chosen: every scheme is run once with alpha = 1 everywhere and once with alpha = 2 everywhere
(user: "try alpha 1 and 2"); the combination is chosen at the alpha it is tested with. Items stay
cross-fitted as for Selected (Sec. 5.4): for fold f, a combination's score is its accuracy on fold 3 - f
(whose test items are fold f's identification half), averaged over the identification tasks; ties go to
fewer features, then to the order of the combinations (``rebuild_select.best``). The chosen combination is
scored on fold f's test items of the held-out task(s).

Accuracy of a (fold, task, set, alpha) cell = mean over the five template means, as in
``rebuild_report.load``. Reference rules at the same fixed alpha, from their own runs: Selected (chosen on all
14 tasks with each task at its better alpha, = DBP), Causal, Causal - generic (its generic heads use all 14
tasks, so it is not held out), D.

Checks before anything is written: the screen is complete (every combination, task, fold and alpha); this
loader reproduces ``rebuild_select.screen_scores`` for every combination; the choice on all 14 tasks with
each task at its better alpha reproduces ``selected.json``.

Writes ``results/rebuild/heldout_tasks/``: ``choices.json`` (every scheme, alpha, fold and held-out unit:
combination, its identification score, the identification tasks), ``per_task.csv`` and ``summary.csv``.

    .venv/bin/python scripts/rebuild_heldout_tasks.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import simple  # noqa: E402
from rebuild_figures import TYPES  # noqa: E402
import rebuild_report  # noqa: E402
from rebuild_select import best, screen_scores  # noqa: E402
from rebuild_sets import COMBOS, FOLDS, OUT, SCREEN_N  # noqa: E402

ALPHAS = (1.0, 2.0)
PREFIX = "screen_"  # per-task screen sets (shared ones are "shared_screen_")
DEST = OUT.parent / "heldout_tasks"
REFERENCES = {"Selected (all 14, best alpha)": f"selected{SCREEN_N}", "Causal": f"causal{SCREEN_N}",
              "Causal - generic (not held out)": f"causalnogen{SCREEN_N}", "D": f"D{SCREEN_N}"}


def load_screen() -> dict:
    """acc[fold][(task, combo)][alpha] = accuracy averaged over the template means (rebuild_report.load)."""
    acc = {}
    for fold in FOLDS:
        cells = defaultdict(lambda: defaultdict(list))
        for path in sorted((OUT / f"fold{fold}").glob("patching_simple.part-s2-*.csv")):
            for r in csv.DictReader(path.open()):
                if r["source"] == "task" and r["heads"].startswith(PREFIX):
                    cells[(r["task"], r["heads"][len(PREFIX):])][float(r["alpha"])].append(float(r["accuracy"]))
        acc[fold] = {k: {a: float(np.mean(v)) for a, v in d.items()} for k, d in cells.items()}
    return acc


def check(acc: dict) -> None:
    for fold in FOLDS:
        missing = [(t, c, a) for t in simple.TASKS for c in COMBOS for a in ALPHAS
                   if a not in acc[fold].get((t, c), {})]
        assert not missing, f"fold {fold}: {len(missing)} screen cells missing, e.g. {missing[:3]}"
        reference = screen_scores(fold)["per task"]
        mine = {c: float(np.mean([max(acc[fold][(t, c)].values()) for t in simple.TASKS])) for c in COMBOS}
        assert set(reference) == set(mine), fold
        worst = max(abs(reference[c] - mine[c]) for c in COMBOS)
        assert worst < 1e-12, f"fold {fold}: loader differs from rebuild_select.screen_scores by {worst}"
    selected = json.loads((OUT / "selected.json").read_text())
    for fold in FOLDS:
        other = 3 - fold
        scores = {c: float(np.mean([max(acc[other][(t, c)].values()) for t in simple.TASKS])) for c in COMBOS}
        assert best(scores, COMBOS) == selected[str(fold)]["per task"]["combo"], fold
    print(f"checks passed: screen complete ({len(COMBOS)} combinations x {len(simple.TASKS)} tasks x 2 folds x "
          f"alpha 1, 2); loader = rebuild_select.screen_scores; all-14 choice = selected.json "
          f"({selected['1']['per task']['combo']}, {selected['2']['per task']['combo']})", flush=True)


def units(scheme: str) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    """(unit name, identification tasks, test tasks) for every held-out unit of the scheme."""
    tasks = tuple(simple.TASKS)
    assert sorted(t for ts in TYPES.values() for t in ts) == sorted(tasks), "TYPES must partition the 14 tasks"
    if scheme == "all14":
        return [("all", tasks, tasks)]
    if scheme == "loto":
        return [(t, tuple(u for u in tasks if u != t), (t,)) for t in tasks]
    if scheme == "loty":
        return [(name, tuple(u for u in tasks if u not in ts), tuple(ts)) for name, ts in TYPES.items()]
    raise ValueError(scheme)


def main() -> None:
    acc = load_screen()
    check(acc)
    ref = rebuild_report.load("s1")
    ref.update(rebuild_report.load("s2"))  # s2 also globs s2sel (Selected) and the screen; keys are set names
    choices, rows = {}, []
    for scheme in ("all14", "loto", "loty"):
        for alpha in ALPHAS:
            for fold in FOLDS:
                other = 3 - fold
                for name, ident, test in units(scheme):
                    assert scheme == "all14" or not set(ident) & set(test)
                    scores = {c: float(np.mean([acc[other][(t, c)][alpha] for t in ident])) for c in COMBOS}
                    combo = best(scores, COMBOS)
                    choices.setdefault(scheme, {}).setdefault(str(alpha), {}).setdefault(str(fold), {})[name] = dict(
                        combo=combo, id_score=round(scores[combo], 4), id_tasks=list(ident))
                    for t in test:
                        rows.append(dict(scheme=scheme, alpha=alpha, fold=fold, task=t, unit=name, combo=combo,
                                         accuracy=round(acc[fold][(t, combo)][alpha], 4)))
    for label, set_name in REFERENCES.items():
        for alpha in ALPHAS:
            for fold in FOLDS:
                for t in simple.TASKS:
                    rows.append(dict(scheme=label, alpha=alpha, fold=fold, task=t, unit="", combo="",
                                     accuracy=round(ref[(fold, t, set_name, "task")][alpha], 4)))
    # every scheme and reference: each task once per fold and alpha
    count = defaultdict(int)
    for r in rows:
        count[(r["scheme"], r["alpha"], r["fold"], r["task"])] += 1
    assert set(count.values()) == {1} and len(count) == (3 + len(REFERENCES)) * 2 * 2 * len(simple.TASKS)

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "choices.json").write_text(json.dumps(choices, indent=1))
    with (DEST / "per_task.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for scheme in ["all14", "loto", "loty", *REFERENCES]:
        for alpha in ALPHAS:
            sel = [r for r in rows if r["scheme"] == scheme and r["alpha"] == alpha]
            line = dict(scheme=scheme, alpha=alpha, all=round(float(np.mean([r["accuracy"] for r in sel])), 3))
            for tname, ts in TYPES.items():
                line[tname] = round(float(np.mean([r["accuracy"] for r in sel if r["task"] in ts])), 3)
            picks = [c["combo"] for f in choices.get(scheme, {}).get(str(alpha), {}).values() for c in f.values()]
            line["combos"] = " ".join(sorted(set(picks)))
            summary.append(line)
    with (DEST / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(f"{s['scheme']:34s} a={s['alpha']:.0f} all={s['all']:.3f}  "
              + "  ".join(f"{k}={s[k]:.3f}" for k in TYPES) + (f"  picks: {s['combos']}" if s["combos"] else ""))


if __name__ == "__main__":
    main()
