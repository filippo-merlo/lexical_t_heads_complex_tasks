"""Signed screen (scripts/rebuild_signed_screen.sh): scores, shortlist, choice and report. No GPU.

Quick score of a set on a fold: first-token accuracy at alpha 2 averaged over the template means, then over the 14
tasks (as ``scripts/rebuild_shortlist.py``). Full score: per task the template-mean accuracy at the better alpha,
then the mean over tasks (as ``scripts/rebuild_select.py``). Cross-fit as Selected: fold f's shortlist = up to
``K`` candidates per scope that score higher than Selected on fold 3 - f's quick pass (ties to fewer features);
fold f then chooses between Selected and its shortlist on fold 3 - f's full pass (ties to Selected, then fewer
features), and its own fold's full score is the test.

Checks: Selected's rows here against the stored TF-IDF screen (quick pass ``part-quick-s2-*``, full pass
``part-s2-screen-*``), every task and template mean.

    .venv/bin/python scripts/rebuild_signed_report.py --shortlist   # after the quick pass: shortlist.json, full_names.txt
    .venv/bin/python scripts/rebuild_signed_report.py               # report.md (and the full pass once it exists)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import RESULTS, simple  # noqa: E402
from rebuild_sets import FOLDS  # noqa: E402

OUT = RESULTS / "rebuild" / "signed"
TFIDF = RESULTS / "rebuild" / "tfidf_screen"
PREFIX = {"per task": "signed_", "shared": "shared_signed_"}
STORED = {"per task": "screen_", "shared": "shared_screen_"}
K, QUICK_ALPHA = 3, 2.0


def rows(paths):
    for p in paths:
        yield from (r for r in csv.DictReader(p.open()) if r["source"] == "task" and r["heads"])


def paths(fold: int, part: str, base: Path = OUT) -> list[Path]:
    return sorted((base / f"fold{fold}").glob(f"patching_simple.part-{part}-*.csv"))


def scores(fold: int, part: str, combos: dict[str, list[str]], quick: bool):
    """score[scope][combo] = {task: score} (quick: first token at alpha 2; full: accuracy at the better alpha)."""
    cell = defaultdict(lambda: defaultdict(list))
    for r in rows(paths(fold, part)):
        if quick and float(r["alpha"]) != QUICK_ALPHA:
            continue
        cell[(r["task"], r["heads"])][float(r["alpha"])].append(float(r["first_token" if quick else "accuracy"]))
    out = {}
    for scope, cs in combos.items():
        out[scope] = {}
        for c in cs:
            per = {t: max(np.mean(v) for v in cell[(t, PREFIX[scope] + c)].values())
                   for t in simple.TASKS if cell.get((t, PREFIX[scope] + c))}
            if len(per) == len(simple.TASKS):
                out[scope][c] = per
    return out


def mean(per_task: dict[str, float]) -> float:
    return float(np.mean(list(per_task.values())))


def check(fold: int, part: str, stored_part: str, base: dict[str, str], column: str) -> float:
    """Largest difference between Selected's rows here and in the stored TF-IDF screen (same task, mean, alpha)."""
    def table(ps, names):
        return {(r["task"], names[r["heads"]], r["mean_wording"], r["alpha"]): float(r[column])
                for r in rows(ps) if r["heads"] in names}
    here = table(paths(fold, part), {PREFIX[s] + base[s]: s for s in base})
    there = table(paths(fold, stored_part, TFIDF), {STORED[s] + base[s]: s for s in base})
    common = here.keys() & there.keys()
    assert common and all(k in there for k in here), f"fold {fold} {part}: rows missing in the stored screen"
    return max(abs(here[k] - there[k]) for k in common)


def order(combos: list[str], score: dict[str, dict[str, float]]) -> list[str]:
    return sorted(combos, key=lambda c: (-mean(score[c]), len(c), combos.index(c)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shortlist", action="store_true", help="write the shortlist for the full pass")
    args = parser.parse_args()
    sets = json.loads((OUT / "sets.json").read_text())
    base, combos, overlap = sets["selected"], sets["candidates"], sets["overlap_with_selected"]
    quick = {f: scores(f, "quick", combos, quick=True) for f in FOLDS}
    for f in FOLDS:
        for scope in combos:
            assert len(quick[f][scope]) == len(combos[scope]), f"fold {f} {scope}: quick pass incomplete"
    lines = ["# Signed screen: Selected with features subtracted (40 heads, TF-IDF lists)", "",
             f"Selected: per task {base['per task']}, shared {base['shared']} (both folds). A subtracted feature "
             "ranks heads by it from the bottom. Quick pass: alpha 2, 10 test items per task, first-token accuracy, "
             "task means only; mean over template means, then over the 14 tasks.", ""]
    qmax = max(check(f, "quick", "quick-s2", base, "first_token") for f in FOLDS)
    lines += [f"Check: Selected's quick rows equal the stored TF-IDF quick pass to {qmax:.4f} (every task, template mean).", ""]
    short = {}
    for scope in combos:
        b = base[scope]
        lines += [f"## Quick pass, {scope}", "",
                  "| combination | heads shared with Selected (f1/f2) | fold 1 | fold 2 | mean | minus Selected |",
                  "|---|---|---|---|---|---|"]
        bm = np.mean([mean(quick[f][scope][b]) for f in FOLDS])
        for c in sorted(combos[scope], key=lambda c: -np.mean([mean(quick[f][scope][c]) for f in FOLDS])):
            q = [mean(quick[f][scope][c]) for f in FOLDS]
            ov = "/".join(f"{overlap[str(f)][scope][PREFIX[scope] + c]:.1f}".rstrip("0").rstrip(".") for f in FOLDS)
            lines.append(f"| {c}{' (Selected)' if c == b else ''} | {ov} | {q[0]:.3f} | {q[1]:.3f} | {np.mean(q):.3f} | "
                         f"{np.mean(q) - bm:+.3f} |")
        lines.append("")
        for f in FOLDS:
            other = 3 - f
            cands = [c for c in order(combos[scope], quick[other][scope]) if c != b
                     and mean(quick[other][scope][c]) > mean(quick[other][scope][b])][:K]
            short.setdefault(str(f), {})[scope] = [[c, round(mean(quick[other][scope][c]), 4)] for c in cands]
        lines += [f"Shortlist (beats Selected on the other fold's quick pass, at most {K}): " + "; ".join(
            f"fold {f}: {', '.join(c for c, _ in short[str(f)][scope]) or 'none'}" for f in FOLDS), ""]
    union = {scope: sorted({c for f in FOLDS for c, _ in short[str(f)][scope]}, key=combos[scope].index)
             for scope in combos}
    if args.shortlist:
        (OUT / "shortlist.json").write_text(json.dumps({**short, "union": union, "k": K}, indent=1))
        names = [PREFIX[s] + c for s in combos if union[s] for c in [base[s]] + union[s]]
        (OUT / "full_names.txt").write_text("".join(n + "\n" for n in names))
        print(f"shortlist: " + "; ".join(f"{s} {union[s] or 'none'}" for s in combos) + f" -> {len(names)} sets in full")

    if all(len(paths(f, "full")) == len(simple.TASKS) for f in FOLDS):
        stored_short = json.loads((OUT / "shortlist.json").read_text())
        run = {s: [base[s]] + stored_short["union"][s] for s in combos if stored_short["union"][s]}
        full = {f: scores(f, "full", run, quick=False) for f in FOLDS}
        fmax = max(check(f, "full", "s2-screen", {s: base[s] for s in run}, "accuracy") for f in FOLDS)
        lines += ["## Full pass (alphas 1 and 2, all test items, task means only)", "",
                  f"Check: Selected's full rows equal the stored TF-IDF full pass to {fmax:.4f}.", ""]
        for scope, cs in run.items():
            lines += [f"### {scope}", "", "| combination | fold 1 | fold 2 |", "|---|---|---|"]
            lines += [f"| {c}{' (Selected)' if c == base[scope] else ''} | {mean(full[1][scope][c]):.3f} | "
                      f"{mean(full[2][scope][c]):.3f} |" for c in cs]
            lines.append("")
            for f in FOLDS:
                other = 3 - f
                pool = [base[scope]] + [c for c, _ in stored_short[str(f)][scope]]
                pick = min(pool, key=lambda c: (-mean(full[other][scope][c]), c != base[scope], len(c)))
                own, b = full[f][scope][pick], full[f][scope][base[scope]]
                lines.append(f"Fold {f} picks **{pick}** (fold {other} full {mean(full[other][scope][pick]):.3f}); "
                             f"on fold {f}: {mean(own):.3f} vs Selected {mean(b):.3f} ({mean(own) - mean(b):+.3f}).")
                if pick != base[scope]:
                    lines.append("Per task on fold %d (pick minus Selected): " % f + ", ".join(
                        f"{t} {own[t] - b[t]:+.2f}" for t in simple.TASKS))
            lines.append("")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
