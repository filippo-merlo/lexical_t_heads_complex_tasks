"""Shared LRP-T with raw scores averaged over tasks (``scripts/lrp_raw_sets.py``) against the shared rules at every
size. No GPU.

Reads ``results/rebuild/lrp_raw_n/raw_summary.csv`` and ``raw_per_task.csv`` (10, 40-120 heads),
``results/rebuild/pham_frame/pham_summary.csv`` and ``pham_per_task.csv`` (20 heads), the stored rank-mean LRP-T
(``results/rebuild/additions/lrp_summary.csv``, ``lrp_per_task.csv``) and the per-size comparison of
``scripts/lrp_shared_compare.py`` (``results/rebuild/lrp_shared_compare.csv``, ``lrp_shared_per_task.csv``: Combined
per size, LRP-Combined, Causal, Causal - generic). Writes ``results/rebuild/lrp_raw_n/compare.csv`` (per size:
accuracy, control and random of both LRP-T forms, the others' accuracy) and ``compare_per_task.csv`` (per size and
task, mean over the folds), and prints both.

    .venv/bin/python scripts/lrp_raw_compare.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import RESULTS, simple  # noqa: E402

R = RESULTS / "rebuild"
SIZES = (10, 20, 40, 60, 80, 100, 120)


def rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def per_task(path: Path, name: str) -> dict[str, float]:
    got = defaultdict(list)
    for r in rows(path):
        if r["set"] == name and r["source"] == "task":
            got[r["task"]].append(float(r["accuracy"]))
    assert all(len(got[t]) == 2 for t in simple.TASKS), (path, name)
    return {t: sum(v) / 2 for t, v in got.items()}


def main() -> None:
    summary = {r["set"]: r for r in rows(R / "lrp_raw_n" / "raw_summary.csv") + rows(R / "pham_frame" / "pham_summary.csv")
               + rows(R / "additions" / "lrp_summary.csv")}
    other = {int(r["n"]): r for r in rows(R / "lrp_shared_compare.csv")}
    other_task = {(int(r["n"]), r["task"]): r for r in rows(R / "lrp_shared_per_task.csv")}
    out, out_task = [], []
    for n in SIZES:
        raw, rank = summary[f"shared_lrpTraw{n}"], summary[f"shared_lrpT{n}"]
        assert raw["tasks_folds"] == rank["tasks_folds"] == "28", n
        o = other[n]
        out.append({"n": n, "raw": raw["task"], "raw_control": raw["control"], "raw_random": raw["random"],
                    "rank": rank["task"], "rank_control": rank["control"], "rank_random": rank["random"],
                    "combined_combo": o["ours_combo"], "combined": o["ours"], "lrp_combined_combo": o["lrp_combo"],
                    "lrp_combined": o["lrp"], "causal": o["Causal"], "causal_gen": o["Causal - gen."]})
        src = R / "pham_frame" if n == 20 else R / "lrp_raw_n"
        raw_t = per_task(src / f"{'pham' if n == 20 else 'raw'}_per_task.csv", f"shared_lrpTraw{n}")
        rank_t = per_task(R / "additions" / "lrp_per_task.csv", f"shared_lrpT{n}")
        for t in simple.TASKS:
            out_task.append({"n": n, "task": t, "raw": round(raw_t[t], 3), "rank": round(rank_t[t], 3),
                             "combined": other_task[(n, t)]["ours"], "lrp_combined": other_task[(n, t)]["lrp"]})
    for name, data in (("compare.csv", out), ("compare_per_task.csv", out_task)):
        with (R / "lrp_raw_n" / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"{'N':>4s} {'raw':>6s} {'(ctl/rnd)':>12s} {'rank':>6s} {'(ctl/rnd)':>12s}  {'Combined':>18s}  {'LRP-Combined':>16s}"
          f" {'Causal':>7s} {'C-gen':>6s}")
    for r in out:
        print(f"{r['n']:4d} {float(r['raw']):6.3f} ({float(r['raw_control']):.2f}/{float(r['raw_random']):.2f})"
              f" {float(r['rank']):6.3f} ({float(r['rank_control']):.2f}/{float(r['rank_random']):.2f})"
              f"  {r['combined_combo']:>11s} {float(r['combined']):.3f}  {r['lrp_combined_combo']:>9s} "
              f"{float(r['lrp_combined']):.3f} {float(r['causal']):7.3f} {float(r['causal_gen']):6.3f}")
    print("\nper task, raw - rank (mean over folds):")
    print(f"{'task':17s} " + " ".join(f"{n:>6d}" for n in SIZES))
    for t in simple.TASKS:
        d = {r["n"]: float(r["raw"]) - float(r["rank"]) for r in out_task if r["task"] == t}
        print(f"{t:17s} " + " ".join(f"{d[n]:+6.2f}" for n in SIZES))
    print("\nper task, raw - Combined (mean over folds):")
    for t in simple.TASKS:
        d = {r["n"]: float(r["raw"]) - float(r["combined"]) for r in out_task if r["task"] == t}
        print(f"{t:17s} " + " ".join(f"{d[n]:+6.2f}" for n in SIZES))


if __name__ == "__main__":
    main()
