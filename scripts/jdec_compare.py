"""J-T and J-answer (``scripts/jdec_sets.py``) against LRP-T and the other shared rules at 20 and 40 heads. No GPU.

Reads ``results/rebuild/jdec/jdec_summary.csv`` and ``jdec_per_task.csv`` (``scripts/rebuild_report.py --dir jdec
--stage jdec``), LRP-T with ranks averaged (``results/rebuild/additions/lrp_summary.csv``, ``lrp_per_task.csv``) and
with raw scores averaged (``pham_frame`` at 20, ``lrp_raw_n`` at 40), and ``results/rebuild/lrp_shared_compare.csv``
/ ``lrp_shared_per_task.csv`` (Combined per size, LRP-Combined, Causal, Causal - generic). Repeated replies: per
patched condition (fold, task, set, alpha, template mean) of a J-T set, the share of its most frequent reply
(``answers/*.jsonl.gz``); the largest per set is reported.

Writes ``results/rebuild/jdec/compare.csv`` and ``compare_per_task.csv`` (mean over folds) and prints both.

    .venv/bin/python scripts/jdec_compare.py
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import RESULTS, simple  # noqa: E402

R = RESULTS / "rebuild"
D = R / "jdec"
SIZES = (20, 40)
VARIANTS = ("jT", "jA")


def rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def per_task(path: Path, name: str) -> dict[str, float]:
    got = defaultdict(list)
    for r in rows(path):
        if r["set"] == name and r["source"] == "task":
            got[r["task"]].append(float(r["accuracy"]))
    assert all(len(got[t]) == 2 for t in simple.TASKS), (path, name)
    return {t: sum(v) / 2 for t, v in got.items()}


def modal_shares() -> dict[str, float]:
    """Per J-T set (rule sets only), the largest share of one reply in a patched condition."""
    cond = defaultdict(Counter)
    for path in sorted(D.glob("fold*/answers/*.jsonl.gz")):
        fold = path.parts[-3]
        with gzip.open(path, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r["source"] != "task" or r["heads"].startswith("random_"):
                    continue
                cond[(fold, r["task"], r["heads"], r["alpha"], r["mean_wording"])][r["predicted"].strip().lower()] += 1
    out = defaultdict(float)
    for (_, _, name, _, _), c in cond.items():
        out[name] = max(out[name], c.most_common(1)[0][1] / sum(c.values()))
    return dict(out)


def main() -> None:
    present = [v for v in VARIANTS if any(r["set"] == f"shared_{v}20" for r in rows(D / "jdec_summary.csv"))]
    summary = {r["set"]: r for r in rows(D / "jdec_summary.csv") + rows(R / "additions" / "lrp_summary.csv")
               + rows(R / "pham_frame" / "pham_summary.csv") + rows(R / "lrp_raw_n" / "raw_summary.csv")}
    other = {int(r["n"]): r for r in rows(R / "lrp_shared_compare.csv")}
    other_task = {(int(r["n"]), r["task"]): r for r in rows(R / "lrp_shared_per_task.csv")}
    modal = modal_shares()
    out, out_task = [], []
    for n in SIZES:
        row = {"n": n}
        for v in present + ["lrpT", "lrpTraw"]:
            s = summary[f"shared_{v}{n}"]
            assert s["tasks_folds"] == "28", (v, n)
            row.update({v: s["task"], f"{v}_control": s["control"], f"{v}_random": s["random"]})
        for v in present:
            row[f"{v}_max_modal_share"] = round(modal.get(f"shared_{v}{n}", float("nan")), 3)
        o = other[n]
        row.update({"combined_combo": o["ours_combo"], "combined": o["ours"], "lrp_combined_combo": o["lrp_combo"],
                    "lrp_combined": o["lrp"], "causal": o["Causal"], "causal_gen": o["Causal - gen."]})
        out.append(row)
        task_scores = {v: per_task(D / "jdec_per_task.csv", f"shared_{v}{n}") for v in present}
        task_scores["lrpT"] = per_task(R / "additions" / "lrp_per_task.csv", f"shared_lrpT{n}")
        raw_src = R / "pham_frame" / "pham_per_task.csv" if n == 20 else R / "lrp_raw_n" / "raw_per_task.csv"
        task_scores["lrpTraw"] = per_task(raw_src, f"shared_lrpTraw{n}")
        for t in simple.TASKS:
            out_task.append({"n": n, "task": t, **{v: round(task_scores[v][t], 3) for v in task_scores},
                             "combined": other_task[(n, t)]["ours"]})
    for name, data in (("compare.csv", out), ("compare_per_task.csv", out_task)):
        with (D / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    names = {"jT": "J-T", "jA": "J-answer", "jTc": "J-T centred", "lrpT": "LRP-T", "lrpTraw": "LRP-T raw"}
    print(f"{'N':>3s} " + " ".join(f"{names[v]:>20s}" for v in present + ["lrpT", "lrpTraw"])
          + f"  {'Combined':>18s} {'LRP-Combined':>16s} {'Causal':>7s}")
    for r in out:
        print(f"{r['n']:3d} " + " ".join(f"{float(r[v]):6.3f} ({float(r[f'{v}_control']):.2f}/{float(r[f'{v}_random']):.2f})"
                                         .rjust(20) for v in present + ["lrpT", "lrpTraw"])
              + f"  {r['combined_combo']:>11s} {float(r['combined']):.3f} {r['lrp_combined_combo']:>9s} "
                f"{float(r['lrp_combined']):.3f} {float(r['causal']):7.3f}")
    print("(accuracy over 14 tasks x 2 folds; in brackets control mean / random sets)")
    for r in out:
        print(f"N={r['n']}: largest share of one reply in a condition: "
              + ", ".join(f"{names[v]} {r[f'{v}_max_modal_share']}" for v in present))
    for v in present:
        print(f"\nper task, {names[v]} - LRP-T (mean over folds):")
        print(f"{'task':17s} " + " ".join(f"{n:>6d}" for n in SIZES) + "   " + " ".join(f"{names[v]}{n:>3d}" for n in SIZES))
        for t in simple.TASKS:
            d = {r["n"]: r for r in out_task if r["task"] == t}
            print(f"{t:17s} " + " ".join(f"{d[n][v] - d[n]['lrpT']:+6.2f}" for n in SIZES) + "   "
                  + " ".join(f"{d[n][v]:6.2f}" for n in SIZES))


if __name__ == "__main__":
    main()
