"""How well the lens estimates of ``scripts/davidson_fv_lens_cie.py`` match the measured CIE (no GPU).

Per task, readout and measure (the CIE and the estimates averaged over a task's three kinds,
as Davidson et al. average them):
- Spearman and Pearson correlation with the measured CIE over all 1,024 heads;
- top-20 overlap: how many of the 20 heads with the largest measured CIE are among the 20
  with the largest estimate (chance: 20 * 20 / 1024 = 0.39).
Over all tasks (mean over tasks and kinds, Davidson et al.'s ranking): how many of the run's
30 FV heads are among the top 30 / 60 / 100 by each estimate (chance at 30: 0.88), and the
rank of each FV head. Also the measured CIE against the saved one (same prompts expected).

Writes ``results/<--out>/report.md`` and ``report.json``.

    .venv/bin/python scripts/davidson_fv_lens_cie_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import READOUTS, RESULTS  # noqa: E402

MEASURES = ("logit", "logprob", "prob")
KINDS = ("equiprobable", "wikitext", "other_task")
N_HEADS = 32


def ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    a = a.ravel()
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    for v in np.unique(a[np.r_[False, np.diff(a[order]) == 0]]):  # tied values
        r[a == v] = r[a == v].mean()
    return r


def spearman(a, b) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def pearson(a, b) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def top(a: np.ndarray, n: int) -> set[int]:
    return set(np.argsort(-a.ravel(), kind="stable")[:n].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", default="davidson_fv_all14_30")
    parser.add_argument("--out", default="davidson_fv_lens_cie")
    args = parser.parse_args()

    out = RESULTS / args.out
    est = np.load(out / "estimates.npz")
    measured = np.load(out / "measured.npz")
    saved = np.load(RESULTS / args.run / "cie.npz")
    prompts = json.loads((out / "prompts.json").read_text())
    fv = [l * N_HEADS + h for l, h, _ in json.loads((RESULTS / args.run / "heads.json").read_text())["heads"]]
    tasks = list(prompts)

    report = {"tasks": {}, "overall": {}, "saved_check": {}}
    meas_task = {}
    for task in tasks:
        kinds = np.array([m["kind"] for m in prompts[task]])
        per_kind = np.stack([measured[task][kinds == k].mean(0) for k in KINDS])  # [3, L, H]
        report["saved_check"][task] = float(np.abs(per_kind - np.stack([saved[f"{task}/{k}"] for k in KINDS])).max())
        meas_task[task] = per_kind.mean(0)
        row = {}
        for readout in READOUTS:
            for measure in MEASURES:
                e = est[f"{task}/{readout}/{measure}"].mean(0)
                row[f"{readout}/{measure}"] = dict(
                    spearman=spearman(e, meas_task[task]), pearson=pearson(e, meas_task[task]),
                    top20=len(top(e, 20) & top(meas_task[task], 20)))
        report["tasks"][task] = row

    overall_meas = np.mean([meas_task[t] for t in tasks], axis=0)
    report["overall"]["measured_top30_is_fv"] = top(overall_meas, 30) == set(fv)
    for readout in READOUTS:
        for measure in MEASURES:
            e = np.mean([est[f"{t}/{readout}/{measure}"].mean(0) for t in tasks], axis=0)
            order = np.argsort(-e.ravel(), kind="stable").tolist()
            report["overall"][f"{readout}/{measure}"] = dict(
                spearman=spearman(e, overall_meas), pearson=pearson(e, overall_meas),
                fv_in_top={n: len(set(order[:n]) & set(fv)) for n in (30, 60, 100)},
                fv_ranks={f"L{s // N_HEADS}H{s % N_HEADS}": order.index(s) + 1 for s in fv},
                top30=[f"L{s // N_HEADS}H{s % N_HEADS}" for s in order[:30]])

    lines = ["# Lens estimate of the FV head score vs the measured score", "",
             f"Measured CIE vs saved cie.npz, max |diff| over tasks: {max(report['saved_check'].values()):.2e}; "
             f"top 30 of the measured mean = the FV heads: {report['overall']['measured_top30_is_fv']}.", "",
             "## Over all 14 tasks (Davidson's ranking: mean over tasks and kinds)", "",
             "| readout / measure | Spearman | Pearson | FV heads in top 30 | top 60 | top 100 |",
             "|---|---|---|---|---|---|"]
    for key, v in report["overall"].items():
        if isinstance(v, dict):
            lines.append(f"| {key} | {v['spearman']:.2f} | {v['pearson']:.2f} | {v['fv_in_top'][30]} | "
                         f"{v['fv_in_top'][60]} | {v['fv_in_top'][100]} |")
    lines += ["", "Chance: 0.88 FV heads in a random top 30, 1.8 in a top 60, 2.9 in a top 100.", "",
              "## Per task (mean over kinds): Spearman / Pearson / top-20 overlap (chance 0.39)", ""]
    keys = [f"{r}/{m}" for r in READOUTS for m in MEASURES]
    lines += ["| task | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for task, row in report["tasks"].items():
        lines.append(f"| {task} | " + " | ".join(
            f"{row[k]['spearman']:.2f} / {row[k]['pearson']:.2f} / {row[k]['top20']}" for k in keys) + " |")
    lines += ["", "## Rank of each FV head (1 = top) under each estimate, over all tasks", "",
              "| FV head (rank by measured) | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for i, s in enumerate(fv, 1):
        name = f"L{s // N_HEADS}H{s % N_HEADS}"
        lines.append(f"| {name} ({i}) | " + " | ".join(str(report['overall'][k]['fv_ranks'][name]) for k in keys) + " |")
    lines += ["", "## Top 30 by the J-head probability estimate (FV heads marked *)", "",
              ", ".join(h + ("*" if h in {f"L{s // N_HEADS}H{s % N_HEADS}" for s in fv} else "")
                        for h in report["overall"]["jhead/prob"]["top30"])]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
