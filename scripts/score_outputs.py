"""Answer evaluation (proposal Sec. 5.5): strict and flexible scores of the saved E0 outputs (no GPU).

For every task and wording in ``data/outputs`` (``scripts/run_outputs.py``):
- strict scores (``lth.evaluation``), checked against the E0 records in ``data/runs``, and
  the flexible score (simple tasks: the whole reference at the reply's start; complex tasks:
  the answer the strict score reads, in any notation).
Per item: ``data/outputs/{family}/{task}/{wording}.scored.jsonl.gz``. Table:
``results/answer_eval/accuracy.csv`` (one row per task and wording).

    .venv/bin/python scripts/score_outputs.py --family simple --tasks antonym
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, RESULTS, evaluation, fu, simple  # noqa: E402
from lth.store import load_outputs, outputs_path, save_outputs  # noqa: E402

RUNS = DATA / "runs"
OUT = RESULTS / "answer_eval"


def style_of(family: str, wording: str) -> str:
    if family == "simple":
        return {"inst": "rule", "shot": "example"}.get(wording[:4], "none")
    return {"examples": "example", "combined": "combined", "bare": "none"}.get(wording, "rule")


def e0_records(family: str, key: str, wording: str) -> dict[str, bool]:
    path = RUNS / family / key.replace("/", "__") / f"{wording}.json"
    return {r["item"]: bool(r["correct"]) for r in json.loads(path.read_text())} if path.exists() else {}


def units(family: str, tasks, levels, variant_names):
    """(task key, task name, wording, mode, rules) for every saved output asked for."""
    if family == "simple":
        for task in tasks:
            for wording in [w for ws in simple.WORDINGS.values() for w in ws] + [simple.ZERO]:
                yield task, task, wording, None, None
        return
    from run_complex import variants

    for key in levels:
        for name, mode, rules in variants(fu.TASK_LEVELS[key]):
            if not variant_names or name in variant_names:
                yield key, fu.TASK_LEVELS[key].task, name, mode, rules


def score_unit(family, key, wording, mode, rules):
    rows = load_outputs(outputs_path(family, key, wording))
    if family == "simple":
        strict = {r["item"]: {**evaluation.strict_simple(r), "flexible": evaluation.flexible_simple(r)}
                  for r in rows}
    else:
        strict = evaluation.strict_complex(key, wording, mode, rules, rows)
        for item, scores in evaluation.flexible_complex(key, wording, mode, rules, rows).items():
            strict[item].update(scores)
    e0 = e0_records(family, key, wording)
    e0_reasoning = e0_records("reasoning", key, wording) if family == "complex" else {}
    scored = []
    for r in rows:
        item, s = r["item"], strict[r["item"]]
        # the study's strict score for a complex task: the whole reply, for every task
        out = {"item": item, **s, "study_strict": s["strict_full"] if family == "complex" else s["strict"]}
        if item in e0:
            out["e0_strict"] = e0[item]
        if item in e0_reasoning:
            out["e0_reasoning"] = e0_reasoning[item]
        scored.append(out)
    save_outputs(outputs_path(family, key, wording).with_name(f"{wording}.scored.jsonl.gz"), scored)
    return scored


def summary(family, key, wording, scored):
    n = len(scored)
    mean = lambda field: sum(bool(x.get(field)) for x in scored) / n  # noqa: E731
    row = {"family": family, "task": key, "style": style_of(family, wording), "wording": wording, "n": n,
           "strict": mean("strict"), "strict_full": mean("strict_full"), "study_strict": mean("study_strict")}
    row["flexible"] = mean("flexible")
    e0 = [x for x in scored if "e0_strict" in x]
    row["e0_items"] = len(e0)
    row["e0_mismatch"] = sum(x["e0_strict"] != x["strict"] for x in e0)
    e0r = [x for x in scored if "e0_reasoning" in x]
    row["e0_reasoning_mismatch"] = sum(x["e0_reasoning"] != x["strict_full"] for x in e0r) if e0r else ""
    return row


def write(name, rows):
    if not rows:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT / name} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--family", choices=("simple", "complex", "all"), default="all")
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS))
    parser.add_argument("--levels", nargs="+", default=list(fu.STUDY_LEVELS))
    parser.add_argument("--variants", nargs="+", default=None)
    args = parser.parse_args()

    rows = []
    for family in ("simple", "complex") if args.family == "all" else (args.family,):
        for key, _task, wording, mode, rules in units(family, args.tasks, args.levels, args.variants):
            if not outputs_path(family, key, wording).exists():
                continue
            scored = score_unit(family, key, wording, mode, rules)
            row = summary(family, key, wording, scored)
            rows.append(row)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)
    write("accuracy.csv", rows)


if __name__ == "__main__":
    main()
