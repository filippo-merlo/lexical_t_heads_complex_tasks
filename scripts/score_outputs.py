"""Answer evaluation (proposal Sec. 5.5): strict and judged scores of the saved E0 outputs (no GPU).

For every task and wording in ``data/outputs`` (``scripts/run_outputs.py``):
- strict scores (``lth.evaluation``), checked against the E0 records in ``data/runs``;
- the judge's verdict (``lth.judge``; ``--no-judge`` skips it), the rule check of the
  answer it extracted where a rule exists, whether the reply begins with that answer;
- the reasons for each disagreement between the study's strict score and the judge.
Per item: ``data/outputs/{family}/{task}/{wording}.scored.jsonl.gz``. Tables in
``results/answer_eval/``: ``accuracy.csv`` (one row per task and wording), ``reasons.csv``
(disagreements by reason), ``spread.csv`` (spread across the wordings of a style).

    .venv/bin/python scripts/score_outputs.py --family simple --tasks antonym
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, RESULTS, evaluation, fu, judge, simple  # noqa: E402
from lth.store import OUTPUTS, load_outputs, outputs_path, save_outputs  # noqa: E402

RUNS = DATA / "runs"
OUT = RESULTS / "answer_eval"


def style_of(family: str, wording: str) -> str:
    if family == "simple":
        return {"inst": "rule", "shot": "example"}.get(wording[:4], "none")
    return {"examples": "example", "combined": "combined", "bare": "none"}.get(wording, "rule")


def e0_records(family: str, key: str, wording: str) -> dict[str, bool]:
    path = RUNS / family / key.replace("/", "__") / f"{wording}.json"
    return {r["item"]: bool(r["correct"]) for r in json.loads(path.read_text())} if path.exists() else {}


def rule_check(task: str, row: dict, answer: str) -> bool | None:
    """The judge's extracted answer checked by rule, where the task has one."""
    if not answer:
        return None
    ref = str(row["reference"]).strip().lower()
    text = evaluation.bare(answer).lower()
    if task == "operator_func":
        numbers = re.findall(r"-?\d+", answer.replace(",", ""))
        return int(numbers[-1]) == int(ref) if numbers else None
    if task in ("noun_class_agreement", "lexical_category_inference"):
        return text == ref
    if task == "tapatan":
        return ref in text
    if task == "set_game":
        import importlib

        set_game = importlib.import_module("tasks.set_game")
        source = row["source"]
        predicted = set_game.extract_outcome("\\boxed{" + answer + "}", source["difficulty"])
        if predicted in ("unknown", "Not boxed"):
            return None
        return bool(set_game.check_outcome_validity(predicted, source["difficulty"], source["problem"],
                                                    source["expected_result"])[0])
    return None


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


def score_unit(family, key, task, wording, mode, rules, cache, use_judge):
    rows = load_outputs(outputs_path(family, key, wording))
    if family == "simple":
        strict = {r["item"]: evaluation.strict_simple(r) for r in rows}
        questions = {r["item"]: evaluation.question_simple(task, r) for r in rows}
    else:
        strict = evaluation.strict_complex(key, wording, mode, rules, rows)
        original = load_outputs(outputs_path(family, key, "orig"))
        questions = {r["item"]: r["prompt"] for r in original}
    reasoning = family == "complex" and task in fu.REASONING_TASKS
    e0 = e0_records(family, key, wording)
    e0_reasoning = e0_records("reasoning", key, wording) if reasoning else {}
    verdicts = {}
    if use_judge:
        cases = [{"item": r["item"], "question": questions[r["item"]], "reference": str(r["reference"]).strip(),
                  "reply": evaluation.reply_for_judge(family, r)} for r in rows]
        by_key = judge.judge(cases, cache)
        verdicts = {c["item"]: by_key[c["key"]] for c in cases}
    scored = []
    for r in rows:
        item, s = r["item"], strict[r["item"]]
        out = {"item": item, **s, "study_strict": evaluation.primary_strict(task, s) if family == "complex"
               else s["strict"]}
        if item in e0:
            out["e0_strict"] = e0[item]
        if item in e0_reasoning:
            out["e0_reasoning"] = e0_reasoning[item]
        if item in verdicts:
            v = verdicts[item]
            answer = v["answer"] or ""
            out.update({"judged": v["verdict"] == "correct",
                        **{f"judge_{k}": v[k] for k in ("verdict", "answer", "relation", "several_answers",
                                                        "reference_suspect", "note")},
                        "straight_judged": evaluation.straight(family, r, answer),
                        "rule_check": rule_check(task, r, answer),
                        "reasons": evaluation.reasons(family, task, r, s, v)})
        scored.append(out)
    save_outputs(outputs_path(family, key, wording).with_name(f"{wording}.scored.jsonl.gz"), scored)
    return scored


def summary(family, key, wording, scored):
    n = len(scored)
    mean = lambda field: sum(bool(x.get(field)) for x in scored) / n  # noqa: E731
    row = {"family": family, "task": key, "style": style_of(family, wording), "wording": wording, "n": n,
           "strict": mean("strict"), "strict_full": mean("strict_full"), "study_strict": mean("study_strict")}
    e0 = [x for x in scored if "e0_strict" in x]
    row["e0_items"] = len(e0)
    row["e0_mismatch"] = sum(x["e0_strict"] != x["strict"] for x in e0)
    e0r = [x for x in scored if "e0_reasoning" in x]
    row["e0_reasoning_mismatch"] = sum(x["e0_reasoning"] != x["strict_full"] for x in e0r) if e0r else ""
    if "judged" in scored[0]:
        row["judged"] = mean("judged")
        row["straight_judged"] = mean("straight_judged")
        for a in (True, False):
            for b in (True, False):
                row[f"strict_{'right' if a else 'wrong'}_judge_{'right' if b else 'wrong'}"] = sum(
                    x["study_strict"] == a and x["judged"] == b for x in scored)
        checked = [x for x in scored if x["rule_check"] is not None]
        row["rule_checked"] = len(checked)
        row["rule_disagrees"] = sum(x["rule_check"] != x["judged"] for x in checked)
        row["no_answer"] = sum(x["judge_verdict"] == "no_answer" for x in scored)
    return row


def spread(rows):
    """sd across the wordings of a style under each score, Spearman between the scores' rankings."""
    out = []
    groups = {}
    for r in rows:
        if r["style"] in ("rule", "example") and "judged" in r:
            groups.setdefault((r["family"], r["task"], r["style"]), []).append(r)
    for (family, task, style), rs in groups.items():
        strict = np.array([r["study_strict"] for r in rs])
        judged = np.array([r["judged"] for r in rs])
        rank = lambda a: np.argsort(np.argsort(-a, kind="stable"), kind="stable")  # noqa: E731
        rho = float(np.corrcoef(rank(strict), rank(judged))[0, 1]) if len(rs) > 2 else float("nan")
        out.append({"family": family, "task": task, "style": style, "wordings": len(rs),
                    "sd_strict": strict.std(), "sd_judged": judged.std(),
                    "range_strict": strict.max() - strict.min(), "range_judged": judged.max() - judged.min(),
                    "spearman": rho, "best_strict": rs[int(strict.argmax())]["wording"],
                    "best_judged": rs[int(judged.argmax())]["wording"]})
    return out


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
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    cache = {} if args.no_judge else judge.load_cache()
    rows, reason_rows = [], []
    for family in ("simple", "complex") if args.family == "all" else (args.family,):
        for key, task, wording, mode, rules in units(family, args.tasks, args.levels, args.variants):
            if not outputs_path(family, key, wording).exists():
                continue
            scored = score_unit(family, key, task, wording, mode, rules, cache, not args.no_judge)
            row = summary(family, key, wording, scored)
            rows.append(row)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)
            primary, anywhere = Counter(), Counter()
            for x in scored:
                if x.get("reasons"):
                    direction = "strict_wrong_judge_right" if x["judged"] else "strict_right_judge_wrong"
                    primary[direction, x["reasons"][0]] += 1
                    anywhere.update((direction, r) for r in x["reasons"])
            reason_rows += [{"family": family, "task": key, "wording": wording, "direction": d, "reason": r,
                             "primary": primary[d, r], "any": anywhere[d, r]} for d, r in sorted(anywhere)]
    write("accuracy.csv", rows)
    write("reasons.csv", reason_rows)
    write("spread.csv", spread(rows))


if __name__ == "__main__":
    main()
