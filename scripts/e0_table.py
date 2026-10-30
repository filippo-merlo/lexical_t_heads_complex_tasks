"""E0: accuracy of every wording of every task, and the share answered straight away (no GPU).

Family ``complex`` holds the complex tasks under Fu et al.'s answer limits, ``reasoning``
the tasks rerun in reasoning mode (accuracy after reasoning).

Writes ``results/e0_accuracy.csv`` with one row per task and wording: its style (``none``
for the prompts without task information), the number of items, accuracy, the share of
replies that begin with the answer, the number of identification prompts (P+: correct and
straight away, or in reasoning mode correct; at most 100), and chance. Chance is the majority label's share for the
tasks with a fixed label set (Noun Class Agreement, Lexical Category Inference, Tapatan)
and 0 for open answers.

Also writes ``results/e0_outcomes.csv``, the error analysis of the complex tasks: per task
and wording, how many replies have each outcome of ``lth.fu.outcome`` (scoring unchanged),
and the counts of its details.

    .venv/bin/python scripts/e0_table.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import RESULTS, simple  # noqa: E402
from lth.store import done, load, run_dir  # noqa: E402

LABEL_FIELDS = {"noun_class_agreement": "expected_label", "lexical_category_inference": "expected_answer",
                "tapatan": "expected_result"}


def complex_style(wording: str) -> str:
    return {"examples": "example", "combined": "combined", "bare": "none"}.get(wording, "rule")


def chance(level, directory: Path) -> float:
    field = LABEL_FIELDS.get(level.task)
    if field is None:
        return 0.0
    files = sorted((directory / "fu_out" / "orig").rglob("*.json"))
    labels = Counter(str(r[field]) for r in json.loads(files[-1].read_text())["results"])
    return max(labels.values()) / sum(labels.values())


def row(family, task, style, wording, run, chance_level):
    records = run["records"]
    return {"family": family, "task": task, "style": style, "wording": wording, "n": len(records),
            "accuracy": sum(r["correct"] for r in records) / len(records),
            "straight": sum(r["straight"] for r in records) / len(records),
            "p_plus": len(run["id_items"]), "chance": chance_level}


DETAILS = {"not a valid set": "not_a_set", "card not on the board": "card_not_on_board",
           "right": "unformatted_right", "label in code": "label_in_code"}


def outcome_row(family, level, wording, directory, run, tokenizer):
    from lth import fu

    records = {fu.item_id(r): r for r in fu.recorded_results(directory / "fu_out" / wording)}
    counts = Counter()
    for r in run["records"]:
        reply_tokens = r.get("reply_tokens") or len(tokenizer(r["response"], add_special_tokens=False).input_ids)
        name, detail = fu.outcome(level, records[r["item"]], r["correct"], reply_tokens,
                                  reasoning=family == "reasoning")
        counts[name] += 1
        if detail in DETAILS:
            counts[DETAILS[detail]] += 1
    if counts["correct"] != sum(r["correct"] for r in run["records"]):
        raise RuntimeError(f"{family} {level.key} {wording}: outcomes disagree with the scores")
    return {"family": family, "task": level.key, "style": complex_style(wording), "wording": wording,
            "n": len(run["records"]), **{k: counts[k] for k in (*fu.OUTCOMES, *DETAILS.values())}}


def write(name: str, rows: list[dict]) -> None:
    with (RESULTS / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {RESULTS / name} ({len(rows)} rows)")


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    from transformers import AutoTokenizer

    from lth import LENS_KEY, fu
    from src.config import LENSES

    tokenizer = AutoTokenizer.from_pretrained(LENSES[LENS_KEY].hf_model_id)
    rows, outcomes = [], []
    for task in simple.TASKS:
        directory = run_dir("simple", task)
        for style, wordings in [*simple.WORDINGS.items(), ("none", (simple.ZERO,))]:
            for wording in wordings:
                if done(directory, wording):
                    rows.append(row("simple", task, style, wording, load(directory, wording), 0.0))
    for family in ("complex", "reasoning"):
        for key in fu.STUDY_LEVELS:
            level = fu.TASK_LEVELS[key]
            directory = run_dir(family, key)
            if not done(directory, "orig"):
                continue
            chance_level = chance(level, directory)
            for wording in ["orig", *[f"w{i}" for i in range(1, 7)], "examples", "combined", "bare"]:
                if done(directory, wording):
                    run = load(directory, wording)
                    rows.append(row(family, key, complex_style(wording), wording, run, chance_level))
                    outcomes.append(outcome_row(family, level, wording, directory, run, tokenizer))
    RESULTS.mkdir(exist_ok=True)
    write("e0_accuracy.csv", rows)
    write("e0_outcomes.csv", outcomes)


if __name__ == "__main__":
    main()
