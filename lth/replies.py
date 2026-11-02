"""The patched replies of the simple-task causal checks, for inspection
(``notebooks/patching_outputs.py``).

Every causal check keeps its per-item answers in ``<results>/answers/patching_<family>.jsonl.gz``
(``scripts/patch.py``). ``PROCEDURES`` names the head-identification procedures that have such
files; ``load`` returns one procedure's replies to one task in one check, from a per-task split
under ``data/replies`` that is rebuilt whenever its source file is newer. ``kind`` sorts a reply
into what the model did instead of, or as well as, answering.
"""

from __future__ import annotations

import gzip
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from lth import DATA, RESULTS, simple

SPLIT = DATA / "replies"
FAMILIES = {"simple": "check 1: no task information", "simple-wrong": "check 2: wrongly answered 1-shot"}
# name -> (results directory, description); in the order of the proposal (E1, E1b)
PROCEDURES = {
    "E1": (RESULTS, "most accurate template (rule) or 5 shots (example)"),
    "k=1": (RESULTS / "consensus_k1", "at least 1 of 5 templates (union)"),
    "k=3": (RESULTS / "consensus_k3", "at least 3 of 5 templates"),
    "k=5": (RESULTS / "consensus", "all 5 templates"),
    "k=1 shared": (RESULTS / "consensus_k1_shared", "J-head k=1, heads in the sets of at least 2 tasks"),
    "k=1 unshared": (RESULTS / "consensus_k1_unshared", "J-head k=1, heads in no other task's set"),
}
HEAD_SETS = {"logithead": "Logit-head", "jhead": "J-head", "jhead_only": "J only"}
KINDS = ("correct", "contains the answer", "names the task", "repeats the input", "other form of the input",
         "answer of another item", "other text", "empty")
# Words with which a reply names the task instead of doing it.
TASK_WORDS = {
    "antonym": r"opposite|antonym", "synonym": r"synonym|same meaning",
    "singular-plural": r"plural|singular", "present-past": r"\bpast\b",
    "country-capital": r"capital", "country-currency": r"currenc",
    "landmark-country": r"\bcountry\b|located", "park-country": r"\bcountry\b|located",
    "product-company": r"compan|manufactur|made by|brand",
    "next_item": r"\bnext (one|item|number|letter|word|day|month)|comes after",
    "prev_item": r"previous (one|item|number|letter|word|day|month)|comes before|precedes",
    "english-french": r"in french|french (word|for|translation)|translat",
    "english-german": r"in german|german (word|for|translation)|translat",
    "english-spanish": r"in spanish|spanish (word|for|translation)|translat",
}


def norm(text: str) -> str:
    return re.sub(r"[^\w' -]", "", text.strip().lower()).strip()


def procedures() -> list[str]:
    """The procedures whose answers exist."""
    return [name for name, (d, _) in PROCEDURES.items() if (d / "answers" / "patching_simple.jsonl.gz").exists()]


def split(procedure: str, family: str) -> Path:
    """The per-task split of a procedure's answers in one check, rebuilt if out of date."""
    source = PROCEDURES[procedure][0] / "answers" / f"patching_{family}.jsonl.gz"
    out = SPLIT / procedure.replace(" ", "_").replace("=", "") / family
    stamp = out / ".source_mtime"
    if not (stamp.exists() and stamp.read_text() == str(source.stat().st_mtime)):
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        with gzip.open(source, "rt") as f:
            for line in f:
                task = json.loads(line)["task"]
                if task not in files:
                    files[task] = gzip.open(out / f"{task}.jsonl.gz", "wt")
                files[task].write(line)
        for handle in files.values():
            handle.close()
        stamp.write_text(str(source.stat().st_mtime))
    return out


@lru_cache(maxsize=32)
def load(procedure: str, family: str, task: str) -> pd.DataFrame | None:
    """One procedure's replies to one task in one check, with the input word and the reply's
    kind; the unpatched replies have ``source == "none"``. None if the task was not patched."""
    path = split(procedure, family) / f"{task}.jsonl.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt") as f:
        frame = pd.DataFrame([json.loads(line) for line in f])
    data = simple.dataset(task)
    frame["item"] = frame["item"].astype(int)
    frame["input"] = [data[i]["input"] for i in frame["item"]]
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="coerce")
    if "reply" not in frame:
        frame["reply"] = frame["predicted"]
    frame["reply"] = frame["reply"].fillna(frame["predicted"])
    answers = {norm(d["output"]) for d in data}
    frame["kind"] = [kind(task, r.input, r.expected, r.predicted, r.correct, answers)
                     for r in frame.itertuples()]
    return frame


def kind(task: str, input_: str, expected: str, predicted: str, correct: bool, answers: set[str]) -> str:
    """What a reply is, checked in the order of ``KINDS``: correct (the flexible score); contains the
    answer as a whole word but not at its start (the flexible score rejects it); names the task
    (a phrase of ``TASK_WORDS``); repeats the input (starts with it); another form of the
    input (its first word shares the input's stem, at least four letters); the answer of
    another item of the task; other text; empty."""
    if correct:
        return "correct"
    text = norm(predicted)
    if not text:
        return "empty"
    target = norm(expected)
    if target and re.search(rf"(?<!\w){re.escape(target)}(?!\w)", text):
        return "contains the answer"
    if re.search(TASK_WORDS[task], text):
        return "names the task"
    source = norm(input_)
    if text == source or text.startswith(source + " "):
        return "repeats the input"
    first, word = text.split()[0], source.split()[0] if source else ""
    stem = word[:-1] if word.endswith("e") else word
    if len(stem) >= 4 and first.startswith(stem[:max(4, len(stem) - 1)]):
        return "other form of the input"
    if text in answers or text.split(";")[0] in answers:
        return "answer of another item"
    return "other text"


def patched_prompt(task: str, family: str, item: int) -> str:
    """The prompt that was patched: the zero-shot prompt (check 1) or the 1-shot one (check 2)."""
    return _prompts(task, simple.FIXING if family == "simple-wrong" else simple.ZERO)[item]


@lru_cache(maxsize=32)
def _prompts(task: str, wording: str) -> tuple[str, ...]:
    return tuple(simple.prompts(task, wording)[0])


def head_sets(procedure: str, task: str) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """``(style, readout)`` -> the (layer, head) pairs of the procedure's sets for the task."""
    heads = json.loads((PROCEDURES[procedure][0] / "heads.json").read_text())["simple"].get(task, {})
    return {(style, readout): [(l, h) for l, h, _ in sets[readout]["heads"]]
            for style, sets in heads.items() for readout in ("logithead", "jhead") if readout in sets}
