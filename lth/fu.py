"""The five complex tasks of Fu et al. (2026), run through their released code.

Their runners (``third_party/rules-vs-examples/code/tasks``) sample the test items and the
demonstrations with their seed, build the rules-only, examples-only and combined prompts,
and parse the answers. This module supplies what the release leaves to its caller:

- the model call, through the runners' ``query_batch_fn`` hook;
- in rules mode, the rule text: an original prompt (proposal App. A), one of our wordings,
  or the input block alone (the prompt without task information);
- the scoring of Tapatan, whose runner records the raw answer only, and of Noun Class
  Agreement, whose runner records the parsed answer next to the expected one;
- whether the model answered straight away, with the answer as its first output.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from lth import DATA, FU

os.environ.setdefault("RULES_VS_EXAMPLES_DATA_DIR", str(DATA / "fu" / "data"))
if str(FU / "code") not in sys.path:
    sys.path.insert(0, str(FU / "code"))

MODEL_NAME = "Llama-3.1-8B-Instruct"
SEED = 123  # Fu's main.py default
MODES = ("rules", "examples", "combined")

# Fu's main.py: answer-length limits, and the demonstration counts its dispatcher passes
# (the other runners choose their own per-level default).
MAX_TOKENS = {"set_game": 50, "operator_func": 50, "tapatan": 4,
              "noun_class_agreement": 8, "lexical_category_inference": 2}
DEMOS = {"set_game": 20, "operator_func": 1}
# Reasoning mode (proposal Sec. 5.3): the tasks on which E0 found no direct answers (none of
# 300 items answered straight away, in any variant) are run with Fu's instruction-tuned
# limit for Operator Function, so the model can reason first.
REASONING_TASKS = ("set_game", "operator_func", "tapatan")
REASONING_MAX_TOKENS = int(os.environ.get("LTH_REASONING_MAX_TOKENS", 6000))  # override: smoke runs only
RUNNERS = {"set_game": "run_set_game_experiment", "operator_func": "run_operator_func_experiment",
           "tapatan": "run_tapatan_experiment",
           "noun_class_agreement": "run_noun_class_agreement_experiment",
           "lexical_category_inference": "run_lexical_category_inference_experiment"}
# Test items per level, drawn with Fu's sampler and seed. Fu use 100 (all 600 for Lexical
# Category Inference); 300 leaves items for the causal check's holdout (lth.patching).
N_TEST = {"lexical_category_inference": 600}
N_TEST_DEFAULT = 300


@dataclass(frozen=True)
class Original:
    """One rules-only prompt of Fu et al.: a section of their Markdown templates.

    ``input_marker`` starts the input block, the lines that introduce the test input and
    the answer; the block alone is the prompt without task information.
    """

    key: str
    markdown: str
    sections: tuple[tuple[str, ...], ...]
    input_marker: str
    placeholders: tuple[str, ...]


def _levels(task, markdown, marker, placeholders):
    return [Original(f"{task}/{name}", markdown, (("Rules", f"Difficulty {d}"),), marker, placeholders)
            for d, name in ((1, "easy"), (2, "medium"), (3, "hard"))]


ORIGINALS = {o.key: o for o in [
    *_levels("set_game", "set_game.md", "Here is the board:", ("board",)),
    Original("tapatan/all", "tapatan.md",
             tuple(("Final Board State Variant", "Rules", d) for d in ("Easy", "Medium", "Hard")),
             "BOARD:", ("board_size", "line_length", "board_state")),
    *_levels("operator_func", "operator_function.md", "Problem:", ("problem",)),
    *_levels("noun_class_agreement", "noun_class_agreement.md", "Sentence:", ("sentence",)),
    *[Original(f"lexical_category_inference/{name}", "lexical_category_inference.md", sections,
               "CATEGORY DEFINITIONS:", ("prompt_labels", "label_block", "candidate_block"))
      for name, sections in (("easy", (("Shared", "Difficulty 1"),)),
                             ("or", (("Either", "Difficulty 2"), ("Either", "Difficulty 3"))),
                             ("and", (("Both", "Difficulty 2"), ("Both", "Difficulty 3"))))],
]}


@dataclass(frozen=True)
class TaskLevel:
    """One complex task at one difficulty level (and logic variant): a task of the study."""

    key: str
    task: str
    original: str
    runner_kwargs: dict = field(hash=False)

    @property
    def n_test(self) -> int:
        return N_TEST.get(self.task, N_TEST_DEFAULT)


TASK_LEVELS = {t.key: t for t in [
    *[TaskLevel(f"{task}/{name}", task, f"{task}/{name}", {"difficulty": d})
      for task in ("set_game", "operator_func", "noun_class_agreement")
      for d, name in ((1, "easy"), (2, "medium"), (3, "hard"))],
    *[TaskLevel(f"tapatan/{name}", "tapatan", "tapatan/all",
                {"difficulty": name, "input_format": "final_board_state"})
      for name in ("easy", "medium", "hard")],
    TaskLevel("lexical_category_inference/easy", "lexical_category_inference",
              "lexical_category_inference/easy", {"difficulty": "d1", "logic_condition": "shared"}),
    *[TaskLevel(f"lexical_category_inference/{name}-{op}", "lexical_category_inference",
                f"lexical_category_inference/{op}", {"difficulty": d, "logic_condition": logic})
      for d, name in (("d2", "medium"), ("d3", "hard"))
      for op, logic in (("or", "either"), ("and", "both"))],
]}


# The levels in the study: only the easy level of each task (user decision, 2026-10-27).
# Data for the other levels may be on disk; no script uses it.
STUDY_LEVELS = tuple(key for key in TASK_LEVELS if key.endswith("/easy"))


# ---- rule text -------------------------------------------------------------------------

def original_template(original: Original) -> str:
    """The rules-only prompt as a ``str.format`` template (literal braces doubled)."""
    from prompts.markdown_loader import get_markdown_section

    texts = {get_markdown_section(original.markdown, section) for section in original.sections}
    if len(texts) != 1:
        raise ValueError(f"{original.key}: its sections differ")
    return texts.pop()


def display(template: str) -> str:
    """The template as a reader sees it (App. A): literal braces single."""
    return template.replace("{{", "{").replace("}}", "}")


def template_from_display(text: str, placeholders: tuple[str, ...]) -> str:
    """Inverse of :func:`display` for a text whose only fields are ``placeholders``."""
    template = text.replace("{", "{{").replace("}", "}}")
    for name in placeholders:
        template = template.replace("{{" + name + "}}", "{" + name + "}")
    return template


def input_block(template: str, original: Original) -> str:
    """The prompt without task information: the original's input block alone."""
    return template[template.index(original.input_marker):]


@contextlib.contextmanager
def rule_text(markdown: str, template: str):
    """Make Fu's builders read ``template`` wherever they read a section of ``markdown``."""
    from prompts import markdown_loader

    original = markdown_loader.get_markdown_section

    def section(filename, heading_path):
        return template if filename == markdown else original(filename, heading_path)

    markdown_loader.get_markdown_section = section
    try:
        yield
    finally:
        markdown_loader.get_markdown_section = original


# ---- running ---------------------------------------------------------------------------

def run(level: TaskLevel, mode: str, generate: Callable[[list[str]], list[str]], output_dir: Path, *,
        rules: str | None = None, n_test: int | None = None, batch: int = 1,
        reasoning: bool = False) -> list[dict]:
    """Fu's runner for one task level and mode, answering its prompts with ``generate``.

    ``generate`` receives up to ``batch`` prompts at a time. ``rules`` replaces the
    rules-only template (rules mode only). In reasoning mode the runner is told the
    reasoning-mode output limit. Returns the runner's per-item records, each with the
    ``prompt`` and the ``raw_response``.
    """
    if rules is not None and mode != "rules":
        raise ValueError("a rule text applies to rules mode only")
    module = importlib.import_module(f"tasks.{level.task}")
    answered: dict[str, str] = {}

    async def query_batch(prompts):
        responses = generate(list(prompts))
        answered.update(zip(prompts, responses))
        return responses

    kwargs = dict(level.runner_kwargs, mode=mode, num_test_samples=n_test or level.n_test,
                  num_demos=DEMOS.get(level.task), output_dir=str(output_dir),
                  model_name=MODEL_NAME, seed=SEED,
                  max_tokens=REASONING_MAX_TOKENS if reasoning else MAX_TOKENS[level.task],
                  query_batch_fn=query_batch, query_batch_size=batch)
    context = (rule_text(ORIGINALS[level.original].markdown, rules)
               if rules is not None else contextlib.nullcontext())
    # Set Game shuffles each board with the unseeded global RNG (set_game.py:349); seeding
    # it makes the boards reproducible and the same in every prompt variant.
    random.seed(SEED)
    with context:
        payload = asyncio.run(getattr(module, RUNNERS[level.task])(**kwargs))
    results = payload["results"]
    # The runners turn a failed model call into its error text; make sure none did.
    for result in results:
        if answered.get(result["prompt"]) != result["raw_response"]:
            raise RuntimeError(f"{level.key} {mode}: a response did not come from the model")
    return results


# ---- scoring ---------------------------------------------------------------------------

_YES_NO_START = re.compile(r"\s*(yes|no)\b", re.IGNORECASE)
_TAPATAN_LABELS = ("a win", "b win", "continue")
_TAPATAN_LABEL = re.compile(r"\b(A win|B win|continue)\b")
_BOXED = re.compile(r"\\boxed\{(.+?)\}", re.S)


def item_id(result: dict) -> str:
    return str(result.get("id", result.get("sample_id")))


def score(level: TaskLevel, result: dict, *, reasoning: bool = False) -> tuple[bool, bool]:
    """``(correct, straight)``: the answer is right; the answer is the first thing written.

    Set Game, Operator Function and Lexical Category Inference use the runners' own
    ``is_correct``; Noun Class Agreement compares the runner's parsed answer with the
    label. Tapatan's release has no check: its first line must be the expected label, or
    in reasoning mode the first label anywhere in the output must be (Sec. 5.3).
    """
    raw = result["raw_response"]
    if level.task in ("set_game", "operator_func"):
        return bool(result["is_correct"]), raw.lstrip().startswith("\\boxed{")
    if level.task == "lexical_category_inference":
        return bool(result["is_correct"]), result["first_token_parse_method"] == "first_answer_token"
    if level.task == "noun_class_agreement":
        start = _YES_NO_START.match(raw)
        straight = bool(start) and start.group(1).lower() == result["prediction"].lower()
        return result["prediction"] == result["expected_label"], straight
    if level.task == "tapatan":
        first = raw.strip().split("\n")[0].strip().rstrip(".").lower()
        straight = first in _TAPATAN_LABELS
        if reasoning:
            label = _TAPATAN_LABEL.search(raw)
            return bool(label) and label.group(1) == str(result["expected_result"]), straight
        return first == str(result["expected_result"]).lower(), straight
    raise ValueError(level.task)


def answer(level: TaskLevel, result: dict, *, reasoning: bool = False) -> tuple:
    """``(expected, predicted)``: the two answers ``score`` compares, as the runner records
    them (Tapatan: the first line, or in reasoning mode the first label; None if absent)."""
    if level.task in ("set_game", "operator_func"):
        return result["expected_result"], result["predicted_result"]
    if level.task == "lexical_category_inference":
        return result["expected_answer"], result["predicted_answer"]
    if level.task == "noun_class_agreement":
        return result["expected_label"], result["prediction"]
    if level.task == "tapatan":
        raw = result["raw_response"]
        if reasoning:
            label = _TAPATAN_LABEL.search(raw)
            return result["expected_result"], label.group(1) if label else None
        return result["expected_result"], raw.strip().split("\n")[0].strip().rstrip(".")
    raise ValueError(level.task)


# ---- error analysis (E0) ----------------------------------------------------------------

OUTCOMES = ("correct", "wrong_answer", "malformed", "unformatted", "limit", "no_answer")
_NUMBER = re.compile(r"-?\d+")


def outcome(level: TaskLevel, result: dict, correct: bool, reply_tokens: int, *,
            reasoning: bool = False) -> tuple[str, str | None]:
    """The outcome of one reply (proposal Sec. 3.6), and a detail for some outcomes.

    ``correct`` is the reply's score (:func:`score`), which the outcome does not change; the
    other outcomes describe the wrong replies: ``wrong_answer``, an answer is read and is
    wrong; ``malformed``, a ``\\boxed{}`` the parser cannot read; ``unformatted``, no
    ``\\boxed{}`` but the last line states a number (Operator Function); ``limit``, the
    output limit is reached before any answer; ``no_answer``, the reply ends before it
    without one. Details: Set Game's reason a set is wrong, whether an unformatted number
    is right, and a Tapatan label read from inside a code block. ``reply_tokens`` is the
    reply's length; direct-mode replies are stored stripped, so theirs may fall up to two
    tokens short of the limit.
    """
    raw = result["raw_response"]
    limit = REASONING_MAX_TOKENS if reasoning else MAX_TOKENS[level.task]
    at_limit = reply_tokens >= (limit if reasoning else limit - 2)
    if level.task == "tapatan":
        label = _TAPATAN_LABEL.search(raw) if reasoning else None
        read = bool(label) if reasoning else raw.strip().split("\n")[0].strip().rstrip(".").lower() in _TAPATAN_LABELS
        if not read:
            return ("limit" if at_limit else "no_answer"), None
        in_code = bool(label) and raw[:label.start()].count("```") % 2 == 1
        return ("correct" if correct else "wrong_answer"), "label in code" if in_code else None
    if correct:
        return "correct", None
    if level.task == "set_game":
        message = result["validation_message"]
        if message == "Model did not return exactly 3 cards":  # Fu's "Not boxed"
            return ("limit" if at_limit else "no_answer"), None
        if message == "Problem with raw response":
            return "malformed", None
        return "wrong_answer", "not a valid set" if "valid set" in message else "card not on the board"
    if level.task == "operator_func":
        if result["validation_message"] == "The answer is not correct":
            return "wrong_answer", None
        if "\\boxed" in raw:
            return "malformed", None
        if at_limit:
            return "limit", None
        lines = [line for line in raw.strip().splitlines() if line.strip()]
        numbers = _NUMBER.findall(lines[-1]) if lines else []
        if numbers:
            return "unformatted", "right" if numbers[-1] == str(result["expected_result"]).strip() else "wrong"
        return "no_answer", None
    if level.task == "noun_class_agreement":
        read = result["parse_method"] != "no_match"
    else:  # lexical_category_inference
        read = result["parse_method"] == "first_answer_token"
    if read:
        return "wrong_answer", None
    return ("limit" if at_limit else "no_answer"), None


def answer_complete(level: TaskLevel, text: str) -> bool:
    """Whether the answer the scoring reads is already fixed in ``text``.

    Scoring reads the first complete ``\\boxed{...}``, or for Tapatan the first label, so
    once it is in the text no continuation can change the score: reasoning-mode replies
    stop there. A label at the very end is not yet fixed (``continue`` may become
    ``continued``).
    """
    if level.task == "tapatan":
        return any(m.end() < len(text) for m in _TAPATAN_LABEL.finditer(text))
    return _BOXED.search(text) is not None


def answer_start(level: TaskLevel, text: str) -> int | None:
    """Where the extracted answer begins in a reasoning-mode reply (character offset).

    The answer is what the scoring reads: the first ``\\boxed{}``'s content (Fu et al.'s
    ``extract_outcome``), and within it the first card for Set Game; for Tapatan the first
    label. The pre-answer position is the token before the one holding this character.
    """
    if level.task == "tapatan":
        label = _TAPATAN_LABEL.search(text)
        return label.start(1) if label else None
    boxed = _BOXED.search(text)
    if not boxed:
        return None
    content = boxed.start(1)
    if level.task == "set_game":
        card = text.find("(", content, boxed.end(1))
        return card if card >= 0 else None
    return content + len(boxed.group(1)) - len(boxed.group(1).lstrip())


def recorded_results(directory: Path) -> list[dict]:
    """The per-item results of the result file Fu's runner wrote into ``directory``."""
    import json

    files = sorted(directory.rglob("*.json"))
    return json.loads(files[-1].read_text())["results"]


def recorded_prompts(directory: Path) -> dict[str, str]:
    """Item -> prompt text, from the result file Fu's runner wrote into ``directory``."""
    return {item_id(r): r["prompt"] for r in recorded_results(directory)}
