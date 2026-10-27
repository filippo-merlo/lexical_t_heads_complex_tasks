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

def run(level: TaskLevel, mode: str, generate: Callable[[str], str], output_dir: Path, *,
        rules: str | None = None, n_test: int | None = None) -> list[dict]:
    """Fu's runner for one task level and mode, answering each prompt with ``generate``.

    ``rules`` replaces the rules-only template (rules mode only). Returns the runner's
    per-item records, each with the ``prompt`` and the ``raw_response``.
    """
    if rules is not None and mode != "rules":
        raise ValueError("a rule text applies to rules mode only")
    module = importlib.import_module(f"tasks.{level.task}")
    answered: dict[str, str] = {}

    async def query_batch(prompts):
        responses = []
        for prompt in prompts:
            answered[prompt] = generate(prompt)
            responses.append(answered[prompt])
        return responses

    kwargs = dict(level.runner_kwargs, mode=mode, num_test_samples=n_test or level.n_test,
                  num_demos=DEMOS.get(level.task), output_dir=str(output_dir),
                  model_name=MODEL_NAME, seed=SEED, max_tokens=MAX_TOKENS[level.task],
                  query_batch_fn=query_batch, query_batch_size=1)
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


def item_id(result: dict) -> str:
    return str(result.get("id", result.get("sample_id")))


def score(level: TaskLevel, result: dict) -> tuple[bool, bool]:
    """``(correct, straight)``: the answer is right; the answer is the first thing written.

    Set Game, Operator Function and Lexical Category Inference use the runners' own
    ``is_correct``; Noun Class Agreement compares the runner's parsed answer with the
    label; Tapatan's release has no check, so its first line must be the expected label.
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
        return first == str(result["expected_result"]).lower(), first in _TAPATAN_LABELS
    raise ValueError(level.task)
