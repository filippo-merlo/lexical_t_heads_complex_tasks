"""Answer evaluation (proposal Sec. 5.5): strict scores, judge inputs and disagreement reasons.

Works on the saved outputs of ``scripts/run_outputs.py``, without the model:
- the strict score is the source evaluation, unchanged: for the simple tasks the reply's
  first token against the first token of the reference (Yang et al.); for the complex
  tasks Fu's runner re-run on the stored replies, cut to Fu's limit (``strict``) and whole
  (``strict_full``, the study's strict score), the same for all five tasks; Tapatan, which
  has no scorer in Fu's release, by its first line;
- the judge's question is neutral: for a simple task its description and the input word,
  for a complex task the item's original rules prompt (the same under every variant);
- a disagreement between the two scores gets reasons, by rule where a rule can tell and
  from the judge's ``relation`` otherwise (:func:`reasons`).
"""

from __future__ import annotations

import re
import tempfile
from functools import lru_cache
from pathlib import Path

from lth import fu

SIMPLE_TASKS = {
    "antonym": "Give an antonym of the input word (a word with the opposite meaning).",
    "synonym": "Give a synonym of the input word (a word with the same meaning).",
    "singular-plural": "Give the plural form of the input noun.",
    "present-past": "Give the past tense of the input verb.",
    "country-capital": "Give the capital city of the input country.",
    "country-currency": "Give the currency of the input country.",
    "landmark-country": "Give the country where the input landmark is located.",
    "park-country": "Give the country where the input national park is located.",
    "product-company": "Give the company that makes the input product.",
    "next_item": "Give the item that comes right after the input in its sequence "
                 "(number, letter, Roman numeral, day, month, ...).",
    "prev_item": "Give the item that comes right before the input in its sequence "
                 "(number, letter, Roman numeral, day, month, ...).",
    "english-french": "Translate the input English word into French.",
    "english-german": "Translate the input English word into German.",
    "english-spanish": "Translate the input English word into Spanish.",
}

# Disagreement reasons (proposal Sec. 5.5), in the order a primary reason is chosen.
WRONG_BUT_RIGHT = ("truncation", "format_missing", "format_malformed", "first_vs_final", "case",
                   "punctuation", "leading_phrase", "tokenization", "surface_variant",
                   "valid_alternative", "reference_error", "other")
RIGHT_BUT_WRONG = ("first_token_false_positive", "parser_luck", "other")

_BOXED = re.compile(r"\\boxed\{(.+?)\}", re.S)
_TAPATAN_LABEL = re.compile(r"\b(A win|B win|continue)\b")
_EDGE = re.compile(r"^[\s\"'`*_.,:;!?()\[\]{}«»“”‘’$\\]+|[\s\"'`*_.,:;!?()\[\]{}«»“”‘’$\\]+$")


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer

    from lth import LENS_KEY
    from src.config import LENSES

    return AutoTokenizer.from_pretrained(LENSES[LENS_KEY].hf_model_id)


def decode(ids: list[int]) -> str:
    """As ``LM.decode``: Fu's vLLM path, no special tokens, stripped."""
    return tokenizer().decode(ids, skip_special_tokens=True).strip()


def bare(text: str) -> str:
    """Text without surrounding whitespace, quotes, punctuation or markup."""
    return _EDGE.sub("", text or "")


# ---- strict scores -----------------------------------------------------------------------

def strict_simple(row: dict) -> dict:
    first = tokenizer()(row["reference"], add_special_tokens=False).input_ids[0]
    return {"strict": row["reply_ids"][:1] == [first], "strict_full": row["reply_ids"][:1] == [first]}


def strict_complex(key: str, name: str, mode: str, rules: str | None, rows: list[dict]) -> dict[str, dict]:
    """Item -> strict scores, by Fu's runner over the stored replies (cut and whole)."""
    level = fu.TASK_LEVELS[key]
    out = {r["item"]: {} for r in rows}
    for cut in (True, False):
        texts = {r["prompt"]: decode(r["reply_ids"][:fu.MAX_TOKENS[level.task]] if cut else r["reply_ids"])
                 for r in rows}
        with tempfile.TemporaryDirectory() as scratch:
            results = fu.run(level, mode, lambda prompts: [texts[p] for p in prompts], Path(scratch),
                             rules=rules, n_test=len(rows))
        for result in results:
            correct, straight = fu.score(level, result)
            predicted = fu.answer(level, result)[1]
            suffix = "" if cut else "_full"
            out[fu.item_id(result)].update({f"strict{suffix}": correct, f"straight{suffix}": straight,
                                            f"parsed{suffix}": None if predicted is None else str(predicted)})
    return out


# ---- judge inputs ------------------------------------------------------------------------

def question_simple(task: str, row: dict) -> str:
    return f"Task: {SIMPLE_TASKS[task]}\nInput: {row['input']}"


def straight(family: str, row: dict, answer: str) -> bool:
    """The reply begins with the answer itself, not with a quote, an article or a preamble."""
    answer = bare(answer)
    return bool(answer) and reply_for_judge(family, row).lstrip().startswith(answer)


def reply_for_judge(family: str, row: dict) -> str:
    """Simple tasks: the reply up to its stop string (after it the model starts a new query)."""
    return row["answer_text"] if family == "simple" else row["reply_text"]


# ---- disagreement reasons ----------------------------------------------------------------

def primary_strict(task: str, scores: dict) -> bool:
    """The strict score the study uses for a complex task: the whole reply, for every task."""
    return scores["strict_full"]


def reasons(family: str, task: str, row: dict, scores: dict, verdict: dict) -> list[str]:
    """Why the strict score and the judge disagree, most specific first ([] if they agree).

    ``scores`` holds the strict scores, ``verdict`` the judge's answer for this reply.
    """
    judged = verdict["verdict"] == "correct"
    if primary_strict(task, scores) == judged:
        return []
    reply = reply_for_judge(family, row)
    answer, reference = verdict.get("answer") or "", str(row["reference"])
    relation = verdict.get("relation")
    found = []
    if judged:  # strict wrong, judge right
        if family == "complex":
            if task in ("set_game", "operator_func"):
                boxes = _BOXED.findall(row["reply_text"])
                if not boxes:
                    found.append("format_missing")
                elif bare(scores.get("parsed_full") or "") != bare(answer) and len(boxes) > 1:
                    found.append("first_vs_final")
                else:
                    found.append("format_malformed")
            if task == "tapatan":
                labels = _TAPATAN_LABEL.findall(row["reply_text"])
                first_line = row["reply_text"].strip().split("\n")[0].strip().rstrip(".").lower()
                if first_line not in ("a win", "b win", "continue"):
                    found.append("format_missing")
                if labels and labels[0] != labels[-1]:
                    found.append("first_vs_final")
        if relation == "case":
            found.append("case")
        if relation == "punctuation":
            found.append("punctuation")
        # Complex tasks let the model reason first, so text before the answer is no reason.
        if answer and family == "simple" and not bare(reply.lstrip()).startswith(bare(answer)):
            found.append("leading_phrase")
        if family == "simple" and relation == "identical" and reply.lstrip().startswith(answer):
            found.append("tokenization")
        if relation in ("surface_variant", "valid_alternative"):
            found.append(relation)
        if verdict.get("reference_suspect"):
            found.append("reference_error")
        order = WRONG_BUT_RIGHT
    else:  # strict right, judge wrong
        if family == "simple":
            found.append("first_token_false_positive")
        elif bare(scores.get("parsed_full") or scores.get("parsed") or "") != bare(answer):
            found.append("parser_luck")
        order = RIGHT_BUT_WRONG
    found = [r for r in order if r in found] or ["other"]
    return found
