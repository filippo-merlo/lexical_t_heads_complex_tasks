"""Answer evaluation (proposal Sec. 5.5): strict scores, judge inputs and disagreement reasons.

Works on the saved outputs of ``scripts/run_outputs.py``, without the model:
- the strict score is the source evaluation, unchanged: for the simple tasks the reply's
  first token against the first token of the reference (Yang et al.), and a second, more
  flexible score, the whole reference as words at the reply's start (:func:`flexible_simple`); for the complex
  tasks Fu's runner re-run on the stored replies, cut to Fu's limit (``strict``) and whole
  (``strict_full``, the study's strict score), the same for all five tasks; Tapatan, which
  has no scorer in Fu's release, by the first label anywhere in the reply, as in E0;
- the judge's question is neutral: for a simple task its description and the input word,
  for a complex task the item's original rules prompt (the same under every variant);
- a disagreement between the two scores gets reasons, in three groups (where the answer is,
  how it is written, which answer is right), by rule where a rule can tell and from the
  judge's ``relation`` otherwise (:func:`reasons`).
"""

from __future__ import annotations

import re
import tempfile
import unicodedata
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
WRONG_BUT_RIGHT = ("no_box", "leading_text", "earlier_answer", "case", "extra_characters", "tokenization",
                   "surface_variant", "valid_alternative", "reference_error", "other")
RIGHT_BUT_WRONG = ("parser_luck", "first_token_false_positive", "other")
GROUPS = {"where": ("no_box", "leading_text", "earlier_answer", "parser_luck"),
          "written": ("case", "extra_characters", "tokenization", "surface_variant",
                      "first_token_false_positive"),
          "which": ("valid_alternative", "reference_error"),
          "other": ("other",)}
GROUP = {reason: group for group, reasons in GROUPS.items() for reason in reasons}

_BOXED = re.compile(r"\\boxed\{(.+?)\}", re.S)
_TAPATAN_LABEL = re.compile(r"\b(A win|B win|continue)\b")
_EDGE = re.compile(r"^[\s\"'`*_.,:;!?()\[\]{}«»“”‘’$\\]+|[\s\"'`*_.,:;!?()\[\]{}«»“”‘’$\\]+$")
_LEADING = re.compile(r"^\W+")
_ARTICLE = re.compile(r"^(?:(?:a|an|the|der|die|das|den|dem|des|ein|eine|einen|einem|einer|le|la|les|un|une|"
                      r"el|los|las|uno|una|to)\s+|l['’ʼ]\s*)")
_PARENTHESIS = re.compile(r"\([^)]*\)")
_POSSESSIVE = re.compile(r"['’ʼ]s\b")
_JOINER = re.compile(r"(?<=\w)[-‐'’ʼ](?=\w)")


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


def normalise(text: str) -> str:
    """Text in lower case, without a leading article or "to", accents or punctuation.

    A hyphen or apostrophe inside a word joins it (``non-corresponding``), a possessive
    ``'s`` is dropped, and any other punctuation separates words (``petit(e)``, ``Peru/Ecuador``).
    """
    text = _ARTICLE.sub("", _LEADING.sub("", text.lower()))
    text = _JOINER.sub("", _POSSESSIVE.sub("", text))
    text = "".join(" " if unicodedata.category(c)[0] == "P" else c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c)[0] != "M" and unicodedata.category(c) != "Lm")
    return " ".join(text.split())


def flexible_simple(row: dict) -> bool:
    """The second simple-task score: the reply begins with the whole reference, as words.

    Case, accents, punctuation (quotes, hyphens, apostrophes) and a leading article or
    "to" are ignored; a parenthesis in the reference is optional (``Euro (EUR)``).
    """
    reply = normalise(row["answer_text"])
    references = {normalise(row["reference"]), normalise(_PARENTHESIS.sub(" ", row["reference"]))} - {""}
    return any(reply == ref or reply.startswith(ref + " ") for ref in references)


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
            # reasoning=True: Tapatan by the first label anywhere (the other tasks ignore it)
            correct, straight = fu.score(level, result, reasoning=True)
            predicted = fu.answer(level, result, reasoning=True)[1]
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


def squash(text: str) -> str:
    """Text in lower case with whitespace collapsed, for finding one answer inside another."""
    return " ".join((text or "").lower().split())


def reasons(family: str, task: str, row: dict, scores: dict, verdict: dict) -> list[str]:
    """Why the strict score and the judge disagree, primary reason first ([] if they agree).

    ``scores`` holds the strict scores, ``verdict`` the judge's answer for this reply, which
    the judge copies as written in the reply. :data:`GROUP` gives each reason's group.
    """
    judged = verdict["verdict"] == "correct"
    if primary_strict(task, scores) == judged:
        return []
    reply = reply_for_judge(family, row)
    answer, reference = (verdict.get("answer") or "").strip(), str(row["reference"]).strip()
    parsed = scores.get("parsed_full") or ""
    relation = verdict.get("relation")
    found = []
    if judged:  # strict wrong, judge right
        # Where the answer is.
        if task in ("set_game", "operator_func"):
            # The parser reads the first box; another card notation is left to the judge.
            box = re.search(_BOXED, reply)
            if not box:
                found.append("no_box")
            elif task == "operator_func" and squash(answer) and squash(answer) in squash(box.group(1)):
                found.append("extra_characters")
            elif squash(answer) and squash(answer) in squash(reply[box.end():]):
                found.append("earlier_answer")
        elif task == "tapatan":
            # No label at all is a paraphrased label, a surface variant (from the judge).
            labels = _TAPATAN_LABEL.findall(reply)
            if labels and labels[0] != labels[-1]:
                found.append("earlier_answer")
        elif task in ("noun_class_agreement", "lexical_category_inference"):
            # Fu's parsers read Yes or No: a different one read is an earlier answer; none
            # read, where Lexical Category Inference needs it as the first word, is leading text.
            if parsed and squash(bare(parsed)) != squash(bare(answer)):
                found.append("earlier_answer")
            elif not parsed and task == "lexical_category_inference" and squash(answer) in squash(reply):
                found.append("leading_text")
        elif answer and not bare(reply.lstrip()).lower().startswith(bare(answer).lower()):
            found.append("leading_text")  # simple tasks: the first token is read
        # How the answer is written.
        if bare(answer).lower() == bare(reference).lower() and bare(answer) != bare(reference):
            found.append("case")
        if family == "simple" and answer and (bare(answer) != answer or (
                not reply.lstrip().startswith(answer) and bare(reply.lstrip()).startswith(bare(answer)))):
            found.append("extra_characters")  # simple tasks: quotes or markup before the first token
        if family == "simple" and relation == "identical" and reply.lstrip().startswith(answer):
            found.append("tokenization")
        if relation == "surface_variant" or (relation == "punctuation" and "extra_characters" not in found):
            found.append("surface_variant")  # punctuation inside the answer (U.S. for US)
        # Which answer is right.
        if relation == "valid_alternative":
            found.append("valid_alternative")
        if verdict.get("reference_suspect"):
            found.append("reference_error")
        order = WRONG_BUT_RIGHT
    else:  # strict right, judge wrong
        if family == "simple":
            found.append("first_token_false_positive")
        elif squash(bare(parsed or scores.get("parsed") or "")) != squash(bare(answer)):
            found.append("parser_luck")
        order = RIGHT_BUT_WRONG
    return [r for r in order if r in found] or ["other"]
