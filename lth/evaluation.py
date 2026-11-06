"""Answer evaluation (proposal Sec. 5.5): strict and flexible scores.

Works on the saved outputs of ``scripts/run_outputs.py``, without the model:
- the strict score is the source evaluation, unchanged: for the simple tasks the reply's
  first token against the first token of the reference (Yang et al.), and a second, more
  flexible score, the whole reference as words at the reply's start (:func:`flexible_simple`); for the complex
  tasks Fu's runner re-run on the stored replies, cut to Fu's limit (``strict``) and whole
  (``strict_full``, the study's strict score), the same for all five tasks; Tapatan, which
  has no scorer in Fu's release, by the first label anywhere in the reply, as in E0; and a
  flexible score, the same answer in any notation (:func:`flexible_complex`).
"""

from __future__ import annotations

import importlib
import re
import tempfile
import unicodedata
from functools import lru_cache
from pathlib import Path

from lth import fu

_TAPATAN_LABEL = re.compile(r"\b(A win|B win|continue)\b")
_LEADING = re.compile(r"^\W+")
_ARTICLE = re.compile(r"^(?:(?:a|an|the|der|die|das|den|dem|des|ein|eine|einen|einem|einer|le|la|les|un|une|"
                      r"el|los|las|uno|una|to)\s+|l['’ʼ]\s*)")
_PARENTHESIS = re.compile(r"\([^)]*\)")
_POSSESSIVE = re.compile(r"['’ʼ]s\b")
_JOINER = re.compile(r"(?<=\w)[-‐'’ʼ](?=\w)")
_BOX_START = re.compile(r"\\boxed\{")
_CARD = re.compile(r"\(?\s*([A-Za-z]+)\s*\|\s*([A-Za-z]+)\s*\)?")
_INTEGER = re.compile(r"-?\d+")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_CONCLUSION = re.compile(r"answer(?: is)?\s*[:=]?\s*\**\s*(yes|no)\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer

    from lth import LENS_KEY
    from src.config import LENSES

    return AutoTokenizer.from_pretrained(LENSES[LENS_KEY].hf_model_id)


def decode(ids: list[int]) -> str:
    """As ``LM.decode``: Fu's vLLM path, no special tokens, stripped."""
    return tokenizer().decode(ids, skip_special_tokens=True).strip()


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


def _boxes(text: str) -> list[str]:
    """Each ``\\boxed{`` 's content, up to its first closing brace or the next box (a
    missing or doubled closing brace or bracket is tolerated)."""
    starts = [m.end() for m in _BOX_START.finditer(text)]
    out = []
    for i, start in enumerate(starts):
        segment = text[start:starts[i + 1] if i + 1 < len(starts) else len(text)]
        close = segment.find("}")
        out.append(segment[:close] if close >= 0 else segment)
    return out


def _number(text: str) -> str | None:
    """The last integer written in ``text`` (Unicode minus and thousands commas allowed)."""
    found = _INTEGER.findall(_THOUSANDS.sub("", text.replace("−", "-")))
    return found[-1] if found else None


def flexible_answer(task: str, reply: str) -> str | None:
    """The answer the strict score looks for, written in Fu's format (None if there is none).

    Operator Function: the number in the first box, or without a box the number on the
    reply's last line; Set Game: the first box holding three readable cards, brackets
    optional; Tapatan: the first label, as the strict score (its labels are written exactly,
    and paraphrases such as "A wins" also occur inside the reasoning); Noun Class Agreement
    and Lexical Category Inference: Fu's Noun Class Agreement reader (a line that is only Yes
    or No, else "Answer: Yes") without its last resort (the first yes or no anywhere), plus
    a closing "the answer is Yes".
    """
    if task == "operator_func":
        numbers = [n for n in map(_number, _boxes(reply)) if n is not None]
        if not numbers:
            lines = [line for line in reply.strip().splitlines() if line.strip()]
            numbers = [n for n in [_number(lines[-1]) if lines else None] if n is not None]
        return f"\\boxed{{{numbers[0]}}}" if numbers else None
    if task == "set_game":
        for content in _boxes(reply):
            cards = _CARD.findall(content)
            if len(cards) == 3:
                return "\\boxed{" + "".join(f"({a.lower()}|{b.lower()})" for a, b in cards) + "}"
        return None
    if task == "tapatan":
        label = _TAPATAN_LABEL.search(reply)
        return label.group(1) if label else None
    reader = importlib.import_module("tasks.noun_class_agreement")
    for line in reply.strip().splitlines():
        if match := reader._STRICT_LINE_RE.match(line):
            return match.group(1).capitalize()
    if match := reader._ANSWER_PREFIX_RE.search(reply):
        return match.group(1).capitalize()
    found = _CONCLUSION.findall(reply)
    return found[-1].capitalize() if found else None


def flexible_complex(key: str, name: str, mode: str, rules: str | None, rows: list[dict]) -> dict[str, dict]:
    """Item -> the second complex-task score: :func:`flexible_answer` scored by Fu's runner
    and the strict scorer, as if it were the whole reply."""
    level = fu.TASK_LEVELS[key]
    answers = {r["prompt"]: flexible_answer(level.task, r["reply_text"]) for r in rows}
    with tempfile.TemporaryDirectory() as scratch:
        results = fu.run(level, mode, lambda prompts: [answers[p] or "" for p in prompts], Path(scratch),
                         rules=rules, n_test=len(rows))
    return {fu.item_id(result): {"flexible": fu.score(level, result, reasoning=True)[0],
                                 "flexible_answer": answers[result["prompt"]]} for result in results}
