"""The LLM judge of the answer evaluation (proposal Sec. 5.5): GPT-5.5 through Azure.

The judge sees, for each case, a neutral question, the reference answer and the reply, not
the strict score. Cases go ``PER_CALL`` to a call; each verdict is cached by the hash of
the prompt version and the case, so identical replies (across wordings, or on a rerun) are
judged once. The cache is one JSON line per verdict in ``data/outputs/judge_cache.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from lth import azure
from lth.store import OUTPUTS

VERSION = "v2"  # v2: a reply that does not answer the question is no_answer
PER_CALL = 20
WORKERS = 8
VERDICTS = ("correct", "incorrect", "no_answer")
RELATIONS = ("identical", "case", "punctuation", "surface_variant", "valid_alternative", "different", "none")

PROMPT = """You are grading the replies of a language model. Each case gives the question the model had to answer (stated neutrally, not necessarily in the words the model saw), the reference answer, and the model's reply. Decide whether the reply's final answer is correct.

Rules:
- Judge only the final answer, not the reasoning or explanations around it.
- The answer may appear anywhere in the reply and in any form: with or without \\boxed{{}}, in quotes, inside a sentence, with other capitalization. The form does not matter, only the answer.
- If the question admits several correct answers (another synonym, antonym or translation of the same word; another valid set of cards on the board), an answer that differs from the reference but is also correct counts as correct.
- If the reply settles on a final answer, judge that one. If it offers several candidate answers without settling on one, it is correct only if every candidate is correct.
- If the reply gives no answer (it refuses, asks for more information, or stops before answering), the verdict is no_answer. The reply is the model's continuation of its prompt: if it does not answer the question but continues the text in some other way, and the reference occurs in it only in passing (say, inside a list of times or a new sentence), that is no answer either.
- The reply may go on after its answer with unrelated text, such as a new question; ignore that text.

For each case return an object with:
- "id": the case id;
- "verdict": "correct", "incorrect" or "no_answer";
- "answer": the reply's final answer, copied exactly as written in the reply (empty if none);
- "relation": how the answer relates to the reference, the first that applies: "identical" (the same apart from surrounding whitespace), "case" (differs only in capitalization), "punctuation" (differs only in punctuation, quotes or markup, possibly also in case), "surface_variant" (the same answer written differently: diacritics, inflection, number format, abbreviation, a paraphrased label), "valid_alternative" (a different answer that is also correct), "different" (a wrong answer), "none" (no answer);
- "several_answers": true if the reply offers more than one candidate final answer;
- "reference_suspect": true only if you are confident that the reference answer itself is wrong;
- "note": at most 15 words, only if something is unusual, else "".

Return only a JSON list with one object per case, in the order given.

{cases}"""

_lock = threading.Lock()


def case_key(question: str, reference: str, reply: str) -> str:
    text = json.dumps([VERSION, question, reference, reply], ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def load_cache() -> dict[str, dict]:
    path = OUTPUTS / "judge_cache.jsonl"
    if not path.exists():
        return {}
    cache = {}
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        cache[entry["key"]] = entry
    return cache


def _append(entries: list[dict]) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with _lock, (OUTPUTS / "judge_cache.jsonl").open("a") as f:
        f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)


def _render(cases: list[dict]) -> str:
    return "\n\n".join(f"### Case {i}\nQuestion:\n{c['question']}\n\nReference answer: {c['reference']}\n\n"
                       f"Reply:\n<<<\n{c['reply']}\n>>>" for i, c in enumerate(cases))


def _parse(content: str, n: int) -> list[dict]:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", content, re.S)
    value = json.loads(fenced.group(1) if fenced else content)
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"expected a list of {n} verdicts")
    for i, v in enumerate(value):
        if int(v["id"]) != i or v["verdict"] not in VERDICTS or v["relation"] not in RELATIONS:
            raise ValueError(f"malformed verdict {i}: {v}")
    return value


def _judge_call(client, url, cases: list[dict], attempts: int = 3) -> list[dict]:
    prompt = PROMPT.format(cases=_render(cases))
    for attempt in range(attempts):
        reply = azure.chat(client, url, prompt)
        try:
            verdicts = _parse(reply["content"], len(cases))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            if attempt == attempts - 1:
                raise
            continue
        return [{"key": c["key"], "version": VERSION, "model": reply["model"],
                 "response_id": reply["response_id"], **{k: v[k] for k in
                 ("verdict", "answer", "relation", "several_answers", "reference_suspect", "note")}}
                for c, v in zip(cases, verdicts)]


def judge(cases: list[dict], cache: dict[str, dict]) -> dict[str, dict]:
    """Verdicts for ``cases`` (dicts with question, reference, reply), keyed by case key.

    Cases already in ``cache`` are not sent again; new verdicts are added to it and to
    the cache file as each call returns.
    """
    for c in cases:
        c["key"] = case_key(c["question"], c["reference"], c["reply"])
    todo = list({c["key"]: c for c in cases if c["key"] not in cache}.values())
    if todo:
        client, url = azure.client_and_url()
        chunks = [todo[i:i + PER_CALL] for i in range(0, len(todo), PER_CALL)]

        def work(chunk):
            entries = _judge_call(client, url, chunk)
            _append(entries)
            return entries

        with ThreadPoolExecutor(WORKERS) as pool:
            for entries in pool.map(work, chunks):
                cache.update({e["key"]: e for e in entries})
    return {c["key"]: cache[c["key"]] for c in cases}
