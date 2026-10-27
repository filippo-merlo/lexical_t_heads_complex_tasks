"""The 14 simple tasks of Yang et al. (2026), with their datasets, prompts and control pairs.

Prompt builders are copied from their ``src/Shared_utils/prompt_utils.py`` (f81ef39):
example-based prompts are ``" in: out;"`` shots ending in ``" query:"`` (the shots are the
items before the query in file order, wrapping round for the first ones), rule-based
prompts are their six instruction templates, and the prompt without task information is
``" query:"``. All are raw text, scored by the greedy first token against the first token
of ``" " + output``.
"""

from __future__ import annotations

import json

from lth import YANG

TASKS = ("antonym", "synonym", "singular-plural", "present-past", "country-capital",
         "country-currency", "landmark-country", "park-country", "product-company",
         "next_item", "prev_item", "english-french", "english-german", "english-spanish")
SHOTS = (1, 5, 10, 20, 30)
INSTRUCTIONS = tuple(range(6))
WORDINGS = {"rule": tuple(f"inst{i}" for i in INSTRUCTIONS),
            "example": tuple(f"shot{n}" for n in SHOTS)}
ZERO = "zero"
FIXING = "shot1"  # the patched items are the first ones answered correctly 1-shot
# Yang et al.'s control_map (src/causal_fix_incorrect.py), restricted to these tasks.
CONTROL = {"country-capital": "park-country", "product-company": "park-country",
           "park-country": "country-currency", "landmark-country": "product-company",
           "country-currency": "park-country", "antonym": "synonym", "synonym": "antonym",
           "singular-plural": "present-past", "present-past": "singular-plural",
           "next_item": "prev_item", "prev_item": "next_item",
           "english-french": "english-german", "english-german": "english-french",
           "english-spanish": "english-german"}


def dataset(task: str) -> list[dict]:
    return json.loads((YANG / "datasets" / "abstractive" / f"{task}.json").read_text())


def instruction_dict() -> dict:
    return json.loads((YANG / "datasets" / "dataset_info" / "instruction_dict.json").read_text())


def prompts(task: str, wording: str) -> tuple[list[str], list[str]]:
    """``(prompts, answers)`` for every item of ``task``, in file order."""
    data = dataset(task)
    if wording == ZERO:
        return create_zs_prompts(data, return_answers=True)
    if wording.startswith("shot"):
        prompts_, answers, _ = create_few_shot_prompts(
            data, n_shot=int(wording[4:]), delimiter=";", q_bos=" ", a_bos=" ", qa_delimiter=":")
        return prompts_, answers
    if wording.startswith("inst"):
        return create_instruction_prompts(data, instruction_dict()[task][wording[4:]])
    raise ValueError(wording)


# ---- copied from Yang et al., src/Shared_utils/prompt_utils.py ---------------------------

def create_few_shot_prompts(data, n_shot, delimiter, q_bos, a_bos, qa_delimiter):

    prompts = []
    answers = []
    query_index = []

    # For each possible last example
   #for i in range(n_shot, len(data)):
    for i in range(0, len(data)):
        context = ""
        for j in range(i - n_shot, i):
            example = data[j]
            context += f"{q_bos}{example['input']}{qa_delimiter}{a_bos}{example['output']}{delimiter}"

        query = data[i]
        context += f"{q_bos}{query['input']}{qa_delimiter}"

        prompts.append(context)
        answers.append(a_bos + query["output"])
        query_index.append(i)

    return prompts, answers, query_index


def create_instruction_prompts(dataset, instruction):
    assert "{input}" in instruction

    prompts, answers = [], []
    for data in dataset:
        prompts.append(instruction.format(input=data["input"]))
        answers.append(" " + data["output"])
    return prompts, answers


def create_zs_prompts(dataset, return_answers=False):
    prompts, answers = [], []
    for data in dataset:
        prompts.append(" " + data["input"] +":")
        answers.append(" " + data["output"])
    if return_answers:
        return prompts, answers
    else:
        return prompts
