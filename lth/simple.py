"""The 14 simple tasks of Yang et al. (2026), with their datasets, prompts and control pairs.

Prompt builders are copied from their ``src/Shared_utils/prompt_utils.py`` (f81ef39):
example-based prompts are ``" in: out;"`` shots ending in ``" query:"`` (the shots are the
items before the query in file order, wrapping round for the first ones), rule-based
prompts are their instruction templates 0-4, and the prompt without task information is
``" query:"``. All are raw text, scored by the greedy first token against the first token
of ``" " + output``.

Prompts run in their batches of 20 (:data:`BATCH`), padded on the left. Their code passes
no attention mask, so the model attends to the padding (nnsight fills a mask of ones); we
pass the mask and matching positions (``LM.next_tokens_batch``), so that every prompt is
computed as if run alone. Which wordings each step uses is theirs (``behavior_variance.py``,
``identify_heads_nnsight.py``, ``causal_fix_incorrect.py``):
- heads are identified on the first 100 correct prompts of one wording per style: the
  5-shot prompts, and the instruction template with the highest accuracy;
- each source wording of a style gives its own mean and its own patch;
- the patched items are the first 100 answered correctly 1-shot.
"""

from __future__ import annotations

import json

from lth import YANG

TASKS = ("antonym", "synonym", "singular-plural", "present-past", "country-capital",
         "country-currency", "landmark-country", "park-country", "product-company",
         "next_item", "prev_item", "english-french", "english-german", "english-spanish")
SHOTS = (1, 5, 10, 20, 30)
# instruction_dict.json has six templates per task; Yang et al.'s code and paper use the
# first five (App. A.3.1, templates A-E).
INSTRUCTIONS = tuple(range(5))
WORDINGS = {"rule": tuple(f"inst{i}" for i in INSTRUCTIONS),
            "example": tuple(f"shot{n}" for n in SHOTS)}
ZERO = "zero"
FIXING = "shot1"  # the patched items are the first ones answered correctly 1-shot
IDENTIFY_EXAMPLE = "shot5"  # identify_heads_nnsight.py: prompt_template_index = 5 for EP
CAP = 100  # exp_size and average_size
BATCH = 20  # their default batch_size
# The control mean comes from the control task's 1-shot prompts, at the items that task
# answers correctly under the mean style's wording with index 1 (causal_recover_corrupted_
# prompts.py: tokens from EP 1-shot, items from ``prompt_type_fix`` index 1), all of them.
CONTROL_PROMPTS = "shot1"
CONTROL_ITEMS = {"rule": "inst1", "example": "shot1"}
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


def token_ids(tokenizer, prompts_: list[str]) -> list[list[int]]:
    """Every prompt with BOS, unpadded (``LM.next_tokens_batch`` pads each batch with a mask)."""
    return [tokenizer(p).input_ids for p in prompts_]


def correct_items(records: list[dict]) -> list[int]:
    """Items answered correctly, in file order (their ``correct_index``)."""
    return [int(r["item"]) for r in records if r["correct"]]


def identification_wording(style: str, accuracy: dict[str, float]) -> str:
    """The wording whose prompts identify a style's heads (first maximum, as ``np.argmax``)."""
    if style == "example":
        return IDENTIFY_EXAMPLE
    return max(WORDINGS["rule"], key=lambda w: (accuracy[w], -WORDINGS["rule"].index(w)))


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
