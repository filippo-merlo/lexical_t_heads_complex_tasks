"""Term lists and their token ids, as in Yang et al. (2026).

``expand_to_token_ids`` is ``convert_expand_candidate_strs_to_token_ids_nnsight``
(``src/Submodules/identify_heads_early_decode.py`` at f81ef39), with the nnsight model
replaced by its tokenizer: every word is tried as written, with a leading space, in upper,
capitalised and lower case, and after a ``.``, ``_`` or ``-``; each variant that is one
token is kept.
"""

from __future__ import annotations

import json

from lth import MATERIALS, YANG


def expand_to_token_ids(tokenizer, candidate_strs: list[str]) -> list[int]:
    unique_tokens = set()

    def process(text):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            unique_tokens.add((ids[0], tokenizer.decode([ids[0]])))

    for string in candidate_strs:
        process(string)
        process(" " + string)
        process(string.upper())
        process(string.capitalize())
        process(string.lower())
        process("." + string)
        process("_" + string)
        process("-" + string)

    return [token_id for token_id, _ in sorted(unique_tokens)]


def simple_term_lists() -> dict[str, list[str]]:
    """Yang et al.'s human-written lists (``task_relation_dict.json``)."""
    return json.loads((YANG / "datasets" / "dataset_info" / "task_relation_dict.json").read_text())


def complex_term_lists() -> dict[str, list[str]]:
    """Our checked lists, one per original rules prompt of Fu et al. (proposal App. D)."""
    return json.loads((MATERIALS / "complex" / "terms.json").read_text())
