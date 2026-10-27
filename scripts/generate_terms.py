"""Term lists for the complex tasks, from GPT-5.5 (proposal Sec. 4.3 and App. B).

Each of the 13 original rules prompts of Fu et al. goes to GPT-5.5 with the App. B
term-list prompt (``materials/generation/term_lists.txt``). Replies are appended to
``data/generation/terms.jsonl`` and reused on a rerun. The lists go to
``data/review/terms.md`` for the hand check, with each word's single-token variants under
Yang et al.'s expansion (``lth.terms``); the approved lists are then written to
``materials/complex/terms.json``.

    . ~/.azure_openai_key && .venv/bin/python scripts/generate_terms.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, MATERIALS, azure, fu  # noqa: E402
from lth.terms import expand_to_token_ids  # noqa: E402

OUT = DATA / "generation" / "terms.jsonl"
REVIEW = DATA / "review" / "terms.md"
# Output labels of the tasks; a term list must not contain them (App. B, requirement 2).
LABELS = {"yes", "no", "a win", "b win", "continue"}


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    from transformers import AutoTokenizer

    from src.config import LENSES

    tokenizer = AutoTokenizer.from_pretrained(LENSES["llama3.1-8b-it"].hf_model_id)
    instruction = (MATERIALS / "generation" / "term_lists.txt").read_text()
    replies = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            reply = json.loads(line)
            replies[reply["key"]] = reply
    OUT.parent.mkdir(parents=True, exist_ok=True)
    client, url = azure.client_and_url()

    lists, review = {}, ["# Term lists to check", "",
                         "Words that describe the task itself, not its answers or the items of one input. "
                         "`tokens` are the single-token variants under Yang et al.'s expansion; "
                         "a word with none can never match.", ""]
    for key, original in fu.ORIGINALS.items():
        if key not in replies:
            text = fu.display(fu.original_template(original))
            prompt = instruction.replace("{rule_prompt}", text)
            replies[key] = {"key": key, **azure.chat(client, url, prompt)}
            with OUT.open("a") as f:
                f.write(json.dumps(replies[key], ensure_ascii=False) + "\n")
        words = list(dict.fromkeys(azure.json_list(replies[key]["content"])))
        lists[key] = words
        review += [f"## {key}", "", "| word | tokens | flag |", "|---|---|---|"]
        for word in words:
            tokens = [tokenizer.decode([t]) for t in expand_to_token_ids(tokenizer, [word])]
            flag = "label" if word.lower() in LABELS else ("several words" if " " in word.strip() else "")
            review.append(f"| {word} | {', '.join(repr(t) for t in tokens) or '—'} | {flag} |")
        review.append("")
        print(f"{key}: {len(words)} words")
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.write_text("\n".join(review))
    (DATA / "generation" / "terms_lists.json").write_text(json.dumps(lists, indent=1, ensure_ascii=False))
    print(f"review file: {REVIEW}")


if __name__ == "__main__":
    main()
