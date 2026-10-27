"""Rule wordings for the complex tasks, from GPT-5.5 (proposal Sec. 4.3 and App. B).

Each of the 13 original rules prompts of Fu et al. goes to GPT-5.5 with the App. B
paraphrase prompt (``materials/generation/rule_wordings.txt``), asking for six wordings.
A wording is kept only if it passes the format checks below; rounds are repeated until six
pass. Every reply is appended to ``data/generation/wordings.jsonl``, so a rerun reuses
them. The kept wordings go to ``data/review/wordings.md`` for the hand check; the approved
ones are then written to ``materials/complex/wordings.json``.

Format checks (the meaning check is the hand check):
- the same placeholders, each as often as in the original;
- the input block (the lines that introduce the input and the answer) unchanged at the end;
- the answer-format instruction unchanged, where it is a rule of its own;
- different from the original and from the other kept wordings.

    . ~/.azure_openai_key && .venv/bin/python scripts/generate_wordings.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, MATERIALS, azure, fu  # noqa: E402

N = 6
MAX_ROUNDS = 5
OUT = DATA / "generation" / "wordings.jsonl"
REVIEW = DATA / "review" / "wordings.md"
# Answer-format instructions that are rules of their own (the others sit in the input block).
FORMAT_RULES = {
    "noun_class_agreement": "Your answer must be exactly one word: Yes or No.",
    "lexical_category_inference": "Your answer must be exactly one word: Yes or No.",
    "tapatan": "You must output a label that is exactly one of: A win, B win, or continue.",
}


def placeholders(text: str) -> Counter:
    return Counter(re.findall(r"\{([A-Za-z_]+)\}", text))


def problems(wording: str, original_text: str, block: str, task: str, kept: list[str]) -> list[str]:
    found = []
    if placeholders(wording) != placeholders(original_text):
        found.append("placeholders differ")
    if not wording.rstrip().endswith(block.rstrip()):
        found.append("input block changed")
    if task in FORMAT_RULES and FORMAT_RULES[task] not in wording:
        found.append("answer-format instruction changed")
    normal = " ".join(wording.split())
    if normal == " ".join(original_text.split()) or normal in {" ".join(k.split()) for k in kept}:
        found.append("not new")
    return found


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()

    instruction = (MATERIALS / "generation" / "rule_wordings.txt").read_text()
    replies = [json.loads(line) for line in OUT.read_text().splitlines()] if OUT.exists() else []
    OUT.parent.mkdir(parents=True, exist_ok=True)
    client, url = azure.client_and_url()

    kept_by_key, review = {}, ["# Rule wordings to check", "",
                               "Each wording must state exactly the rules of the original. "
                               "Placeholders, input block and answer format were checked automatically.", ""]
    for key, original in fu.ORIGINALS.items():
        text = fu.display(fu.original_template(original))
        block = fu.display(fu.input_block(fu.original_template(original), original))
        task = key.split("/")[0]
        prompt = instruction.replace("{N}", str(N)).replace("{rule_prompt}", text)
        kept, rejected = [], []
        rounds = [r for r in replies if r["key"] == key]
        for round_index in range(MAX_ROUNDS):
            if len(kept) >= N:
                break
            if round_index >= len(rounds):
                reply = {"key": key, "round": round_index, **azure.chat(client, url, prompt)}
                with OUT.open("a") as f:
                    f.write(json.dumps(reply, ensure_ascii=False) + "\n")
                rounds.append(reply)
            for wording in azure.json_list(rounds[round_index]["content"]):
                issues = problems(wording, text, block, task, kept)
                if issues or len(kept) >= N:
                    rejected.append((wording, issues or ["surplus"]))
                else:
                    template = fu.template_from_display(wording, original.placeholders)
                    template.format(**{p: "x" for p in original.placeholders})  # no stray braces
                    kept.append(wording)
        kept_by_key[key] = kept
        print(f"{key}: kept {len(kept)}, rejected {sum(1 for _, i in rejected if i != ['surplus'])}")
        review += [f"## {key}", "", "### Original", "", "```", text, "```", ""]
        for i, wording in enumerate(kept, 1):
            review += [f"### Wording {i}", "", "```", wording, "```", ""]
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.write_text("\n".join(review))
    (DATA / "generation" / "wordings_kept.json").write_text(json.dumps(kept_by_key, indent=1, ensure_ascii=False))
    print(f"review file: {REVIEW}")


if __name__ == "__main__":
    main()
