"""E4b (paper/e4_heldout.tex): new instruction templates for the 14 simple tasks, from GPT-5.5.

For each task, six cells of five templates: three forms (cloze, question-answer, label) x
whether the instruction names the task. Each cell has one fixed prompt
(``materials/generation/e4b_wordings.txt``) with the task's description and its naming words
(``materials/generation/e4b_tasks.json``) and two of Yang et al.'s templates as examples (inst0,
the cloze one, and inst1, a question-answer one); no items. GPT-5.5 is asked for ``N_ASK``
templates per call; the first ``KEEP`` that pass the checks below are kept, in reply order;
calls are repeated (at most ``MAX_ROUNDS``) until the cell is full. Every reply is appended to
``data/generation/e4b_wordings.jsonl`` and reused on a rerun if its prompt is the cell's prompt.

Checks (the meaning and ambiguity check is the author's, by hand):
- the placeholder ``{input}`` exactly once, no other braces, no leading or trailing spaces;
- the form: cloze = one line, no colon; question-answer =
  an instruction of at least two words, then exactly `` Q: {input}\\nA:`` at the end;
  label = exactly two lines, ``<label>: {input}`` and ``<label>:``, labels of 1-4 words, not Q/A;
- naming, on the lowercased words of the template: named = at least one of the task's
  ``named`` words; not named = none of its ``forbidden`` words (words that name the answer
  relation: Yang et al.'s term list and the task's name for it; words for the input, such as
  "country" for capitals or "landmark", are allowed), also as a plural or as the start of a
  longer word; a line break written as backslash-n is read as a line break;
- different from Yang et al.'s six templates and from the cell's kept templates (spacing and
  case ignored), and from the kept templates of the task's other cells.

Writes ``data/generation/e4b_wordings_kept.json`` ({task: {cell: [templates]}}), the other
passing templates of the same replies as spares (``e4b_wordings_spares.json``), the
rejections with their reasons (``e4b_wordings_rejected.json``) and the review file
``data/review/e4b_wordings.md``. Approved templates go to ``materials/simple/e4b_wordings.json``.

    . ~/.azure_openai_key && .venv/bin/python scripts/generate_e4b_wordings.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, MATERIALS, azure, simple  # noqa: E402

N_ASK, KEEP, MAX_ROUNDS, WORKERS = 10, 5, 5, 8
OUT = DATA / "generation" / "e4b_wordings.jsonl"
KEPT = DATA / "generation" / "e4b_wordings_kept.json"
REJECTED = DATA / "generation" / "e4b_wordings_rejected.json"
SPARES = DATA / "generation" / "e4b_wordings_spares.json"
REVIEW = DATA / "review" / "e4b_wordings.md"
QA_TAIL = " Q: {input}\nA:"
FORMS = {
    "cloze": "a single sentence fragment, with no line break, that the answer completes, as in the first "
             "example. It must not contain a colon.",
    "qa": "an instruction or question, followed at the very end by exactly ' Q: {input}\\nA:' (a space, Q, a "
          "colon, a space, the placeholder, a line break, A, a colon), as in the second example. The text "
          "before ' Q:' must have at least two words.",
    "label": "exactly two lines and nothing else: a short label (one to four words) followed by ': {input}' on "
             "the first line, and a second short label followed by ':' on the second line, as in "
             "'Word: {input}\\nAnswer:'. No other instruction. Do not use 'Q' or 'A' as labels.",
}
NAMING = {
    "named": "the template must name the task: it must contain at least one of these words: {words}.",
    "unnamed": "the template must not name the task: it must not contain any of these words, in any form "
               "(singular or plural, any capitalisation, or as part of a longer word): {words}. Describe "
               "what is wanted in other words.",
}
CELLS = [(form, naming) for form in FORMS for naming in NAMING]
LABEL = re.compile(r"[A-Za-z][A-Za-z'\-/ ]*")
lock = threading.Lock()


def words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower())


def stem(w: str) -> str:
    for end, repl in (("ies", "y"), ("es", ""), ("s", "")):
        if w.endswith(end) and len(w) - len(end) >= 3:
            return w[: -len(end)] + repl
    return w


def contains(text: str, vocab: list[str]) -> list[str]:
    """The words of ``vocab`` the text contains (equal up to a plural, or starting a longer word)."""
    found = []
    for v in vocab:
        v = v.lower()
        if any(w == v or stem(w) == stem(v) or (len(v) >= 4 and w.startswith(v)) for w in words(text)):
            found.append(v)
    return found


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def problems(t: str, form: str, naming: str, task: dict, seen: set[str]) -> list[str]:
    found = []
    if t.count("{input}") != 1 or t.replace("{input}", "").count("{") or t.replace("{input}", "").count("}"):
        found.append("placeholder")
    if t != t.strip():
        found.append("leading or trailing space")
    if form == "cloze":
        if "\n" in t or ":" in t:
            found.append("not a cloze")
    elif form == "qa":
        head = t[: -len(QA_TAIL)] if t.endswith(QA_TAIL) else None
        if head is None or "\n" in head or "{input}" in head or len(words(head)) < 2:
            found.append("not question-answer")
    else:
        lines = t.split("\n")
        ok = len(lines) == 2 and lines[0].endswith(": {input}") and lines[1].endswith(":")
        labels = [lines[0][: -len(": {input}")], lines[1][:-1]] if ok else []
        if not ok or not all(LABEL.fullmatch(x) and 1 <= len(x.split()) <= 4 and x.strip() not in ("Q", "A")
                             for x in labels):
            found.append("not label")
    text = t.replace("{input}", " ")
    if naming == "named" and not contains(text, task["named"]):
        found.append("does not name the task")
    if naming == "unnamed" and (hit := contains(text, task["forbidden"])):
        found.append("names the task (" + ", ".join(hit) + ")")
    if norm(t) in seen:
        found.append("not new")
    return found


def prompt_for(template: str, task_name: str, task: dict, form: str, naming: str) -> str:
    inst = simple.instruction_dict()[task_name]
    vocab = task["named"] if naming == "named" else task["forbidden"]
    fill = {"<<DESCRIPTION>>": task["description"], "<<EXAMPLE1>>": json.dumps(inst["0"]),
            "<<EXAMPLE2>>": json.dumps(inst["1"]), "<<N>>": str(N_ASK), "<<FORM>>": "each template is " + FORMS[form],
            "<<NAMING>>": NAMING[naming].format(words=", ".join(f'"{v}"' for v in vocab)), "<<INPUT>>": "{input}"}
    for k, v in fill.items():
        template = template.replace(k, v)
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS))
    args = parser.parse_args()
    template = (MATERIALS / "generation" / "e4b_wordings.txt").read_text()
    tasks = json.loads((MATERIALS / "generation" / "e4b_tasks.json").read_text())
    replies = [json.loads(line) for line in OUT.read_text().splitlines()] if OUT.exists() else []
    OUT.parent.mkdir(parents=True, exist_ok=True)
    client, url = azure.client_and_url()
    yang = {t: {norm(v) for v in simple.instruction_dict()[t].values()} for t in simple.TASKS}
    kept: dict = {t: {} for t in args.tasks}
    rejected: dict = {t: {} for t in args.tasks}
    spares: dict = {t: {} for t in args.tasks}

    def fill(task_name: str, form: str, naming: str) -> None:
        cell = f"{form}-{naming}"
        prompt = prompt_for(template, task_name, tasks[task_name], form, naming)
        rounds = [r for r in replies if r["task"] == task_name and r["cell"] == cell and r["prompt"] == prompt]
        keep, spare, rej = [], [], []
        for i in range(MAX_ROUNDS):
            if len(keep) >= KEEP:
                break
            if i >= len(rounds):
                reply = {"task": task_name, "cell": cell, "round": i, "prompt": prompt,
                         **azure.chat(client, url, prompt)}
                with lock, OUT.open("a") as f:
                    f.write(json.dumps(reply, ensure_ascii=False) + "\n")
                rounds.append(reply)
            try:
                candidates = azure.json_list(rounds[i]["content"])
            except ValueError as error:
                rej.append(("<reply>", [f"unreadable reply: {error}"]))
                continue
            for t in candidates:
                t = t.replace("\\n", "\n")  # a line break written as backslash-n
                with lock:
                    seen = yang[task_name] | {norm(x) for c in kept[task_name].values() for x in c} | {norm(x) for x in keep}
                issues = problems(t, form, naming, tasks[task_name], seen)
                if issues:
                    rej.append((t, issues))
                elif len(keep) < KEEP:
                    keep.append(t)
                elif norm(t) not in {norm(x) for x in spare}:
                    spare.append(t)  # passes too: a replacement if a kept one fails the hand check
        with lock:
            kept[task_name][cell] = keep
            spares[task_name][cell] = spare
            rejected[task_name][cell] = rej
        print(f"{task_name:17s} {cell:14s} kept {len(keep)}/{KEEP} after {min(len(rounds), MAX_ROUNDS)} call(s), "
              f"rejected {len(rej)}", flush=True)

    with ThreadPoolExecutor(WORKERS) as pool:
        for f in [pool.submit(fill, t, form, naming) for t in args.tasks for form, naming in CELLS]:
            f.result()
    # Cross-cell duplicates can slip in when two cells finish at once; drop later copies.
    for t in args.tasks:
        seen = set()
        for cell in [f"{f}-{n}" for f, n in CELLS]:
            unique = [x for x in kept[t][cell] if norm(x) not in seen]
            seen |= {norm(x) for x in unique}
            if len(unique) < len(kept[t][cell]):
                print(f"{t} {cell}: {len(kept[t][cell]) - len(unique)} duplicate(s) of another cell dropped")
            kept[t][cell] = unique
    KEPT.write_text(json.dumps(kept, indent=1, ensure_ascii=False))
    REJECTED.write_text(json.dumps(rejected, indent=1, ensure_ascii=False))
    SPARES.write_text(json.dumps(spares, indent=1, ensure_ascii=False))
    review = ["# E4b instruction templates to check", "",
              "Mark each template OK or not: it must ask for exactly the task, be unambiguous, and lead the model",
              "to give the answer next. Form, placeholder and naming were checked by script.", "",
              "How to mark: `[x]` = OK. Leave `[ ]` or write `no` after a template to reject it. A rejected",
              "template is replaced by the first spare of its cell that you mark `[x]`.", ""]
    for t in args.tasks:
        review += [f"## {t}", "", f"*{tasks[t]['description']}*", ""]
        for cell in [f"{f}-{n}" for f, n in CELLS]:
            review += [f"### {cell} ({len(kept[t][cell])}/{KEEP})", ""]
            review += [f"- [ ] `{x!r}`" for x in kept[t][cell]] + [""]
            if spares[t][cell]:
                review += ["Spares (replace a rejected one, in this order):", ""]
                review += [f"- [ ] `{x!r}`" for x in spares[t][cell] if norm(x) not in
                           {norm(k) for c in kept[t].values() for k in c}] + [""]
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.write_text("\n".join(review))
    short = [(t, c, len(v)) for t in args.tasks for c, v in kept[t].items() if len(v) < KEEP]
    print("cells not filled:", short or "none")
    print(f"review file: {REVIEW}")


if __name__ == "__main__":
    main()
