"""E5 (paper/proposal-llm-heads.tex): the head tables and the prompts for the LLM (no GPU).

For each task, choosing half (A for fold 1, B for fold 2) and readout, from the top-20 readings
of ``scripts/heldout_readings.py --k 20`` (``data/readings/heldout_top20``), pooled over the five
rule templates (the first 100 correct prompts of each, as E4's P+):

- every head's answer share (share of prompts whose own answer's first token, with its leading
  space, is among the head's 20 tokens), input share (the same with the item's input) and its
  20 most frequent tokens with their share of prompts (ties by token id);
- anonymous ids ``H0000``-``H1023``: id j is head ``PERM[j]`` (``l * 32 + h``), ``PERM`` a
  random permutation (seed 0), the same for every task, half and readout; rows in id order;
- the 8 batches of 128 ids, in id order;
- the prompts (``materials/llm_heads/prompt.txt``): the ``all`` prompt with every row, and one
  prompt per batch; ten example items drawn (seed 0) from the half's pooled prompts.

Also compares the new readings with E4's stored top-10 readings (``data/readings/heldout``): the
share of (prompt, readout, head) whose E4 top 10 is inside the new top 20, whose top-10 set is the
same, and the mean overlap of the two top 10s (near-ties in bfloat16 can reorder tokens).

Writes ``data/llm_heads/idmap.json``, ``data/llm_heads/tables/{task}_{half}_{readout}.json``
and ``data/llm_heads/prompts/{task}_{half}_{readout}_{all|batchN}.txt.gz``.

    .venv/bin/python scripts/llm_heads_tables.py --tasks english-german
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, MATERIALS, READOUTS, simple  # noqa: E402
from lth.store import load  # noqa: E402

N_LAYERS, N_HEADS = 32, 32
N_ALL = N_LAYERS * N_HEADS
K, N_TOKENS, BATCH, N_EXAMPLES, SEED = 20, 20, 128, 10, 0
READINGS = DATA / "readings" / "heldout_top20"
E4_READINGS = DATA / "readings" / "heldout"
OUT = DATA / "llm_heads"
PROMPT = MATERIALS / "llm_heads" / "prompt.txt"
NOTE = {"all": "The table lists all 1,024 heads.",
        "batch": "The table lists 128 of the 1,024 heads; the others are scored in separate requests."}
PERM = np.random.default_rng(SEED).permutation(N_ALL)  # id j -> head PERM[j]


def head_id(j: int) -> str:
    return f"H{j:04d}"


def share(v: float) -> str:
    """Two decimals, without the leading zero: .41, .00, 1.00."""
    s = f"{v:.2f}"
    return s[1:] if s.startswith("0") else s


def pooled(task: str, half: str):
    """Top-20 readings of the five rule templates of a half, pooled: (top [P, 2, 1024, 20], items)."""
    tops, items = [], []
    for wording in simple.WORDINGS["rule"]:
        run = load(READINGS / half / "simple" / task, wording)
        assert run["top"].shape[1:] == (len(READOUTS), N_ALL, K), run["top"].shape
        old = load(E4_READINGS / half / "simple" / task, wording)
        assert list(old["id_items"]) == list(run["id_items"]), f"{task} {half} {wording}: other prompts than E4"
        new10, old10 = run["top"][..., :10], old["top"]
        inside = (old10[..., :, None] == run["top"][..., None, :]).any(-1).all(-1).mean()
        same_set = (np.sort(new10, -1) == np.sort(old10, -1)).all(-1).mean()
        overlap = (new10[..., :, None] == old10[..., None, :]).any(-1).sum(-1).mean()
        print(f"  {half} {wording}: {len(run['id_items'])} prompts; of (prompt, readout, head): E4's top 10 "
              f"inside the top 20 {inside:.4f}, same top-10 set {same_set:.4f}, mean overlap {overlap:.2f}/10",
              flush=True)
        tops.append(run["top"])
        items += [int(i) for i in run["id_items"]]
    return np.concatenate(tops), items


def rows(top: np.ndarray, answers: np.ndarray, inputs: np.ndarray, decode) -> list[dict]:
    """One row per head, indexed by head (l * 32 + h), for one readout: top [P, 1024, 20]."""
    n = len(top)
    answer = (top == answers[:, None, None]).any(-1).mean(0)
    inp = (top == inputs[:, None, None]).any(-1).mean(0)
    out = []
    for h in range(N_ALL):
        counts = Counter(top[:, h].ravel().tolist())
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:N_TOKENS]
        out.append({"layer": h // N_HEADS, "head": h % N_HEADS, "answer_share": round(float(answer[h]), 4),
                    "input_share": round(float(inp[h]), 4),
                    "tokens": [[int(t), decode(t), round(c / n, 4)] for t, c in best]})
    return out


def render_row(j: int, row: dict) -> str:
    tokens = ", ".join(f"{json.dumps(s, ensure_ascii=False)} {share(v)}" for _, s, v in row["tokens"])
    return f"{head_id(j)} | answer {share(row['answer_share'])} | input {share(row['input_share'])} | {tokens}"


def fill(template: str, *, TABLE: str, **values: str) -> str:
    """The prompt with its placeholders filled; the table last, since tokens may contain ``<<``."""
    for key, value in values.items():
        assert f"<<{key}>>" in template, key
        template = template.replace(f"<<{key}>>", value)
    assert re.findall(r"<<[A-Z_]+>>", template) == ["<<TABLE>>"], "unfilled placeholder"
    return template.replace("<<TABLE>>", TABLE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--halves", nargs="+", default=["A", "B"])
    args = parser.parse_args()

    from lth.evaluation import tokenizer as get_tokenizer

    tok = get_tokenizer()
    cache: dict[int, str] = {}

    def decode(t: int) -> str:
        if t not in cache:
            cache[t] = tok.decode([t])
        return cache[t]

    def first(text: str) -> int:
        return tok(text, add_special_tokens=False).input_ids[0]

    template = PROMPT.read_text()
    descriptions = json.loads((MATERIALS / "generation" / "e4b_tasks.json").read_text())
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "prompts").mkdir(parents=True, exist_ok=True)
    idmap = {head_id(j): [int(PERM[j]) // N_HEADS, int(PERM[j]) % N_HEADS] for j in range(N_ALL)}
    (OUT / "idmap.json").write_text(json.dumps({"seed": SEED, "batch": BATCH, "ids": idmap}, indent=0))

    for task in args.tasks:
        data = simple.dataset(task)
        instructions = simple.instruction_dict()[task]
        templates = "\n".join(f"- {json.dumps(instructions[w[4:]], ensure_ascii=False)}"
                              for w in simple.WORDINGS["rule"])
        for half in args.halves:
            print(f"{task} {half}", flush=True)
            top, items = pooled(task, half)
            answers = np.array([first(" " + data[i]["output"]) for i in items])
            inputs = np.array([first(" " + data[i]["input"]) for i in items])
            unique = sorted(set(items))
            chosen = sorted(np.random.default_rng(SEED).choice(unique, N_EXAMPLES, replace=False).tolist())
            examples = "\n".join(f"- {data[i]['input']} -> {data[i]['output']}" for i in chosen)
            common = dict(DESCRIPTION=descriptions[task]["description"], TEMPLATES=templates,
                          EXAMPLES=examples, N_PROMPTS=str(len(items)))
            for r, readout in enumerate(READOUTS):
                by_head = rows(top[:, r], answers, inputs, decode)
                lines = [render_row(j, by_head[PERM[j]]) for j in range(N_ALL)]
                table = {"task": task, "half": half, "readout": readout, "n_prompts": len(items),
                         "examples": chosen, "rows": {head_id(j): by_head[PERM[j]] for j in range(N_ALL)}}
                stem = f"{task}_{half}_{readout}"
                (OUT / "tables" / f"{stem}.json").write_text(json.dumps(table, ensure_ascii=False))
                prompts = {"all": fill(template, TABLE_NOTE=NOTE["all"], TABLE="\n".join(lines), **common)}
                for b in range(N_ALL // BATCH):
                    prompts[f"batch{b}"] = fill(template, TABLE_NOTE=NOTE["batch"],
                                                TABLE="\n".join(lines[b * BATCH:(b + 1) * BATCH]), **common)
                for name, text in prompts.items():
                    with gzip.open(OUT / "prompts" / f"{stem}_{name}.txt.gz", "wt", encoding="utf-8") as f:
                        f.write(text)
                n_answer = sum(row["answer_share"] >= 0.10 for row in by_head)
                print(f"  {readout}: {len(items)} prompts; heads with answer share >= .10: {n_answer}; "
                      f"all prompt {len(prompts['all']):,} characters", flush=True)


if __name__ == "__main__":
    main()
