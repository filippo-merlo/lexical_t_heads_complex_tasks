"""Lexical task heads and the measures of proposal Sec. 4.4 (no GPU).

A head matches a prompt when at least one of its top-10 tokens is in the task's term list
(expanded to token ids as in Yang et al.); it is a lexical task head for a task, style and
readout when it matches on at least 10% of that style's identification prompts, pooled
over the style's wordings (Definition 1; n = 1, k = 10, rho = 0.1).

Writes ``results/heads.json`` (every head set with each head's match rate),
``results/measures.csv`` (prompts, heads, mean layer and copy share per set) and
``results/overlaps.csv`` (Logit-head vs J-head, and rule vs example, per task). Families:
``simple``, ``complex``, and for the tasks run in reasoning mode ``reasoning`` (last prompt
token) and ``reasoning-pre`` (pre-answer position).

    .venv/bin/python scripts/identify_heads.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import MATERIALS, READOUTS, RESULTS, simple  # noqa: E402
from lth.store import done, load, run_dir  # noqa: E402

RHO = 0.10
N_HEADS = 32
PRE = "@pre"  # reasoning mode: readings at the pre-answer position (scripts/run_reasoning.py)


def complex_styles() -> dict[str, tuple[str, ...]]:
    wordings = ("orig",)
    if (MATERIALS / "complex" / "wordings.json").exists():
        wordings += tuple(f"w{i}" for i in range(1, 7))
    return {"rule": wordings, "example": ("examples",), "combined": ("combined",)}


def pre_answer_texts(directory: Path, wording: str) -> dict[str, str]:
    """Item -> prompt plus the reply up to the pre-answer position (reasoning mode)."""
    from lth import fu

    prompts = fu.recorded_prompts(directory / "fu_out" / wording.removesuffix(PRE))
    return {r["item"]: prompts[r["item"]] + r["pre_text"] for r in load(directory, wording)["records"]
            if "pre_text" in r}


def copy_share(top: np.ndarray, matched: np.ndarray, texts: list[str], heads, term_ids: set[int],
               tokenizer) -> float:
    """Share of a head's matching prompts that contain one of its matched words, averaged over heads."""
    shares = []
    for head in heads:
        rows = np.flatnonzero(matched[:, head])
        hits = 0
        for row in rows:
            words = {tokenizer.decode([t]).strip(" ._-").lower() for t in top[row, head] if t in term_ids}
            text = texts[row].lower()
            hits += any(w and re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", text) for w in words)
        shares.append(hits / len(rows))
    return float(np.mean(shares)) if shares else float("nan")


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    from transformers import AutoTokenizer

    from lth import fu
    from lth.terms import complex_term_lists, expand_to_token_ids, simple_term_lists
    from src.config import LENSES

    tokenizer = AutoTokenizer.from_pretrained(LENSES["llama3.1-8b-it"].hf_model_id)
    jobs = []  # (family, task, styles, term list, prompt texts by wording)
    simple_terms = simple_term_lists()
    for task in simple.TASKS:
        directory = run_dir("simple", task)
        if not done(directory, simple.FIXING):
            continue
        texts = {w: dict(enumerate(simple.prompts(task, w)[0])) for ws in simple.WORDINGS.values() for w in ws}
        jobs.append(("simple", task, simple.WORDINGS, simple_terms[task], directory,
                     lambda w, texts=texts: {str(k): v for k, v in texts[w].items()}))
    if (MATERIALS / "complex" / "terms.json").exists():
        complex_terms = complex_term_lists()
        for key in fu.STUDY_LEVELS:
            level = fu.TASK_LEVELS[key]
            terms = complex_terms[level.original]
            if level.task not in fu.REASONING_TASKS:
                directory = run_dir("complex", key)
                jobs.append(("complex", key, complex_styles(), terms, directory,
                             lambda w, directory=directory: fu.recorded_prompts(directory / "fu_out" / w)))
                continue
            # Reasoning mode: heads at the last prompt token, and at the pre-answer position,
            # where the copy share also checks the reply up to that point.
            directory = run_dir("reasoning", key)
            jobs.append(("reasoning", key, complex_styles(), terms, directory,
                         lambda w, directory=directory: fu.recorded_prompts(directory / "fu_out" / w)))
            jobs.append(("reasoning-pre", key,
                         {s: tuple(w + PRE for w in ws) for s, ws in complex_styles().items()},
                         terms, directory, lambda w, directory=directory: pre_answer_texts(directory, w)))
    else:
        print("no materials/complex/terms.json yet: complex tasks skipped")

    heads_out, measures, overlaps = {}, [], []
    for family, task, styles, words, directory, prompts_of in jobs:
        term_ids = set(expand_to_token_ids(tokenizer, words))
        ids = np.array(sorted(term_ids))
        sets = {}
        for style, wordings in styles.items():
            wordings = [w for w in wordings if done(directory, w)]
            if not wordings:
                continue
            tops, texts = [], []
            for wording in wordings:
                run = load(directory, wording)
                if len(run["id_items"]):
                    tops.append(run["top"])
                    by_item = prompts_of(wording)
                    texts += [by_item[item] for item in run["id_items"]]
            top = np.concatenate(tops) if tops else np.zeros((0, len(READOUTS), N_HEADS * N_HEADS, 10), np.int32)
            for r, readout in enumerate(READOUTS):
                matched = np.isin(top[:, r], ids).any(-1)  # [prompts, heads]
                rate = matched.mean(0) if len(top) else np.zeros(top.shape[2])
                heads = [int(h) for h in np.flatnonzero(rate >= RHO)]
                sets[(style, readout)] = set(heads)
                heads_out.setdefault(family, {}).setdefault(task, {}).setdefault(style, {})[readout] = {
                    "n_prompts": int(len(top)),
                    "heads": [[h // N_HEADS, h % N_HEADS, round(float(rate[h]), 4)] for h in heads]}
                measures.append({
                    "family": family, "task": task, "style": style, "readout": readout,
                    "n_prompts": len(top), "n_heads": len(heads),
                    "mean_layer": float(np.mean([h // N_HEADS for h in heads])) if heads else float("nan"),
                    "copy_share": copy_share(top[:, r], matched, texts, heads, term_ids, tokenizer)})
        pairs = [((s, "logithead"), (s, "jhead")) for s in styles] + \
                [(("rule", r), ("example", r)) for r in READOUTS]
        for a, b in pairs:
            if a in sets and b in sets:
                inter = len(sets[a] & sets[b])
                overlaps.append({"family": family, "task": task, "a": "/".join(a), "b": "/".join(b),
                                 "n_a": len(sets[a]), "n_b": len(sets[b]), "n_both": inter,
                                 "share_of_a": inter / len(sets[a]) if sets[a] else float("nan"),
                                 "share_of_b": inter / len(sets[b]) if sets[b] else float("nan")})
        print(f"{family} {task}: " + ", ".join(f"{s}/{r[:5]} {len(h)}" for (s, r), h in sets.items()))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "heads.json").write_text(json.dumps(heads_out, indent=1))
    for name, rows in (("measures", measures), ("overlaps", overlaps)):
        with (RESULTS / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {RESULTS}/heads.json, measures.csv, overlaps.csv")


if __name__ == "__main__":
    main()
