"""E20 step 1 (GPU, forward passes only): E13's last-token attention pass with its "rest" split
into the token types of Bai et al. (arXiv 2401.11323v4, Sec. 5.1).

E13 (``scripts/e13_attention.py``) keeps four masses per head: task-word positions, query, rest,
BOS. Bai et al. find that a test example draws most of what it needs from template tokens,
punctuation and stopwords, not from content words. Here every position of a rule prompt gets one
category, in this order of priority:

  bos       position 0;
  task      a task-word position, E13's rule exactly (whole-word, case-insensitive match of a word
            of the fold's E13 list, outside the query);
  query     a token overlapping the query (E13's rule);
  self      the last prompt token (the read-out position; ":" of "A:" or the cloze's last word);
  template  the question--answer scaffold around the query, " Q:" before it and "\\nA" after it
            (Bai et al.'s template tokens T_in/T_out with ":" and "\\n"); the cloze templates have none;
  punct     any other token without a letter or digit (".", "?", ":", "->", "'");
  function  any other token whose word is in NLTK's English stopword list (Bai et al.'s App. I list,
            ``materials/stopwords/nltk_english.txt``; "the", "of", "what", "is");
  content   every other instruction token.

Bai et al. sort tokens by hand, and their stopword type takes punctuation and function words
together; here the rule is automatic and the two are kept apart (sum them for Bai et al.'s type).
Bai et al. never mask the instruction, so the split of the instruction into punct / function /
content is our extension of their categories, not theirs.

Prompts, model and attention as E13: the correctly answered prompts stored in the half's readings
for Yang et al.'s five rule templates, run alone (no padding), eager attention, last-token row.

Writes ``data/readings/heldout/attention20_<half>/<task>.npy`` ([prompts, 1024 heads, 8] in the
order of ``CATS``, float16), ``templates.npy`` (the template index of each prompt) and
``info.json``; ``--show`` prints the category of every token of each template's first prompt and
needs no GPU.

    LTH_READINGS=data/readings/heldout/A CUDA_DEVICE_ORDER=PCI_BUS_ID \\
        .venv/bin/python scripts/e20_bai_attention.py --gpu 0 --fold 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lth import DATA, RESULTS, simple  # noqa: E402

HALF = {1: "A", 2: "B"}
CATS = ("bos", "task", "query", "self", "template", "punct", "function", "content")
STOPWORDS = frozenset(l.strip() for l in (ROOT / "materials" / "stopwords" / "nltk_english.txt").read_text().splitlines()
                      if l.strip() and not l.startswith("#"))
WORD = re.compile(r"\w+")


def template_spans(template: str, text: str, qspan: tuple[int, int]) -> list[tuple[int, int]]:
    """Character spans of the question--answer scaffold: " Q:" right before the query and the
    whole suffix after it when it starts with a newline ("\\nA:"). Cloze templates give none."""
    prefix, suffix = template.split("{input}")
    spans = []
    head = prefix.rstrip(" ")
    if head.endswith(" Q:"):
        spans.append((len(head) - 3, len(head)))
    if suffix.startswith("\n"):
        spans.append((qspan[1], qspan[1] + len(suffix)))
    assert text[:len(prefix)] == prefix and text.endswith(suffix), "prompt does not follow its template"
    return spans


def categorise(text: str, template: str, query: str, ids: list[int], off: list[tuple[int, int]],
               pattern: re.Pattern) -> tuple[np.ndarray, list[int], list[int]]:
    """Category index (into ``CATS``) of every position, and E13's task-word and query positions."""
    qs = text.rfind(query)
    qspan = (qs, qs + len(query))
    spans = [m.span() for m in pattern.finditer(text) if not (m.start() < qspan[1] and m.end() > qspan[0])]
    pos_kw = [i for i, (a, b) in enumerate(off) if i > 0 and any(a < e and b > s for s, e in spans)]
    pos_q = [i for i, (a, b) in enumerate(off) if i > 0 and a < qspan[1] and b > qspan[0]]
    tspans = template_spans(template, text, qspan)
    words = [(m.start(), m.end(), m.group(0).lower()) for m in WORD.finditer(text)]
    cat = np.empty(len(ids), dtype=np.int8)
    for i, (a, b) in enumerate(off):
        if i == 0:
            c = "bos"
        elif i in pos_kw:
            c = "task"
        elif i in pos_q:
            c = "query"
        elif i == len(ids) - 1:
            c = "self"
        elif any(a < e and b > s for s, e in tspans):
            c = "template"
        elif not any(ch.isalnum() for ch in text[a:b]):
            c = "punct"
        elif all(w in STOPWORDS for s, e, w in words if s < b and e > a):
            c = "function"
        else:
            c = "content"
        cat[i] = CATS.index(c)
    return cat, pos_kw, pos_q


def task_pattern(words: list[str]) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--show", action="store_true", help="print each template's first prompt by category; no GPU")
    args = parser.parse_args()
    half = HALF[args.fold]
    wordings = simple.WORDINGS["rule"]
    lists = json.loads((RESULTS / "e13_capped" / f"fold{args.fold}" / "word_lists.json").read_text())["tasks"]
    templates = {t: {w: simple.instruction_dict()[t][w[4:]] for w in wordings} for t in simple.TASKS}

    if args.show:
        from lth.evaluation import tokenizer
        tok = tokenizer()
        for t in simple.TASKS:
            data = simple.dataset(t)
            pat = task_pattern(lists[t]["words"])
            print(f"## {t}")
            for w in wordings:
                text = simple.prompts(t, w)[0][0]
                enc = tok(text, return_offsets_mapping=True)
                cat, _, _ = categorise(text, templates[t][w], data[0]["input"], enc.input_ids, enc.offset_mapping, pat)
                print(f"{w}: " + " ".join(f"[{tok.decode([i])!r} {CATS[c]}]" for i, c in zip(enc.input_ids[1:], cat[1:])))
        return

    from lth import gpu
    gpu.pin(args.gpu)
    import torch

    from lth.headvis import _eager
    from lth.model import LM
    from lth.store import READINGS, load, readings_dir

    assert READINGS.resolve() == (DATA / "readings" / "heldout" / half).resolve(), f"set LTH_READINGS to the half's readings ({half})"
    out = DATA / "readings" / "heldout" / f"attention20_{half}"
    out.mkdir(parents=True, exist_ok=True)
    lm = LM.load()
    tok = lm.tokenizer
    n_layers = lm.spec.n_layers
    rows = {}

    def hook(layer):
        def fn(module, inputs, output):
            rows[layer] = output[1][0, :, -1].float().cpu().numpy()  # [n_heads, T]
            return output[0], None
        return fn

    handles = [lm.model.layers[l].self_attn.register_forward_hook(hook(l)) for l in range(n_layers)]
    info = {}
    try:
        with _eager(lm.model.hf_model), torch.no_grad():
            for t in simple.TASKS:
                data = simple.dataset(t)
                pat = task_pattern(lists[t]["words"])
                masses, tmpl, counts, overlap = [], [], np.zeros((len(wordings), len(CATS)), dtype=np.int64), 0
                for j, w in enumerate(wordings):
                    prompts = simple.prompts(t, w)[0]
                    for item in load(readings_dir("simple", t), w)["id_items"]:
                        text = prompts[int(item)]
                        enc = tok(text, return_offsets_mapping=True)
                        ids, off = enc.input_ids, enc.offset_mapping
                        cat, pos_kw, pos_q = categorise(text, templates[t][w], data[int(item)]["input"], ids, off, pat)
                        overlap += len(set(pos_kw) & set(pos_q))
                        lm.model.hf_model(input_ids=torch.tensor([ids], device=lm.device), use_cache=False,
                                          logits_to_keep=1)
                        a = np.stack([rows[l] for l in range(n_layers)]).reshape(n_layers * rows[0].shape[0], -1)
                        onehot = np.zeros((len(ids), len(CATS)))
                        onehot[np.arange(len(ids)), cat] = 1
                        masses.append((a @ onehot).astype(np.float16))
                        tmpl.append(j)
                        counts[j] += onehot.sum(0).astype(np.int64)
                np.save(out / f"{t}.npy", np.stack(masses))
                np.save(out / f"{t}.templates.npy", np.array(tmpl, dtype=np.int8))
                info[t] = dict(n_prompts=len(masses), task_query_overlap=overlap,
                               positions_per_template={w: dict(zip(CATS, map(int, counts[j]))) for j, w in enumerate(wordings)})
                print(t, len(masses), "prompts", flush=True)
    finally:
        for h in handles:
            h.remove()
    (out / "info.json").write_text(json.dumps(dict(categories=CATS, tasks=info), indent=1))


if __name__ == "__main__":
    main()
