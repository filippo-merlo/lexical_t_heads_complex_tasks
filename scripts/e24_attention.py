"""E24 step B (GPU, forward passes only): E20's last-token attention pass on the 5-shot example prompts.

The example prompts are `` x1: y1; ... x5: y5; x:`` (Yang et al.'s example style). Every position
gets one of E20's categories (``scripts/e20_bai_attention.py``, ``CATS``), in this order of priority:

  bos       position 0;
  task      never (example prompts have no instruction, so no task words; T is not built, user 2026-11-16);
  query     a token overlapping the query x;
  self      the last prompt token (the final ":");
  template  a token overlapping the ":" after a demonstration input or the ";" after its output
            (Bai et al.'s template tokens of in-context learning);
  punct     any other token without a letter or digit;
  function  any other token whose word is in NLTK's English stopword list;
  content   every other token (the demonstrations' words).

M (template) and S (punct + function) are then Bai et al.'s template and stopword types in their own
setting. Prompts, model and attention as E20: the correctly answered prompts stored in the half's
shot5 readings (``scripts/heldout_readings.py --wordings shot5``), run alone, eager attention,
last-token row.

Writes ``data/readings/heldout/attention24_<half>/<task>.npy`` ([prompts, 1024 heads, 8] in the
order of ``CATS``, float16) and ``info.json``; ``--show`` prints the categories of each task's first
prompt and needs no GPU.

    LTH_READINGS=data/readings/heldout/A CUDA_DEVICE_ORDER=PCI_BUS_ID \\
        .venv/bin/python scripts/e24_attention.py --gpu 0 --fold 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, simple  # noqa: E402

from e20_bai_attention import CATS, STOPWORDS, WORD  # noqa: E402

HALF = {1: "A", 2: "B"}
WORDING = "shot5"
N_SHOT = 5


def scaffold(data: list[dict], item: int) -> tuple[str, list[tuple[int, int]], tuple[int, int]]:
    """The 5-shot prompt of ``item`` rebuilt as ``simple.create_few_shot_prompts`` builds it, the
    character spans of its demonstration delimiters (":" and ";"), and the query's span."""
    text, spans = "", []
    for j in range(item - N_SHOT, item):
        ex = data[j]
        text += " " + ex["input"]
        spans.append((len(text), len(text) + 1))
        text += ": " + ex["output"]
        spans.append((len(text), len(text) + 1))
        text += ";"
    qs = len(text) + 1
    text += " " + data[item]["input"] + ":"
    return text, spans, (qs, qs + len(data[item]["input"]))


def categorise(text: str, spans, qspan, ids: list[int], off: list[tuple[int, int]]) -> np.ndarray:
    words = [(m.start(), m.end(), m.group(0).lower()) for m in WORD.finditer(text)]
    cat = np.empty(len(ids), dtype=np.int8)
    for i, (a, b) in enumerate(off):
        if i == 0:
            c = "bos"
        elif a < qspan[1] and b > qspan[0]:
            c = "query"
        elif i == len(ids) - 1:
            c = "self"
        elif any(a < e and b > s for s, e in spans):
            c = "template"
        elif not any(ch.isalnum() for ch in text[a:b]):
            c = "punct"
        elif all(w in STOPWORDS for s, e, w in words if s < b and e > a):
            c = "function"
        else:
            c = "content"
        cat[i] = CATS.index(c)
    return cat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--show", action="store_true", help="print each task's first prompt by category; no GPU")
    args = parser.parse_args()
    half = HALF[args.fold]

    if args.show:
        from lth.evaluation import tokenizer
        tok = tokenizer()
        for t in simple.TASKS:
            data = simple.dataset(t)
            text, spans, qspan = scaffold(data, 7)
            assert text == simple.prompts(t, WORDING)[0][7]
            enc = tok(text, return_offsets_mapping=True)
            cat = categorise(text, spans, qspan, enc.input_ids, enc.offset_mapping)
            print(f"{t}: " + " ".join(f"[{tok.decode([i])!r} {CATS[c]}]" for i, c in zip(enc.input_ids[1:], cat[1:])))
        return

    from lth import gpu
    gpu.pin(args.gpu)
    import torch

    from lth.headvis import _eager
    from lth.model import LM
    from lth.store import READINGS, load, readings_dir

    assert READINGS.resolve() == (DATA / "readings" / "heldout" / half).resolve(), f"set LTH_READINGS to the half's readings ({half})"
    out = DATA / "readings" / "heldout" / f"attention24_{half}"
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
                prompts = simple.prompts(t, WORDING)[0]
                masses, counts = [], np.zeros(len(CATS), dtype=np.int64)
                for item in load(readings_dir("simple", t), WORDING)["id_items"]:
                    text, spans, qspan = scaffold(data, int(item))
                    assert text == prompts[int(item)], (t, item)
                    enc = tok(text, return_offsets_mapping=True)
                    ids, off = enc.input_ids, enc.offset_mapping
                    cat = categorise(text, spans, qspan, ids, off)
                    lm.model.hf_model(input_ids=torch.tensor([ids], device=lm.device), use_cache=False, logits_to_keep=1)
                    a = np.stack([rows[l] for l in range(n_layers)]).reshape(n_layers * rows[0].shape[0], -1)
                    onehot = np.zeros((len(ids), len(CATS)))
                    onehot[np.arange(len(ids)), cat] = 1
                    masses.append((a @ onehot).astype(np.float16))
                    counts += onehot.sum(0).astype(np.int64)
                np.save(out / f"{t}.npy", np.stack(masses))
                info[t] = dict(n_prompts=len(masses), positions=dict(zip(CATS, map(int, counts))))
                print(t, len(masses), "prompts", flush=True)
    finally:
        for h in handles:
            h.remove()
    (out / "info.json").write_text(json.dumps(dict(categories=CATS, wording=WORDING, tasks=info), indent=1))


if __name__ == "__main__":
    main()
