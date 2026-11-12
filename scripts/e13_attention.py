"""E13 step 2 (GPU, forward passes only): last-token attention of every head on the choosing half's
rule prompts, reduced to masses on the task-word positions, the query, the rest, and BOS.

As ``scripts/e3/attention_pass.py`` (prompts run alone, no padding, eager attention, the correct
prompts stored in the half's readings, five templates pooled), except for the task words: a
position is a task-word position if its token overlaps a whole-word, case-insensitive match of a
word of the fold's E13 list (``results/e13_capped/fold{f}/word_lists.json``) outside the query.
Yang et al.'s terms are no longer added.

Writes ``data/readings/heldout/attention13_<half>/<task>.npy`` ([prompts, 1024 heads, 4]: task
words, query, rest, BOS; float16) and ``info.json`` (the list, prompts without any task word, and
the words found per template).

    LTH_READINGS=data/readings/heldout/A CUDA_DEVICE_ORDER=PCI_BUS_ID \
        .venv/bin/python scripts/e13_attention.py --gpu 0 --fold 1
    (E14: add ``--e14``; the 35 templates' prompts, E14's lists, ``attention14_<half>``.)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, RESULTS, gpu, simple  # noqa: E402

HALF = {1: "A", 2: "B"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--e14", action="store_true",
                        help="E14: all 35 rule templates (Yang's five and the 30 new ones), the lists of "
                             "results/e14_wordings, output data/readings/heldout/attention14_<half>")
    args = parser.parse_args()
    half = HALF[args.fold]
    gpu.pin(args.gpu)

    import torch

    from lth.headvis import _eager
    from lth.model import LM
    from lth.store import READINGS, load, readings_dir

    assert READINGS.resolve() == (DATA / "readings" / "heldout" / half).resolve(), f"set LTH_READINGS to the half's readings ({half})"
    exp, tag = ("e14_wordings", "14") if args.e14 else ("e13_capped", "13")
    wordings = simple.ALL_RULE if args.e14 else simple.WORDINGS["rule"]
    lists = json.loads((RESULTS / exp / f"fold{args.fold}" / "word_lists.json").read_text())["tasks"]
    out = DATA / "readings" / "heldout" / f"attention{tag}_{half}"
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
                words = lists[t]["words"]
                data = simple.dataset(t)
                pat = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I)
                masses, n_kw, found = [], [], {}
                for w in wordings:
                    prompts = simple.prompts(t, w)[0]
                    seen = Counter()
                    for item in load(readings_dir("simple", t), w)["id_items"]:
                        text = prompts[int(item)]
                        q = data[int(item)]["input"]
                        qs = text.rfind(q)
                        qspan = (qs, qs + len(q))
                        enc = tok(text, return_offsets_mapping=True)
                        ids, off = enc.input_ids, enc.offset_mapping
                        kw = [m for m in pat.finditer(text) if not (m.start() < qspan[1] and m.end() > qspan[0])]
                        seen.update(m.group(0).lower() for m in kw)
                        spans = [m.span() for m in kw]
                        pos_kw = [i for i, (a, b) in enumerate(off) if i > 0 and any(a < e and b > s for s, e in spans)]
                        pos_q = [i for i, (a, b) in enumerate(off) if i > 0 and a < qspan[1] and b > qspan[0]]
                        pos_rest = [i for i in range(1, len(ids)) if i not in pos_kw and i not in pos_q]
                        lm.model.hf_model(input_ids=torch.tensor([ids], device=lm.device), use_cache=False,
                                          logits_to_keep=1)
                        a = np.stack([rows[l] for l in range(n_layers)]).reshape(n_layers * rows[0].shape[0], -1)
                        masses.append(np.stack([a[:, pos_kw].sum(1), a[:, pos_q].sum(1), a[:, pos_rest].sum(1),
                                                a[:, 0]], 1).astype(np.float16))
                        n_kw.append(len(pos_kw))
                    found[w] = dict(seen)
                np.save(out / f"{t}.npy", np.stack(masses))
                info[t] = dict(words=words, n_prompts=len(masses),
                               prompts_without_task_word=int(sum(k == 0 for k in n_kw)), found=found)
                print(t, len(masses), "prompts;", info[t]["prompts_without_task_word"], "without a task word", flush=True)
    finally:
        for h in handles:
            h.remove()
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
