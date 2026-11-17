"""Rebuilt design (paper/head-sets-simple-design.tex, Sec. 5.3): the last-token attention row of every
head on the choosing prompts of each half, stored whole so that segments are assigned afterwards
without a GPU (the task-word segment depends on the fold's word list).

Prompts: for each of Yang et al.'s five templates, the first 100 correctly answered prompts of the half
(those of the half's readings, ``data/readings/heldout/<half>/simple/<task>/inst<k>``), run alone (no
padding), eager attention, the attention row of the last position. Writes
``data/readings/rebuild/attention_<half>/<task>.npz``: ``rows`` [sum of prompt lengths, n_layers *
n_heads] (float16, prompts one after another, positions in order), ``lengths``, ``wordings`` (template
index into ``lth.simple.WORDINGS["rule"]``) and ``items``. A task already there is skipped.

    .venv/bin/python scripts/rebuild_attention.py --gpu 0 --halves A B --tasks antonym synonym
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, simple  # noqa: E402

OUT = DATA / "readings" / "rebuild"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--halves", nargs="+", choices=("A", "B"), default=["A", "B"])
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS))
    args = parser.parse_args()

    from lth import gpu
    gpu.pin(args.gpu)
    import torch

    from lth.headvis import _eager
    from lth.model import LM
    from lth.store import load

    lm = LM.load()
    n_layers = lm.spec.n_layers
    rows = {}

    def hook(layer):
        def fn(module, inputs, output):
            rows[layer] = output[1][0, :, -1].to(torch.float16).cpu().numpy()  # [n_heads, T]
            return output[0], None
        return fn

    handles = [lm.model.layers[l].self_attn.register_forward_hook(hook(l)) for l in range(n_layers)]
    wordings = simple.WORDINGS["rule"]
    try:
        with _eager(lm.model.hf_model), torch.no_grad():
            for half in args.halves:
                out = OUT / f"attention_{half}"
                out.mkdir(parents=True, exist_ok=True)
                for task in args.tasks:
                    path = out / f"{task}.npz"
                    if path.exists():
                        continue
                    start = time.time()
                    blocks, lengths, wids, items = [], [], [], []
                    for j, w in enumerate(wordings):
                        prompts = simple.prompts(task, w)[0]
                        directory = DATA / "readings" / "heldout" / half / "simple" / task
                        for item in load(directory, w)["id_items"]:
                            ids = lm.tokenizer(prompts[int(item)]).input_ids
                            lm.model.hf_model(input_ids=torch.tensor([ids], device=lm.device), use_cache=False,
                                              logits_to_keep=1)
                            a = np.stack([rows[l] for l in range(n_layers)])  # [L, H, T]
                            blocks.append(a.reshape(-1, len(ids)).T)  # [T, L*H]
                            lengths.append(len(ids))
                            wids.append(j)
                            items.append(int(item))
                    np.savez(path, rows=np.concatenate(blocks), lengths=np.array(lengths),
                             wordings=np.array(wids, dtype=np.int8), items=np.array(items))
                    print(f"{half} {task}: {len(lengths)} prompts ({time.time() - start:.0f} s)", flush=True)
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
