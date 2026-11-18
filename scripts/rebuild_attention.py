"""Rebuilt design (paper/head-sets-simple-design.tex, Sec. 5.3): the last-token attention row of every
head on the choosing prompts of each half, stored whole so that segments are assigned afterwards
without a GPU (the task-word segment depends on the fold's word list).

Prompts: for each of Yang et al.'s five templates, the first 100 correctly answered prompts of the half
(those of the half's readings, ``data/readings/heldout/<half>/simple/<task>/inst<k>``), run alone (no
padding), eager attention, the attention row of the last position. Writes
``data/readings/rebuild/attention_<half>/<task>.npz``: ``rows`` [sum of prompt lengths, n_layers *
n_heads] (float16, prompts one after another, positions in order), ``lengths``, ``wordings`` (template
index into ``lth.simple.WORDINGS["rule"]``) and ``items``. A task already there is skipped.

Stage 3 (Sec. 6): ``--pool p``, the first ``--per`` (20) correctly answered prompts of each of the
task's P wordings (``data/readings/rebuild/attention_p_<half>``); ``--pool pformats``, the first 100
of each P example format (``attention_fmt_<half>``). Their files also hold ``names``, the wording ids
that ``wordings`` indexes.

    .venv/bin/python scripts/rebuild_attention.py --gpu 0 --halves A B --tasks antonym synonym
    .venv/bin/python scripts/rebuild_attention.py --gpu 0 --pool p
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
    parser.add_argument("--pool", choices=("rule", "p", "pformats"), default="rule",
                        help="Yang et al.'s five templates, the task's P wordings or the P example formats")
    parser.add_argument("--per", type=int, default=None, help="prompts per wording (default: 100; P wordings 20)")
    args = parser.parse_args()
    per = args.per or (20 if args.pool == "p" else 100)

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
    try:
        with _eager(lm.model.hf_model), torch.no_grad():
            for half in args.halves:
                out = OUT / {"rule": f"attention_{half}", "p": f"attention_p_{half}",
                             "pformats": f"attention_fmt_{half}"}[args.pool]
                out.mkdir(parents=True, exist_ok=True)
                for task in args.tasks:
                    path = out / f"{task}.npz"
                    if path.exists():
                        continue
                    start = time.time()
                    wordings = {"rule": simple.WORDINGS["rule"], "p": simple.split_wordings(task, "P"),
                                "pformats": simple.split_formats("P")}[args.pool]
                    blocks, lengths, wids, items = [], [], [], []
                    for j, w in enumerate(wordings):
                        prompts = simple.prompts(task, w)[0]
                        directory = DATA / "readings" / "heldout" / half / "simple" / task
                        for item in load(directory, w)["id_items"][:per]:
                            ids = lm.tokenizer(prompts[int(item)]).input_ids
                            lm.model.hf_model(input_ids=torch.tensor([ids], device=lm.device), use_cache=False,
                                              logits_to_keep=1)
                            a = np.stack([rows[l] for l in range(n_layers)])  # [L, H, T]
                            blocks.append(a.reshape(-1, len(ids)).T)  # [T, L*H]
                            lengths.append(len(ids))
                            wids.append(j)
                            items.append(int(item))
                    np.savez(path, rows=np.concatenate(blocks), lengths=np.array(lengths),
                             wordings=np.array(wids, dtype=np.int8), items=np.array(items), names=np.array(wordings))
                    print(f"{half} {task}: {len(lengths)} prompts ({time.time() - start:.0f} s)", flush=True)
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
