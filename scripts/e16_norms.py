"""E16 step 1 (GPU, forward passes only): every head's output norm on the choosing half's rule prompts,
for the Avg Head Output Norm score of Sandoval-Segura et al. (2025, arXiv:2504.03889, Sec. 3).

The head output is the head's slice of ``o_proj``'s input (before the output projection), as in the
paper. Per prompt and head: the l2 norm of that slice at every token, averaged over all tokens
(AvgNorm, BOS included), and the norm at the last token (kept for reference, not used by E16).
Same prompts as ``e13_attention.py`` (the half's stored correct prompts, five rule templates
pooled, run alone without padding).

Writes ``data/readings/heldout/headnorm16_<half>/<task>.npy`` ([prompts, 1024 heads, 2]: AvgNorm,
last-token norm; float32).

    LTH_READINGS=data/readings/heldout/A CUDA_DEVICE_ORDER=PCI_BUS_ID \
        .venv/bin/python scripts/e16_norms.py --gpu 0 --fold 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, gpu, simple  # noqa: E402

HALF = {1: "A", 2: "B"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    half = HALF[args.fold]
    gpu.pin(args.gpu)

    import torch

    from lth.model import LM
    from lth.store import READINGS, load, readings_dir

    assert READINGS.resolve() == (DATA / "readings" / "heldout" / half).resolve(), f"set LTH_READINGS to the half's readings ({half})"
    out = DATA / "readings" / "heldout" / f"headnorm16_{half}"
    out.mkdir(parents=True, exist_ok=True)

    lm = LM.load()
    tok = lm.tokenizer
    n_layers, n_heads = lm.spec.n_layers, 32
    rows = {}

    def hook(layer):
        def fn(module, inputs):
            z = inputs[0][0].float()  # [T, n_heads * d_head]
            norms = z.view(z.shape[0], n_heads, -1).norm(dim=-1)  # [T, n_heads]
            rows[layer] = torch.stack([norms.mean(0), norms[-1]], 1).cpu().numpy()  # [n_heads, 2]
        return fn

    handles = [lm.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook(l)) for l in range(n_layers)]
    assert lm.model.layers[0].self_attn.o_proj.in_features == n_heads * 128
    try:
        with torch.no_grad():
            for t in simple.TASKS:
                vals = []
                for w in simple.WORDINGS["rule"]:
                    prompts = simple.prompts(t, w)[0]
                    for item in load(readings_dir("simple", t), w)["id_items"]:
                        ids = tok(prompts[int(item)]).input_ids
                        lm.model.hf_model(input_ids=torch.tensor([ids], device=lm.device), use_cache=False,
                                          logits_to_keep=1)
                        vals.append(np.concatenate([rows[l] for l in range(n_layers)]))  # [1024, 2]
                np.save(out / f"{t}.npy", np.stack(vals).astype(np.float32))
                print(t, len(vals), "prompts", flush=True)
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
