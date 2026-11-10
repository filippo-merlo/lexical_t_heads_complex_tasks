"""E9: last-token attention on BOS of every head on the complex tasks' rule prompts.

For each complex task (easy level) and rule wording (``orig``, ``w1``..``w6``), the prompts of
its identification items (the first 100 in P+, ``id_items`` of ``data/readings/complex``, the
prompts whose J-head readings E9's D filter uses), each run alone under the chat template as in
``scripts/build_readings.py``, under eager attention: the attention weight of the last prompt
token on position 0 (BOS) for every head. Only the BOS mass is kept, which is all the B filter
needs (E7/E8 on the simple tasks: ``scripts/e3/attention_pass.py``).

Writes ``data/readings/attention_complex/<task>.npy`` (prompts x 1,024 heads, float16, wordings
in the order of ``identify_heads.complex_styles``, items in ``id_items`` order) and
``<task>.json`` (the wording and item of each row).

    .venv/bin/python scripts/e9_attention.py --gpu 0 --levels set_game/easy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, fu, gpu  # noqa: E402

OUT = DATA / "readings" / "attention_complex"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--levels", nargs="+", default=list(fu.STUDY_LEVELS))
    args = parser.parse_args()
    gpu.pin(args.gpu)

    import torch

    from identify_heads import complex_styles
    from lth.headvis import _eager
    from lth.model import LM
    from lth.store import load, load_outputs, outputs_path, readings_dir

    lm = LM.load()
    n_layers = lm.spec.n_layers
    rows: dict[int, np.ndarray] = {}

    def hook(layer):
        def fn(module, inputs, output):
            rows[layer] = output[1][0, :, -1, 0].float().cpu().numpy()  # [n_heads]: last token -> BOS
            return output[0], None
        return fn

    OUT.mkdir(parents=True, exist_ok=True)
    handles = [lm.model.layers[l].self_attn.register_forward_hook(hook(l)) for l in range(n_layers)]
    try:
        with _eager(lm.model.hf_model), torch.no_grad():
            for key in args.levels:
                start = time.time()
                directory = readings_dir("complex", key)
                bos, order = [], []
                for wording in complex_styles()["rule"]:
                    prompts = {r["item"]: r["prompt"] for r in load_outputs(outputs_path("complex", key, wording))}
                    for item in load(directory, wording)["id_items"].tolist():
                        ids = lm.encode_chat(prompts[item])
                        assert ids[0] == lm.tokenizer.bos_token_id
                        rows.clear()
                        lm.model.hf_model(input_ids=torch.tensor([ids], device=lm.device), use_cache=False,
                                          logits_to_keep=1)
                        bos.append(np.concatenate([rows[l] for l in range(n_layers)]).astype(np.float16))
                        order.append({"wording": wording, "item": item, "n_tokens": len(ids)})
                name = key.replace("/", "__")
                np.save(OUT / f"{name}.npy", np.stack(bos))
                (OUT / f"{name}.json").write_text(json.dumps(order))
                print(f"{key}: {len(bos)} prompts, mean BOS {np.stack(bos).astype(np.float32).mean():.3f} "
                      f"({time.time() - start:.0f} s)", flush=True)
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
