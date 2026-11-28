"""J decomposition of the last token's head outputs over positions (user 2026-11-28), for the rule J-T
(``scripts/jdec_sets.py``): forward passes only.

A head's output at the last position ``n`` is the attention-weighted mix of its value vectors,
``z = sum_j a[n, j] v_j``, and its J-head score of a direction ``q`` (a scaled unembedding row, ``w_t * g``,
``src.heads.effective_unembed``) is linear in ``z``: ``q^T J_{l,h} z = sum_j a[n, j] (J_{l,h}^T q)^T v_j``. Each term
``c_j = a[n, j] (J_{l,h}^T q)^T v_j`` is what the head reads from position ``j`` towards ``q``; the terms add up to
the head's score. It is AttnLRP's relevance of the attention weight (``lth.lrp``: ``a[n, j] g~^T v_j``) with the
fitted Jacobian's pullback of ``q`` in place of the backward gradient.

Prompts: each half's identification pool, as ``scripts/rebuild_lrp.py`` (``rebuild_sets.pool``), run alone, eager
attention. Directions (``q``, all through the lens ``J^z`` of ``lth.model.LM``):
- ``list``: the half's fold's 30-word list (``results/rebuild/additions/fold<f>/word_lists.json``; half A = fold 1,
  B = fold 2): each word the mean of the scaled unembedding rows of its tokens (``e13_lists.token_words``: every
  token whose stripped, lower-cased form is the word), the list the mean of its words;
- ``mean``: the mean scaled unembedding row over the vocabulary (a head writing along it raises every logit alike;
  ``list - mean`` is the list's direction with that removed);
- ``answer``: the prompt's target, the first token of " " + answer (LRP-T's target).

Writes ``data/readings/rebuild/jdec_<half>/<task>.npz``: ``rows_<dir>`` [sum of prompt lengths, n_layers * n_heads]
(float32, ``c_j`` of every head, prompts one after another, positions in order), ``total_<dir>`` [prompts,
n_layers * n_heads] (``(J_{l,h}^T q)^T z_h`` at the last position, ``z`` being ``o_proj``'s input), ``lengths``,
``wordings``, ``items``, ``target``. A task already there is skipped.

Checks on the first prompt of every task, before its file is written (the run stops at a failure): (a) the
attention row equals the one ``scripts/rebuild_attention.py`` stored, bit for bit; (b) for every direction and head,
the row summed over positions equals ``total`` within 1% of the largest ``|total|`` (the value heads are mapped to
query heads correctly); (c) ``q^T`` of head_level_jlens's transported head vectors (``all_heads_transported``,
the J-head readout of ``LM.top_tokens``) equals ``total`` within 1e-4 of its largest value (the head columns of
``J^z`` are sliced as the readout slices them).

    .venv/bin/python scripts/rebuild_jdec.py --gpu 0 --halves A --tasks antonym --check-only
    .venv/bin/python scripts/rebuild_jdec.py --gpu 0 --halves A
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

from lth import DATA, RESULTS, simple  # noqa: E402

OUT = DATA / "readings" / "rebuild"
FOLD_OF = {"A": 1, "B": 2}
DIRS = ("list", "mean", "answer")


def list_direction(lm, words: list[str], tmap: np.ndarray, vocab_words: list[str]):
    """Mean over the list's words of the mean scaled unembedding row of each word's tokens [d_model] (float32)."""
    import torch

    from src.heads import effective_unembed

    index = {w: i for i, w in enumerate(vocab_words)}
    rows = []
    for w in words:
        ids = np.flatnonzero(tmap == index[w])
        assert len(ids), w
        rows.append(effective_unembed(lm.model, torch.tensor(ids, device=lm.device)).float().mean(0))
    return torch.stack(rows).mean(0)


def vocab_mean(lm):
    import torch

    from src.heads import effective_unembed

    n = lm.model._lm_head.weight.shape[0]
    total = torch.zeros(lm.spec.d_model, dtype=torch.float32, device=lm.device)
    for s in range(0, n, 16384):
        total += effective_unembed(lm.model, torch.arange(s, min(s + 16384, n), device=lm.device)).float().sum(0)
    return total / n


def decompose(lm, ids: list[int], q, *, extras: bool = False) -> dict:
    """One forward pass; ``q`` [n_dir, d_model] (float32). ``rows`` [n_dir, T, L * H], ``total`` [n_dir, L * H]."""
    import torch

    from lth.headvis import _eager

    hf = lm.model.hf_model
    L, H, D = lm.spec.n_layers, lm.spec.n_heads, lm.spec.head_dim
    n_kv = hf.config.num_key_value_heads
    att, val, zin, handles = {}, {}, {}, []

    for layer in range(L):
        block = lm.model.layers[layer].self_attn
        handles.append(block.register_forward_hook(
            lambda _m, _a, out, layer=layer: att.__setitem__(layer, out[1][0, :, -1])))  # [H, T]
        handles.append(block.v_proj.register_forward_hook(
            lambda _m, _a, out, layer=layer: val.__setitem__(layer, out[0])))  # [T, n_kv * D]
        handles.append(block.o_proj.register_forward_pre_hook(
            lambda _m, args, layer=layer: zin.__setitem__(layer, args[0][0, -1])))  # [H * D]
    try:
        with _eager(hf), torch.no_grad():
            hf(input_ids=torch.tensor([ids], device=lm.device), use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    kv_of = torch.arange(H, device=lm.device) // (H // n_kv)
    rows, total = [], []
    with torch.no_grad():
        for layer in range(L):
            w = (q @ lm.jz[layer].float()).view(len(q), H, D)  # J_{l,h}^T q per head
            v = val[layer].float().view(len(ids), n_kv, D)[:, kv_of]  # [T, H, D], value head of each query head
            proj = torch.einsum("thd,khd->kth", v, w)
            rows.append(proj * att[layer].float().T[None])  # [n_dir, T, H]
            total.append(torch.einsum("hd,khd->kh", zin[layer].float().view(H, D), w))
        out = {"rows": torch.cat(rows, dim=-1).cpu().numpy(), "total": torch.cat(total, dim=-1).cpu().numpy()}
        if extras:
            from src.readout import all_heads_transported

            out["att"] = torch.stack([att[l].to(torch.float16) for l in range(L)]).reshape(L * H, -1).T.cpu().numpy()
            out["readout"] = torch.cat([q @ all_heads_transported(lm.jz[l], zin[l], lm.spec, arm="jhead").float().T
                                        for l in range(L)], dim=-1).cpu().numpy()
    return out


def check(r, stored, label: str) -> None:
    a = r["att"].shape == stored.shape and np.array_equal(r["att"], stored)
    scale = np.abs(r["total"]).max(-1)  # per direction
    b = float((np.abs(r["rows"].sum(1) - r["total"]).max(-1) / scale).max())
    c = float((np.abs(r["readout"] - r["total"]).max(-1) / scale).max())
    ok = a and b <= 1e-2 and c <= 1e-4
    print(f"check {label}: (a) attention row equal {a}; (b) row sum vs total {b:.1e}; (c) J-head readout vs total "
          f"{c:.1e} -> {'ok' if ok else 'FAILED'}", flush=True)
    if not ok:
        raise SystemExit(f"check failed on {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--halves", nargs="+", choices=("A", "B"), default=["A", "B"])
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS))
    parser.add_argument("--check-only", action="store_true", help="run the checks on each task's first prompt only")
    args = parser.parse_args()

    from lth import gpu
    gpu.pin(args.gpu)

    import torch
    from e13_lists import token_words
    from rebuild_lrp import stored_rows
    from rebuild_sets import pool
    from src.heads import effective_unembed

    from lth.model import LM

    lm = LM.load()
    tok = lm.tokenizer
    tmap, vocab_words = token_words(tok)
    q_mean = vocab_mean(lm)
    for half in args.halves:
        out = OUT / f"jdec_{half}"
        out.mkdir(parents=True, exist_ok=True)
        lists = json.loads((RESULTS / "rebuild" / "additions" / f"fold{FOLD_OF[half]}" / "word_lists.json").read_text())["tasks"]
        for task in args.tasks:
            path = out / f"{task}.npz"
            if path.exists() and not args.check_only:
                print(f"{half} {task}: done before, skipped", flush=True)
                continue
            start = time.time()
            q_list = list_direction(lm, lists[task]["words"], tmap, vocab_words)
            keys = pool(half, task)[0]
            data = simple.dataset(task)
            prompts = {w: simple.prompts(task, w)[0] for w in simple.WORDINGS["rule"]}
            stored = stored_rows(half, task)
            kept = {k: [] for k in ("lengths", "wordings", "items", "target")}
            kept.update({f"{p}_{d}": [] for p in ("rows", "total") for d in DIRS})
            for k, (j, item) in enumerate(keys):
                ids = tok(prompts[simple.WORDINGS["rule"][j]][item]).input_ids
                target = lm.first_token(" " + data[item]["output"])
                q_y = effective_unembed(lm.model, torch.tensor([target], device=lm.device)).float()[0]
                r = decompose(lm, ids, torch.stack([q_list, q_mean, q_y]), extras=k == 0)
                if k == 0:
                    check(r, stored[(j, item)], f"{half} {task} {simple.WORDINGS['rule'][j]} item {item}")
                    if args.check_only:
                        break
                for name, value in (("lengths", len(ids)), ("wordings", j), ("items", item), ("target", target)):
                    kept[name].append(value)
                for d, direction in enumerate(DIRS):
                    kept[f"rows_{direction}"].append(r["rows"][d])
                    kept[f"total_{direction}"].append(r["total"][d])
            if args.check_only:
                continue
            partial = out / f"{task}.partial.npz"
            np.savez(partial, lengths=np.array(kept["lengths"]), wordings=np.array(kept["wordings"], dtype=np.int8),
                     items=np.array(kept["items"]), target=np.array(kept["target"]),
                     **{f"rows_{d}": np.concatenate(kept[f"rows_{d}"]).astype(np.float32) for d in DIRS},
                     **{f"total_{d}": np.stack(kept[f"total_{d}"]).astype(np.float32) for d in DIRS})
            partial.replace(path)
            print(f"{half} {task}: {len(keys)} prompts ({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
