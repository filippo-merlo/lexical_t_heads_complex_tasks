"""A lens estimate of Davidson et al.'s head score (CIE), next to the score measured on the
same prompts.

The CIE of a head (``scripts/davidson_fv.py``) is the gain in the answer's probability when,
on a prompt whose instruction was replaced by an uninformative one, the head's output at the
last token is set to its mean over correct prompts. The change made to the head is
``d = mean_z - z``. A readout maps it into the final residual stream, to first order:
``v = J_{l,h} d`` (J-head; ``J_{l,h}``, the head's block of the fitted ``J^z``, is an
average Jacobian of the final residual with respect to the head's input to ``o_proj``) or
``v = W_O^h d`` (Logit-head: the direct path only). Through the final RMSNorm, linearised at
the prompt's own final residual ``x`` (``r = rms(x)``, ``x_hat = x / r``), and the
unembedding ``W_U``:

    dy = gamma * (v - x_hat (x_hat . v) / d_model) / r,     dlogit_t = W_U[t] . dy,

and, with ``p`` the prompt's next-token distribution and ``a`` the answer's first token:

- ``logit``: ``dlogit_a``;
- ``logprob``: ``dlogit_a - sum_t p_t dlogit_t`` (first-order change in ``log p_a``);
- ``prob``: ``p_a`` times that (first-order change in ``p_a``, the CIE's unit).

No word list is used: the target of each prompt is its own answer token, as in the CIE.

The CIE itself is measured again, per prompt, on the same prompts (the uninformative
instructions are saved as text and re-encoded here, ``scripts/davidson_fv_diff.py``), with
``davidson_fv.py``'s intervention; its mean per task and kind can be compared with the saved
``cie.npz``. The task means are checked against the saved FV.

Writes ``results/<--out>/``: ``estimates.npz`` (``{task}/{readout}/{measure}``, the mean over
each kind's prompts, ``[3 kinds, n_layers, n_heads]``), ``measured.npz`` (``{task}``, per-prompt
CIE, ``[P, n_layers, n_heads]``), ``prompts.json`` (per prompt: kind, item, answer, ``p_a``,
the model's next token) and ``fv_check.json``; ``data/estimates/davidson_lens_cie/{task}.npz``
holds the per-prompt estimates (``[P, 2 readouts, 3 measures, n_layers * n_heads]``).

    .venv/bin/python scripts/davidson_fv_lens_cie.py --gpu 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, READOUTS, RESULTS, gpu  # noqa: E402

MEASURES = ("logit", "logprob", "prob")


def cie_rows(lm, rows: list[list[int]], targets: list[int], mean_z):
    """Per-prompt CIE of every head, ``[rows, n_layers, n_heads]``: ``davidson_fv.cie`` without
    the mean over rows."""
    import torch

    from davidson_fv import answer_prob

    n_heads, dim = lm.spec.n_heads, lm.spec.head_dim
    clean = answer_prob(lm, rows, targets)
    out = torch.zeros(len(rows), lm.spec.n_layers, n_heads)
    for layer in range(lm.spec.n_layers):
        target = mean_z[layer].to(lm.device)
        head_of_row = torch.arange(n_heads, device=lm.device).repeat_interleave(len(rows))

        def edit(z, target=target, head_of_row=head_of_row):
            z = z.clone()
            for h in range(n_heads):
                cols = slice(h * dim, (h + 1) * dim)
                z[head_of_row == h, -1, cols] = target[cols].to(z.dtype)
            return z

        p = answer_prob(lm, rows * n_heads, targets * n_heads, {layer: edit})
        out[:, layer] = (p.view(n_heads, len(rows)) - clean[None]).T.cpu()
    return out, clean.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--run", default="davidson_fv_all14_30", help="Davidson FV run under results/")
    parser.add_argument("--out", default="davidson_fv_lens_cie", help="output folder under results/")
    parser.add_argument("--tasks", nargs="+", default=None, help="default: the run's tasks")
    args = parser.parse_args()
    gpu.pin(args.gpu)

    import numpy as np
    import torch

    from davidson_fv import KINDS, function_vector
    from davidson_fv_diff import corrupted_rows, task_mean

    from lth.model import LM

    torch.set_grad_enabled(False)
    run = RESULTS / args.run
    plan = json.loads((run / "items.json").read_text())
    corrupted = json.loads((run / "uninformative.json").read_text())
    fv_saved = np.load(run / "fv.npz")
    heads = [(l, h) for l, h, _ in json.loads((run / "heads.json").read_text())["heads"]]
    tasks = args.tasks or list(plan)
    out = RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)
    per_prompt_dir = DATA / "estimates" / "davidson_lens_cie"
    per_prompt_dir.mkdir(parents=True, exist_ok=True)

    lm = LM.load()
    tok = lm.tokenizer
    spec = lm.spec
    n_layers, n_heads, dim = spec.n_layers, spec.n_heads, spec.head_dim
    norm, head = lm.model._final_norm, lm.model._lm_head
    gamma = norm.weight.float()
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)))
    w_u = head.weight.float()  # [vocab, d_model]
    transports = {"logithead": {l: lm.w_o[l].float().view(spec.d_model, n_heads, dim) for l in range(n_layers)},
                  "jhead": {l: lm.jz[l].float().view(spec.d_model, n_heads, dim) for l in range(n_layers)}}
    assert tuple(transports) == READOUTS

    captured = {}

    def grab(_module, inputs, _output):
        captured["x"] = inputs[0][:, -1].float()

    estimates, measured, prompts_log, fv_check, fd_check = {}, {}, {}, {}, {}
    for task in tasks:
        mean = task_mean(lm, task, plan)
        fv = function_vector(lm, heads, mean).cpu().numpy()
        saved = fv_saved[f"{task}/task"]
        fv_check[task] = float(np.abs(fv - saved).max())
        rows, meta = corrupted_rows(lm, task, plan, corrupted)
        targets = [m["answer_token"] for m in meta]

        # Forward passes: z and the final norm's input at the last token.
        z_rows, x_rows, next_tokens = [], [], []
        handle = norm.register_forward_hook(grab)
        try:
            for b in range(0, len(rows), 20):
                t, z = lm.next_tokens_batch(rows[b:b + 20])
                next_tokens += t
                z_rows.append(z.float())
                x_rows.append(captured["x"])
        finally:
            handle.remove()
        z_rows, x_rows = torch.cat(z_rows), torch.cat(x_rows)  # [P, L, d_source], [P, d_model]

        mean_dev = mean.to(lm.device)
        est = torch.zeros(len(rows), len(READOUTS), len(MEASURES), n_layers * n_heads)
        for p in range(len(rows)):
            x = x_rows[p]
            r = torch.sqrt(x.pow(2).mean() + eps)
            x_hat = x / r
            y = gamma * x_hat
            logits = w_u @ y  # the prompt's own logits, float32
            probs = logits.softmax(-1)
            u_a = w_u[targets[p]]
            u_bar = probs @ w_u
            g_a, g_bar = gamma * u_a, gamma * u_bar
            d = mean_dev - z_rows[p]  # [L, d_source]
            for ri, readout in enumerate(READOUTS):
                v = torch.cat([torch.einsum("mhd,hd->hm", transports[readout][l], d[l].view(n_heads, dim))
                               for l in range(n_layers)])  # [L * H, d_model]
                along = (v @ x_hat) / spec.d_model  # x_hat . v / d_model
                dlogit = (v @ g_a - along * (x_hat @ g_a)) / r
                dbar = (v @ g_bar - along * (x_hat @ g_bar)) / r
                est[p, ri, 0] = dlogit.cpu()
                est[p, ri, 1] = (dlogit - dbar).cpu()
                est[p, ri, 2] = (probs[targets[p]] * (dlogit - dbar)).cpu()
                if p == 0 and readout == "jhead":  # finite-difference check of the norm linearisation
                    i = int(dlogit.abs().argmax())
                    step = 1e-3 * float(x.norm() / v[i].norm())
                    fd = (w_u[targets[p]] @ norm(x + step * v[i]) - w_u[targets[p]] @ norm(x)) / step
                    fd_check[task] = dict(head=[i // n_heads, i % n_heads], estimate=float(dlogit[i]),
                                          finite_difference=float(fd))
            meta[p].update(p_answer=float(probs[targets[p]]), predicted_token=int(next_tokens[p]),
                           predicted=tok.decode([int(next_tokens[p])]),
                           correct=int(next_tokens[p]) == targets[p])
        np.savez_compressed(per_prompt_dir / f"{task}.npz", estimates=est.numpy(),
                            kinds=np.array([m["kind"] for m in meta]))
        kinds = np.array([m["kind"] for m in meta])
        for ri, readout in enumerate(READOUTS):
            for mi, measure in enumerate(MEASURES):
                estimates[f"{task}/{readout}/{measure}"] = np.stack(
                    [est[kinds == k, ri, mi].mean(0).view(n_layers, n_heads).numpy() for k in KINDS])

        # The CIE measured on the same prompts, per prompt; one call per kind (20 prompts), as
        # davidson_fv.py batches it.
        parts = [cie_rows(lm, [rows[i] for i in idx], [targets[i] for i in idx], mean)
                 for idx in (np.flatnonzero(kinds == k).tolist() for k in KINDS)]
        cie, clean = torch.cat([c for c, _ in parts]), torch.cat([c for _, c in parts])
        measured[task] = cie.numpy()
        for m, c in zip(meta, clean.tolist()):
            m["p_answer_model"] = c  # from the full forward pass (bf16), the CIE's baseline
        prompts_log[task] = meta
        print(f"{task}: {len(rows)} prompts, FV check {fv_check[task]:.1e}, mean p_a {np.mean(clean.numpy()):.3f}, "
              f"norm check {fd_check[task]['estimate']:.4f} vs {fd_check[task]['finite_difference']:.4f}", flush=True)

    np.savez_compressed(out / "estimates.npz", **estimates)
    np.savez_compressed(out / "measured.npz", **measured)
    (out / "prompts.json").write_text(json.dumps(prompts_log, indent=1, ensure_ascii=False))
    (out / "fv_check.json").write_text(json.dumps(dict(fv_max_abs_diff=fv_check, norm_finite_difference=fd_check),
                                                  indent=1))


if __name__ == "__main__":
    main()
