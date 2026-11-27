"""AttnLRP relevance of the last token's attention, as Pham et al. (2026) score heads.

Pham et al. ("Fast & Faithful Function Vectors", arXiv 2606.05079) compute AttnLRP (Achtibat et al., 2024) with
lxt 2.1's "efficient" rules: the forward pass is unchanged and the backward pass is modified so that gradient
times input is the relevance. lxt 2.1 does not import under transformers 5 (it imports a helper removed there),
and its attention patch replaces ``ALL_ATTENTION_FUNCTIONS``, on which transformers 5.15's Llama calls
``get_interface``. :func:`attnlrp` applies the same rules (``lxt/efficient/patches.py``, ``rules.py``) by hooks and
per-instance forwards instead:

- attention: the gradient into the query and key projections is divided by 4 and into the value projection by 2
  (lxt divides query, key and value as they enter the attention function; RoPE is linear, so dividing before it
  is the same);
- RMSNorm: no gradient through the factor 1 / rms (``rsqrt(...).detach()``);
- MLP: the gradient of SiLU replaced by SiLU(x) / x (lxt's identity rule), and the gradient of gate * up divided
  by 2;
- softmax, the residual additions and the linear layers keep their gradient.

:func:`relevance_row` runs one prompt forward and backward from the logit of a target token and returns, for every
head, the relevance of each attention weight of the last position, ``R[j] = A[n, j] * dlogit/dA[n, j]``. Summed over
``j`` it is the head's relevance at ``o_proj``'s input, ``z . dlogit/dz``, also returned for checking.
"""

from __future__ import annotations

import contextlib
import types

import numpy as np
import torch


class _Scale(torch.autograd.Function):
    """Identity forward; the gradient divided by ``factor`` (lxt's ``divide_gradient``)."""

    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad / ctx.factor, None


class _IdentityRule(torch.autograd.Function):
    """``fn(x)`` forward; backward ``fn(x) / x`` times the gradient (lxt's ``identity_rule_implicit``)."""

    @staticmethod
    def forward(ctx, fn, x):
        y = fn(x)
        ctx.save_for_backward(y / (x + 1e-10))
        return y

    @staticmethod
    def backward(ctx, grad):
        return None, ctx.saved_tensors[0] * grad


def _rms_norm_forward(self, hidden_states):
    """transformers 5.15's ``LlamaRMSNorm.forward`` with the 1 / rms factor detached."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon).detach()
    return self.weight * hidden_states.to(input_dtype)


def _mlp_forward(self, x):
    """transformers 5.15's ``LlamaMLP.forward`` with the identity rule on the activation and gate * up halved."""
    gate = _IdentityRule.apply(self.act_fn, self.gate_proj(x))
    return self.down_proj(_Scale.apply(gate * self.up_proj(x), 2.0))


@contextlib.contextmanager
def attnlrp(hf_model):
    """AttnLRP's backward rules on a Llama model for the duration of the block; the forward pass is unchanged."""
    from transformers.models.llama import modeling_llama as ml

    handles, patched = [], []
    for module in hf_model.modules():
        new = {ml.LlamaRMSNorm: _rms_norm_forward, ml.LlamaMLP: _mlp_forward}.get(type(module))
        if new is not None:
            patched.append((module, module.__dict__.get("forward")))
            module.forward = types.MethodType(new, module)
        elif isinstance(module, ml.LlamaAttention):
            for proj, factor in ((module.q_proj, 4.0), (module.k_proj, 4.0), (module.v_proj, 2.0)):
                handles.append(proj.register_forward_hook(
                    lambda _m, _inp, out, factor=factor: _Scale.apply(out, factor)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for module, before in patched:
            del module.forward
            if before is not None:
                module.forward = before


def relevance_row(lm, ids: list[int], target: int, *, lrp: bool = True, extras: bool = False) -> dict:
    """One forward and backward pass of ``ids`` (eager attention) from the logit of ``target`` at the last token.

    Returns ``rel`` [T, n_layers * n_heads] (float32): ``A[n, j] * dlogit/dA[n, j]`` of the last position ``n``,
    under AttnLRP's rules (plain gradients with ``lrp=False``); ``att`` [T, n_layers * n_heads] (float16, as
    ``scripts/rebuild_attention.py`` stores it); ``head_rel`` [n_layers * n_heads]: ``z . dlogit/dz`` at the last
    token, ``z`` being ``o_proj``'s input; the target's ``logit`` and ``rank`` (1 = top) and the ``greedy`` token.
    With ``extras``, also the full last-token ``logits`` and the gradient at the input embeddings, for checks.
    """
    from lth.headvis import _eager

    hf = lm.model.hf_model
    n_layers, n_heads, d_head = lm.spec.n_layers, lm.spec.n_heads, lm.spec.head_dim
    att, zin, handles = {}, {}, []

    def keep_att(layer):
        def fn(_module, _args, output):
            output[1].retain_grad()
            att[layer] = output[1]
        return fn

    def keep_z(layer):
        def fn(_module, args):
            args[0].retain_grad()
            zin[layer] = args[0]
        return fn

    for layer in range(n_layers):
        block = lm.model.layers[layer].self_attn
        handles.append(block.register_forward_hook(keep_att(layer)))
        handles.append(block.o_proj.register_forward_pre_hook(keep_z(layer)))
    rules = attnlrp(hf) if lrp else contextlib.nullcontext()
    try:
        with _eager(hf), rules, torch.enable_grad():
            emb = hf.get_input_embeddings()(torch.tensor([ids], device=lm.device)).detach().requires_grad_(True)
            logits = hf(inputs_embeds=emb, use_cache=False, logits_to_keep=1).logits[0, -1]
            logits[target].backward()
    finally:
        for handle in handles:
            handle.remove()
    with torch.no_grad():
        rel = torch.stack([(att[l].float() * att[l].grad.float())[0, :, -1] for l in range(n_layers)])  # [L, H, T]
        rows = torch.stack([att[l][0, :, -1].to(torch.float16) for l in range(n_layers)])
        head_rel = torch.stack([(zin[l].float() * zin[l].grad.float())[0, -1].view(n_heads, d_head).sum(-1)
                                for l in range(n_layers)])
        last = logits.detach().float()
        out = {"rel": rel.reshape(n_layers * n_heads, -1).T.cpu().numpy(),
               "att": rows.reshape(n_layers * n_heads, -1).T.cpu().numpy(),
               "head_rel": head_rel.reshape(-1).cpu().numpy(),
               "logit": float(last[target]), "rank": int((last > last[target]).sum()) + 1,
               "greedy": int(last.argmax())}
        if extras:
            out["logits"] = last.cpu()
            out["input_grad"] = emb.grad[0].float().cpu()
    return out


def clamped(rows: np.ndarray) -> np.ndarray:
    """Pham et al.'s per-key relevance: ``max(R, 0)``."""
    return np.maximum(rows, 0.0)
