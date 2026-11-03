"""HeadVis on our own prompts: what attention heads attend to, and what they write.

Neuronpedia's HeadVis (e.g. neuronpedia.org/llama3.1-8b-it/head/25/7) gives each head a few
attention scores and the corpus sequences it attends most strongly on. Here the same kind
of data comes from live passes over prompts we choose, for every head at once, plus the
full pattern of a few chosen heads. Exploration only: no number here is a result.

Attention weights exist only under eager attention, so :func:`scan` switches the model to
eager for its pass and back afterwards; the readouts (:func:`readouts`) use the normal
pass, as the runs do. Every score skips the first query (BOS can attend only to itself),
and "max" scores skip the BOS key, where most heads park most of their attention.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from lth import RESULTS, READOUTS

# Per-head scores on a prompt, in the order of the tables. ``induction`` is undefined
# (NaN) on a prompt with no repeated token.
SCORES = {
    "bos": "mean attention on BOS",
    "prev": "mean attention on the previous token (queries from the 3rd token on)",
    "self": "mean attention on the query token itself",
    "induction": "mean attention on the token after an earlier copy of the query token",
    "entropy": "mean entropy of a query's attention, over log(number of keys)",
    "distance": "mean distance from a query to its most-attended non-BOS key",
    "max": "largest attention any query pays to a non-BOS key",
}
INDUCTION_LENGTH = 50  # random tokens per half of the repeated sequence
INDUCTION_SEQUENCES = 8


@dataclass
class Scan:
    """One prompt's pass: per-head scores and the chosen heads' attention."""

    ids: list[int]
    tokens: list[str]
    scores: pd.DataFrame  # layer, head, one column per SCORES key
    # (layer, head) -> per query: largest attention to a non-BOS key, and that key;
    # per key: largest attention from a later query, and that query.
    query_max: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
    key_max: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
    patterns: dict[tuple[int, int], np.ndarray]  # [query, key], with ``patterns=True`` only


def parse_heads(text: str) -> list[tuple[int, int]]:
    """``"L25H7, L13H22"`` (or ``"25.7 13.22"``) -> ``[(25, 7), (13, 22)]``."""
    import re

    return [(int(l), int(h)) for l, h in re.findall(r"L?(\d+)\s*[Hh.:/]\s*(\d+)", text)]


def head_label(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def head_sets() -> dict[str, list[tuple[int, int]]]:
    """The identified sets in ``results/heads.json``, keyed ``family/task/style/readout``."""
    data = json.loads((RESULTS / "heads.json").read_text())
    return {f"{family}/{task}/{style}/{readout}": [(l, h) for l, h, _ in sets[readout]["heads"]]
            for family, tasks in data.items() for task, styles in tasks.items()
            for style, sets in styles.items() for readout in READOUTS if readout in sets}


def token_strings(tokenizer, ids: list[int]) -> list[str]:
    return [tokenizer.decode([i]) for i in ids]


@contextlib.contextmanager
def _eager(hf):
    before = hf.config._attn_implementation
    hf.set_attn_implementation("eager")
    try:
        yield
    finally:
        hf.set_attn_implementation(before)


@torch.no_grad()
def scan(lm, ids: list[int], heads: list[tuple[int, int]] = (), *, patterns: bool = False) -> Scan:
    """Scores of every head on one prompt, and the attention of ``heads``.

    A hook on each layer's attention reduces its weights ``[1, n_heads, T, T]`` to the
    scores at once and drops them, so a long prompt never holds more than one layer's.
    """
    n = len(ids)
    if n < 3:
        raise ValueError("a prompt needs at least 3 tokens (BOS included)")
    device = lm.device
    tok = torch.tensor(ids, device=device)
    q = torch.arange(n, device=device)
    # Induction keys: j is an induction key of query i when the token before j equals
    # token i (an earlier copy) -- j - 1 < i, so j <= i.
    induct = torch.zeros(n, n, dtype=torch.bool, device=device)
    induct[:, 1:] = tok[:, None] == tok[None, :-1]
    induct &= q[None, :] <= q[:, None]
    has_copy = induct[1:].any(-1)
    wanted = {}
    for layer, head in heads:
        wanted.setdefault(layer, []).append(head)
    rows, query_max, key_max, kept = [], {}, {}, {}

    def hook(layer):
        def fn(module, args, output):
            a = output[1][0].float()  # [n_heads, T, T]
            body = a[:, 1:]  # queries 1..T-1, which see 2..T keys
            n_keys = torch.arange(2, n + 1, device=device).float()
            ent = -(body * body.clamp_min(1e-12).log()).sum(-1) / n_keys.log()
            no_bos = a[:, :, 1:]  # [n_heads, T, T-1]: keys 1..T-1
            best, where = no_bos[:, 1:].max(-1)
            ind = (body * induct[1:]).sum(-1)
            scores = {
                "bos": body[..., 0].mean(-1),
                "prev": a[:, q[2:], q[2:] - 1].mean(-1),
                "self": a.diagonal(dim1=1, dim2=2)[:, 1:].mean(-1),
                "induction": ind[:, has_copy].mean(-1) if has_copy.any() else ind[:, 0] * float("nan"),
                "entropy": ent.mean(-1),
                "distance": (q[1:][None] - (where + 1)).float().mean(-1),
                "max": best.amax(-1),
            }
            table = torch.stack(list(scores.values()), dim=1).cpu().numpy()
            for h in range(a.shape[0]):
                rows.append((layer, h, *table[h]))
            for h in wanted.get(layer, []):
                qm, qa = no_bos[h].max(-1)  # query -> its strongest non-BOS key
                km, ka = no_bos[h, 1:].max(0)  # non-BOS key -> its strongest query
                query_max[(layer, h)] = (qm.cpu().numpy(), qa.cpu().numpy() + 1)
                key_max[(layer, h)] = (np.concatenate([[np.nan], km.cpu().numpy()]),
                                       np.concatenate([[0], ka.cpu().numpy() + 1]))
                if patterns:
                    kept[(layer, h)] = a[h].cpu().numpy()
            return output[0], None
        return fn

    handles = [lm.model.layers[l].self_attn.register_forward_hook(hook(l)) for l in range(lm.spec.n_layers)]
    try:
        with _eager(lm.model.hf_model):
            lm.model.hf_model(input_ids=tok[None], use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    frame = (pd.DataFrame(rows, columns=["layer", "head", *SCORES]).astype({k: float for k in SCORES})
             .sort_values(["layer", "head"], ignore_index=True))
    return Scan(ids, token_strings(lm.tokenizer, ids), frame, query_max, key_max, kept)


def induction_scores(lm, *, seed: int = 0) -> pd.DataFrame:
    """Every head's induction score on repeated random tokens, as Neuronpedia's.

    BOS, then ``INDUCTION_LENGTH`` distinct random tokens, then the same tokens again; the
    score is the mean attention from the second copy of a token to the token after its
    first copy, averaged over ``INDUCTION_SEQUENCES`` sequences.
    """
    rng = np.random.default_rng(seed)
    bos = lm.tokenizer.bos_token_id
    frames = []
    for _ in range(INDUCTION_SEQUENCES):
        half = rng.choice(np.arange(1000, 30000), INDUCTION_LENGTH, replace=False).tolist()
        frames.append(scan(lm, [bos, *half, *half]).scores[["layer", "head", "induction"]])
    return pd.concat(frames).groupby(["layer", "head"], as_index=False).mean()


@torch.no_grad()
def readouts(lm, ids: list[int], position: int, heads: list[tuple[int, int]], *, k: int = 10,
             target: int | None = None) -> pd.DataFrame:
    """Logit-head and J-head top-``k`` tokens of ``heads`` at ``position``, from the normal pass.

    With ``target``, also that token's rank (0 = top) under each readout.
    """
    from src.readout import all_heads_transported

    z = lm.states(ids, [position])[0]
    rows = []
    for layer in sorted({l for l, _ in heads}):
        for readout in READOUTS:
            vectors = all_heads_transported(lm.jz[layer], z[layer], lm.spec, arm=readout, w_o=lm.w_o[layer])
            logits = lm.model.unembed(vectors).float()
            top = logits.topk(k, dim=-1).indices.tolist()
            for _, head in [(l, h) for l, h in heads if l == layer]:
                row = {"head": head_label(layer, head), "readout": readout,
                       "top tokens": " | ".join(repr(lm.tokenizer.decode([t])) for t in top[head])}
                if target is not None:
                    row["target rank"] = int((logits[head] > logits[head, target]).sum())
                rows.append(row)
    return pd.DataFrame(rows)


def top_tokens(scans: list[Scan], head: tuple[int, int], *, side: str, n: int = 15) -> pd.DataFrame:
    """The (query or key) tokens with the largest non-BOS attention for ``head``, over ``scans``.

    ``side="key"``: tokens most attended to; ``side="query"``: tokens that attend most. Each
    row gives the other end of that strongest link.
    """
    rows = []
    for p, s in enumerate(scans):
        if head not in s.query_max:
            continue
        value, other = (s.key_max if side == "key" else s.query_max)[head]
        for pos in range(1, len(s.ids)):
            if np.isnan(value[pos]):
                continue
            rows.append({"prompt": p, side: s.tokens[pos], "position": pos, "attention": float(value[pos]),
                         "query" if side == "key" else "key": s.tokens[int(other[pos])],
                         "other position": int(other[pos])})
    frame = pd.DataFrame(rows)
    return frame.nlargest(n, "attention").reset_index(drop=True) if len(frame) else frame


def token_html(tokens: list[str], weights: np.ndarray, *, marked: int | None = None) -> str:
    """The prompt as text, each token shaded by ``weights`` (scaled to their max, BOS left out).

    Hovering a token shows its position and weight; ``marked`` (the query) is outlined.
    """
    import html

    top = np.nanmax(weights[1:]) if len(weights) > 1 and np.nanmax(weights[1:]) > 0 else 1.0
    spans = []
    for pos, (text, w) in enumerate(zip(tokens, weights)):
        shade = 0.0 if pos == 0 or np.isnan(w) else float(w) / top
        shown = html.escape(text).replace("\n", "⏎<br>") or "∅"
        outline = "outline:2px solid #eb6834;" if pos == marked else ""
        spans.append(f'<span title="{pos}: {html.escape(repr(text))}  {0 if np.isnan(w) else w:.3f}" '
                     f'style="background:rgba(42,120,214,{shade:.3f});{outline}border-radius:2px;'
                     f'white-space:pre-wrap">{shown}</span>')
    return ('<div style="font-family:ui-monospace,monospace;font-size:13px;line-height:1.9;'
            'max-height:420px;overflow:auto">' + "".join(spans) + "</div>")
