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


DECODE_CHUNK = 512  # positions unembedded at once


@torch.no_grad()
def decode(lm, ids: list[int], head: tuple[int, int], *, k: int = 10) -> dict[str, np.ndarray]:
    """Logit-head and J-head top-``k`` token ids of ``head`` at every position, ``[T, k]`` each.

    From the normal pass (:meth:`LM.states`), with the arithmetic of
    ``all_heads_transported`` restricted to one head: the head's columns of ``W_O`` (or of
    ``J^z``) times its slice of ``z``, in float32, through ``model.unembed``.
    """
    from src.heads import COMPUTE_DTYPE

    layer, h = head
    cols = lm.spec.head_slice(h)
    z = torch.stack([s[layer] for s in lm.states(ids, list(range(len(ids))))])[:, cols].to(COMPUTE_DTYPE)
    out = {}
    for readout in READOUTS:
        matrix = (lm.w_o[layer] if readout == "logithead" else lm.jz[layer]).to(COMPUTE_DTYPE)[:, cols]
        out[readout] = np.concatenate([
            lm.model.unembed(z[i:i + DECODE_CHUNK] @ matrix.T).float().topk(k, dim=-1).indices.cpu().numpy()
            for i in range(0, len(ids), DECODE_CHUNK)])
    return out


def top_decoded(decodes: list[dict[str, np.ndarray]], tokenizer, *, last_only: bool = True,
                n: int = 15) -> dict[str, pd.DataFrame]:
    """Per readout, the tokens most often in the head's top-``k``, over prompts (at the last
    token) or over every position but BOS. ``share``: of the prompts (positions) counted."""
    out = {}
    for readout in READOUTS:
        tops = [d[readout][-1:] if last_only else d[readout][1:] for d in decodes]
        ranks = pd.DataFrame([(int(t), r) for top in tops for row in top for r, t in enumerate(row)],
                             columns=["id", "rank"])
        frame = (ranks.groupby("id").agg(count=("rank", "size"), mean_rank=("rank", "mean")).reset_index()
                 .sort_values(["count", "mean_rank"], ascending=[False, True]).head(n))
        frame.insert(0, "token", [tokenizer.decode([i]) for i in frame["id"]])
        frame["share"] = frame["count"] / sum(len(top) for top in tops)
        out[readout] = frame.drop(columns="id").reset_index(drop=True)
    return out


def top_tokens(scans: list[Scan], head: tuple[int, int], *, side: str, n: int = 15,
               unique: bool = False) -> pd.DataFrame:
    """The (query or key) tokens with the largest non-BOS attention for ``head``, over ``scans``.

    ``side="key"``: tokens most attended to; ``side="query"``: tokens that attend most. Each
    row gives the other end of that strongest link. With ``unique``, a token string appears
    once, at its largest value.
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
    if not len(frame):
        return frame
    frame = frame.sort_values("attention", ascending=False)
    if unique:
        frame = frame.drop_duplicates(side)
    return frame.head(n).reset_index(drop=True)


def chips_html(frame: pd.DataFrame, side: str) -> str:
    """Tokens from :func:`top_tokens` as shaded chips; hovering one shows the other end."""
    import html

    other = "query" if side == "key" else "key"
    chips = [f'<span title="prompt {r["prompt"]}, @{r["position"]}; {other} {html.escape(repr(r[other]))}'
             f' @{r["other position"]}" style="display:inline-block;margin:2px;padding:1px 6px;border-radius:4px;'
             f'background:rgba(42,120,214,{min(r["attention"], 1):.3f});font-family:ui-monospace,monospace;'
             f'font-size:12px;white-space:pre">{html.escape(repr(r[side])[1:-1])}'
             f' <b>{r["attention"]:.2f}</b></span>' for r in frame.to_dict("records")]
    return '<div style="display:flex;flex-wrap:wrap;gap:2px">' + ("".join(chips) or "<i>none</i>") + "</div>"


def decoded_chips_html(frame: pd.DataFrame) -> str:
    """Tokens from :func:`top_decoded` as chips shaded by their share; hover for count and rank."""
    import html

    chips = [f'<span title="in the top {r["count"]} times ({r["share"]:.0%}); mean rank {r["mean_rank"]:.1f}" '
             f'style="display:inline-block;margin:2px;padding:1px 6px;border-radius:4px;'
             f'background:rgba(42,120,214,{min(r["share"], 1):.3f});font-family:ui-monospace,monospace;'
             f'font-size:12px;white-space:pre">{html.escape(repr(r["token"])[1:-1])}'
             f' <b>{r["share"]:.0%}</b></span>' for r in frame.to_dict("records")]
    return '<div style="display:flex;flex-wrap:wrap;gap:2px">' + ("".join(chips) or "<i>none</i>") + "</div>"


VIEWER_THRESHOLD = 0.002  # attention below this is left out of the page (BOS is always kept)


def viewer_html(scans: list[Scan], head: tuple[int, int], order: list[int], *, header: str = "",
                decodes: list[dict[str, np.ndarray]] | None = None, tokenizer=None) -> str:
    """A standalone page: the prompts in ``order``, hover a token to see its attention.

    Needs ``scan(..., [head], patterns=True)``. Each row starts at the prompt's strongest
    query -> non-BOS key link (shown by default); hovering a token shows what it attends to
    (or what attends to it), clicking pins it. Shades are relative to the largest non-BOS
    value shown, so BOS is reported in numbers only.
    """
    data = []
    for p in order:
        s = scans[p]
        a = s.patterns[head]
        value, key = s.query_max[head]
        q = int(value[1:].argmax()) + 1
        rows = [[[j, round(float(a[i, j]), 4)] for j in range(i + 1) if j == 0 or a[i, j] >= VIEWER_THRESHOLD]
                for i in range(len(s.ids))]
        entry = {"p": p, "tokens": s.tokens, "rows": rows, "q": q, "k": int(key[q]), "max": float(value[q])}
        if decodes is not None:
            entry["dec"] = [[" | ".join(repr(tokenizer.decode([int(x)])) for x in decodes[p][r][i, :5])
                             for r in READOUTS] for i in range(len(s.ids))]
        data.append(entry)
    payload = json.dumps(data).replace("</", "<\\/")
    return _VIEWER.replace("__DATA__", payload).replace("__HEADER__", header).replace("__TITLE__", head_label(*head))


_VIEWER = r"""<!doctype html><html><head><meta charset="utf-8"><title>HeadVis __TITLE__</title>
<style>
:root{--bg:#fcfcfb;--ink:#1f1f1e;--muted:#6b6a65;--line:#e1e0d9;--q:#eb6834}
@media (prefers-color-scheme:dark){:root{--bg:#1f1f1e;--ink:#e8e8e3;--muted:#a8a7a2;--line:#383835}}
body{margin:0;padding:8px 10px;background:var(--bg);color:var(--ink);font:13px system-ui,sans-serif}
.bar{display:flex;flex-wrap:wrap;gap:14px;align-items:center;position:sticky;top:0;background:var(--bg);
     padding:4px 0 8px;border-bottom:1px solid var(--line);z-index:1}
.hint{color:var(--muted)}
.row{padding:6px 0;border-bottom:1px solid var(--line)}
.meta{color:var(--muted);font-size:12px;margin-bottom:2px}
.toks{font-family:ui-monospace,monospace;font-size:13px;line-height:1.75;max-height:260px;overflow:auto}
.t{white-space:pre-wrap;border-radius:2px;cursor:pointer;padding:0 1px}
.t.on{outline:2px solid var(--q)}
.status{font-family:ui-monospace,monospace;font-size:12px;color:var(--muted);min-height:1.3em}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid var(--line);padding:2px 6px;text-align:left}
</style></head><body>
__HEADER__
<div class="bar">
  <span>Hover a token to see</span>
  <label><input type="radio" name="mode" value="q" checked> what it attends to</label>
  <label><input type="radio" name="mode" value="k"> what attends to it</label>
  <label><input type="checkbox" id="full"> whole prompts</label>
  <span class="hint">click pins a token; shade = share of the largest non-BOS value</span>
</div>
<div id="rows"></div>
<script>
const D = __DATA__;
const show = t => t.replace(/\n/g, "⏎");
D.forEach(d => {
  d.out = d.rows.map(r => new Map(r));
  d.in = d.tokens.map(() => new Map());
  d.rows.forEach((r, i) => r.forEach(([j, v]) => d.in[j].set(i, v)));
});
function mode() { return document.querySelector('input[name=mode]:checked').value; }
function paint(d, el, t, m) {
  const vals = m === "q" ? d.out[t] : d.in[t];
  let top = 0; vals.forEach((v, j) => { if (j !== 0 && v > top) top = v; });
  el.querySelectorAll(".t").forEach(s => {
    const j = +s.dataset.i, v = vals.get(j) || 0;
    s.style.background = j === 0 || !top ? "" : `rgba(42,120,214,${(v / top).toFixed(3)})`;
    s.classList.toggle("on", j === t);
    s.title = `@${j} ${JSON.stringify(d.tokens[j])}  ${v.toFixed(3)}`;
  });
  const best = [...vals].filter(([j]) => j !== 0).sort((a, b) => b[1] - a[1]).slice(0, 5)
    .map(([j, v]) => `${JSON.stringify(show(d.tokens[j]))} @${j} ${v.toFixed(3)}`).join(" · ");
  const bos = m === "q" ? `BOS ${(vals.get(0) || 0).toFixed(3)} · ` : "";
  el.querySelector(".status").textContent = (m === "q"
    ? `query ${JSON.stringify(show(d.tokens[t]))} @${t} → ` : `key ${JSON.stringify(show(d.tokens[t]))} @${t} ← `)
    + bos + (best || "nothing above threshold");
  if (d.dec) el.querySelector(".dec").textContent =
    `writes at @${t}:  Logit-head ${d.dec[t][0]}   ·   J-head ${d.dec[t][1]}`;
}
function render() {
  const full = document.getElementById("full").checked, box = document.getElementById("rows");
  box.innerHTML = "";
  D.forEach((d, rank) => {
    const n = d.tokens.length;
    const lo = full ? 0 : Math.max(1, Math.min(d.k, d.q) - 12), hi = full ? n : Math.min(n, d.q + 6);
    const el = document.createElement("div"); el.className = "row";
    el.innerHTML = `<div class="meta"><b>#${rank + 1}</b> · prompt ${d.p} · ${n} tokens · strongest link
      ${d.max.toFixed(3)}: ${JSON.stringify(show(d.tokens[d.q]))} @${d.q} → ${JSON.stringify(show(d.tokens[d.k]))} @${d.k}</div>
      <div class="toks"></div><div class="status"></div><div class="status dec"></div>`;
    const toks = el.querySelector(".toks");
    if (lo > 0) toks.append("… ");
    for (let i = lo; i < hi; i++) {
      const s = document.createElement("span"); s.className = "t"; s.dataset.i = i;
      s.textContent = full ? d.tokens[i] : show(d.tokens[i]);
      if (full && d.tokens[i].includes("\n")) s.textContent = show(d.tokens[i]) + "\n";
      toks.append(s);
    }
    if (hi < n) toks.append(" …");
    let pinned = null;
    const rest = () => pinned === null ? paint(d, el, d.q, "q") : paint(d, el, pinned, mode());
    toks.addEventListener("mouseover", e => { if (e.target.dataset.i) paint(d, el, +e.target.dataset.i, mode()); });
    toks.addEventListener("mouseleave", rest);
    toks.addEventListener("click", e => {
      if (!e.target.dataset.i) return;
      const i = +e.target.dataset.i; pinned = pinned === i ? null : i; rest();
    });
    box.append(el); rest();
  });
}
document.getElementById("full").addEventListener("change", render);
document.querySelectorAll("input[name=mode]").forEach(r => r.addEventListener("change", render));
render();
</script></body></html>"""
