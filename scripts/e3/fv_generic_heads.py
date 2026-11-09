"""Which Davidson FV heads decode generic content, and what else sets them apart (no GPU).

Per head (all 1,024, so that the FV heads can be set against heads of the same layers), J-head
and Logit-head readings at the last token of the correct inst1-4 prompts
(``data/readings/simple``):
- ``generic``: how alike its decoded tokens are across the 14 tasks: per task a vector of
  token counts over its top 10s, then the cosine of every pair of tasks, averaged (1 = the same
  tokens on every task);
- ``same_as_diff``: the share of its 10 most frequent raw tokens that are also among the 10 most
  frequent tokens of its instruction difference (``data/readings/davidson_diff``, mean task z
  minus z on each corrupted prompt), averaged over tasks (low = what it decodes does not come
  from the instruction);
- CIE (``results/davidson_fv_all14_30/cie.npz``, mean over the three kinds): mean over tasks,
  ``breadth`` = tasks where the head is among the task's 50 largest CIE, ``top_share`` = share of
  its positive CIE that comes from its best task;
- attention at the last token (``data/readings/attention_rule``, inst1-4 prompts): mass on task
  words of the instruction, on the query's input, on the rest (other instruction tokens and the
  template's ``Q:`` / ``A:`` / newline) and on BOS;
- with ``data/estimates/fv_task_means.npz`` (``scripts/e3/fv_task_means.py``): ``out_cos``, the
  mean cosine across task pairs of its J-head-transported task-mean output (1 = the same output
  vector on every task), and ``instr_change``, 1 - cosine of the task mean with the corrupted
  prompts' mean (how much the instruction changes its output).

Writes ``results/davidson_fv_diff/generic_heads.json`` (every head) and ``generic_heads.md`` (the
FV heads by ``generic``, their pooled top tokens and the measures, next to the FV heads'
and all heads' medians in layers 13-31).

    .venv/bin/python scripts/e3/fv_generic_heads.py
"""
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lth import DATA, READOUTS, RESULTS, simple  # noqa: E402
from lth.evaluation import tokenizer as get_tok  # noqa: E402
from lth.store import load, readings_dir  # noqa: E402

N_ALL, N_H = 1024, 32
WORDINGS = ("inst1", "inst2", "inst3", "inst4")
OUT = RESULTS / "davidson_fv_diff"
tok = get_tok()
fv = [l * N_H + h for l, h, _ in json.loads((RESULTS / "davidson_fv_all14_30/heads.json").read_text())["heads"]]
tasks = list(simple.TASKS)


def counts(top):  # top [P, 10] -> Counter of tokens (once per prompt)
    c = Counter()
    for row in top:
        c.update(set(row.tolist()))
    return c


def cos(a: Counter, b: Counter) -> float:
    num = sum(v * b.get(k, 0) for k, v in a.items())
    den = np.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return num / den if den else 0.0


raw = {t: np.concatenate([load(readings_dir("simple", t), w)["top"] for w in WORDINGS]) for t in tasks}  # P, 2, 1024, 10
diff = {t: np.load(DATA / "readings/davidson_diff" / f"{t}.npz")["top"] for t in tasks}
res = {r: {k: np.zeros(N_ALL) for k in ("generic", "same_as_diff")} for r in READOUTS}
tops = {r: {} for r in READOUTS}
for ri, r in enumerate(READOUTS):
    for s in range(N_ALL):
        c = {t: counts(raw[t][:, ri, s]) for t in tasks}
        res[r]["generic"][s] = np.mean([cos(c[a], c[b]) for a, b in combinations(tasks, 2)])
        same = []
        for t in tasks:
            a = {k for k, _ in c[t].most_common(10)}
            b = {k for k, _ in counts(diff[t][:, ri, s]).most_common(10)}
            same.append(len(a & b) / 10)
        res[r]["same_as_diff"][s] = np.mean(same)
        if s in fv:
            pooled = Counter()
            for t in tasks:
                for k, v in c[t].items():
                    pooled[k] += v / len(raw[t]) / len(tasks)
            tops[r][s] = [[tok.decode([k]), round(v, 2)] for k, v in pooled.most_common(8)]
    print(r, "readings done", flush=True)

cie = np.load(RESULTS / "davidson_fv_all14_30/cie.npz")
per_task = np.stack([np.mean([cie[f"{t}/{k}"] for k in ("equiprobable", "wikitext", "other_task")], 0).ravel() for t in tasks])
top50 = np.zeros_like(per_task, dtype=bool)
for i in range(len(tasks)):
    top50[i, np.argsort(-per_task[i])[:50]] = True
pos = np.clip(per_task, 0, None)
measures = dict(cie_mean=per_task.mean(0), breadth=top50.sum(0).astype(float),
                top_share=np.where(pos.sum(0) > 0, pos.max(0) / np.maximum(pos.sum(0), 1e-12), np.nan))

masses = []
for t in tasks:
    a = np.load(DATA / "readings/attention_rule" / f"{t}.npy").astype(np.float32)  # P, 1024, 4 (inst0..inst4)
    n = [len(load(readings_dir("simple", t), w)["id_items"]) for w in simple.WORDINGS["rule"]]
    assert sum(n) == len(a), (t, n, len(a))
    masses.append(a[n[0]:].mean(0))  # inst1-4
masses = np.mean(masses, 0)
for i, name in enumerate(("att_taskword", "att_input", "att_rest", "att_bos")):
    measures[name] = masses[:, i]

means_file = DATA / "estimates/fv_task_means.npz"
if means_file.exists():
    import torch
    import lth.model  # noqa: F401  (puts head_level_jlens's src on the path)
    from lth import LENS_KEY
    from src.config import LENSES, jz_path
    from src.model import load_jz

    m = np.load(means_file)
    spec = LENSES[LENS_KEY]  # the lens is loaded as LM.load does, on the CPU
    jz = load_jz(spec, jz_path(spec, n_prompts=spec.jz_n_prompts)).jacobians
    out_cos, change = np.zeros(N_ALL), np.zeros(N_ALL)
    for l in range(spec.n_layers):
        J = jz[l].float().view(spec.d_model, spec.n_heads, spec.head_dim)
        v = {t: torch.einsum("mhd,hd->hm", J, torch.tensor(m[f"{t}/task"][l]).view(spec.n_heads, -1)) for t in tasks}
        vc = {t: torch.einsum("mhd,hd->hm", J, torch.tensor(m[f"{t}/corrupt"][l]).view(spec.n_heads, -1)) for t in tasks}
        for h in range(spec.n_heads):
            out_cos[l * N_H + h] = np.mean([float(torch.cosine_similarity(v[a][h], v[b][h], 0)) for a, b in combinations(tasks, 2)])
            change[l * N_H + h] = np.mean([1 - float(torch.cosine_similarity(v[t][h], vc[t][h], 0)) for t in tasks])
    measures["out_cos"], measures["instr_change"] = out_cos, change

table = {f"L{s // N_H}H{s % N_H}": {**{f"{r}_{k}": float(res[r][k][s]) for r in READOUTS for k in res[r]},
                                     **{k: float(v[s]) for k, v in measures.items()}, "layer": s // N_H,
                                     "fv_rank": fv.index(s) + 1 if s in fv else None}
         for s in range(N_ALL)}
(OUT / "generic_heads.json").write_text(json.dumps(dict(heads=table, fv_tops={r: {f"L{s // N_H}H{s % N_H}": v for s, v in tops[r].items()} for r in READOUTS}), indent=1))

cols = ["jhead_generic", "logithead_generic", "jhead_same_as_diff", "breadth", "top_share", "cie_mean",
        "att_taskword", "att_input", "att_rest", "att_bos"] + [k for k in ("out_cos", "instr_change") if k in measures]
order = sorted(fv, key=lambda s: -res["jhead"]["generic"][s])
lines = ["# Davidson FV heads: generic vs task-specific decoding", "",
         "FV heads ordered by `jhead_generic` (mean cosine of the head's J-head token counts across the 14 tasks). "
         "Medians: FV heads; all heads of layers 13-31.", "",
         "| head (FV rank) | J-head pooled tokens | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
for s in order:
    name = f"L{s // N_H}H{s % N_H}"
    toks = ", ".join(f"{t!r}" for t, _ in tops["jhead"][s][:5]).replace("|", "/")
    lines.append(f"| {name} ({fv.index(s) + 1}) | {toks} | " + " | ".join(f"{table[name][c]:.2f}" for c in cols) + " |")
band = [s for s in range(13 * N_H, N_ALL)]
for label, group in (("median, FV heads", fv), ("median, all heads L13-31", band)):
    lines.append(f"| **{label}** | | " + " | ".join(f"{np.nanmedian([table[f'L{s // N_H}H{s % N_H}'][c] for s in group]):.2f}" for c in cols) + " |")
(OUT / "generic_heads.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
