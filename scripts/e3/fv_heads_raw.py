"""What the Davidson FV heads decode: top-10 tokens at the last prompt token, both readouts.

Readings: data/readings/simple/<task>/inst1..inst4 (the instruction prompts the FV uses),
P+ prompts (first 100 correct per template), pooled over the four templates per task.
Per head, readout and task: answer share, input share, token shares (share of prompts whose
top-10 holds the token). Pooled over tasks: mean of the per-task shares.

    .venv/bin/python scripts/e3/fv_heads_raw.py results/davidson_fv_diff/raw_readings.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lth import READOUTS, simple  # noqa: E402
from lth.store import load, readings_dir  # noqa: E402
from lth.evaluation import tokenizer as get_tok  # noqa: E402

OUT = Path(sys.argv[1])
SETS = {"fv30_14": ROOT / "results/davidson_fv_all14_30/heads.json",
        "fv20_4": ROOT / "results/davidson_fv/heads.json"}
WORDINGS = ("inst1", "inst2", "inst3", "inst4")

tok = get_tok()
cache = {}


def dec(t):
    if t not in cache:
        cache[t] = tok.decode([int(t)])
    return cache[t]


def first(text):
    return tok(text, add_special_tokens=False).input_ids[0]


sets = {k: [(l, h) for l, h, *_ in json.loads(p.read_text())["heads"]] for k, p in SETS.items()}
heads = list(dict.fromkeys(sets["fv30_14"] + sets["fv20_4"]))
sites = [l * 32 + h for l, h in heads]
tasks = json.loads(SETS["fv30_14"].read_text())["tasks"]

res = {"sets": {k: [f"L{l}H{h}" for l, h in v] for k, v in sets.items()}, "tasks": {}}
for task in tasks:
    data = simple.dataset(task)
    tops, items = [], []
    for w in WORDINGS:
        run = load(readings_dir("simple", task), w)
        tops.append(run["top"][:, :, sites])
        items += [int(i) for i in run["id_items"]]
    top = np.concatenate(tops)  # P, 2, n_sites, 10
    ans = np.array([first(" " + data[i]["output"]) for i in items])
    inp = np.array([first(" " + data[i]["input"]) for i in items])
    per = {}
    for j, (l, h) in enumerate(heads):
        for r, readout in enumerate(READOUTS):
            t = top[:, r, j]
            c = Counter()
            for row in t:
                c.update(set(row.tolist()))
            per[f"L{l}H{h}/{readout}"] = {
                "answer": float((t == ans[:, None]).any(-1).mean()),
                "input": float((t == inp[:, None]).any(-1).mean()),
                "tokens": [[dec(k), round(v / len(t), 3)] for k, v in c.most_common(15)]}
    res["tasks"][task] = {"n_prompts": len(items), "heads": per}
    print(task, len(items), flush=True)

OUT.write_text(json.dumps(res, ensure_ascii=False))

# text dump
lines = []
for l, h in heads:
    tag = "/".join(k for k, v in sets.items() if (l, h) in v)
    lines.append(f"===== L{l}H{h} [{tag}] rank30={sets['fv30_14'].index((l,h))+1 if (l,h) in sets['fv30_14'] else '-'}")
    for readout in READOUTS:
        key = f"L{l}H{h}/{readout}"
        pooled = defaultdict(float)
        a = np.mean([res["tasks"][t]["heads"][key]["answer"] for t in tasks])
        i = np.mean([res["tasks"][t]["heads"][key]["input"] for t in tasks])
        for t in tasks:
            for s, v in res["tasks"][t]["heads"][key]["tokens"]:
                pooled[s] += v / len(tasks)
        best = sorted(pooled.items(), key=lambda kv: -kv[1])[:12]
        lines.append(f"  {readout:9s} answer {a:.2f} input {i:.2f} | pooled: " +
                     ", ".join(f"{s!r} {v:.2f}" for s, v in best))
        for t in tasks:
            d = res["tasks"][t]["heads"][key]
            lines.append(f"     {t:17s} a{d['answer']:.2f} i{d['input']:.2f} " +
                         ", ".join(f"{s!r} {v:.2f}" for s, v in d["tokens"][:6]))
OUT.with_suffix(".txt").write_text("\n".join(lines))
