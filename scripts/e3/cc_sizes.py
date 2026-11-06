"""CC-pooled size curve: rank every head of a task's CC-pooled set by its pooled match rate
(share of the five rule wordings' correct prompts whose top-10 holds a CC list token; ties: mean
number of list tokens in the top 10, layer, head, as scripts/rank_heads.py) and keep the top N;
N_RANDOM random sets per size in RANDOM_SIZES with the top-N's heads per layer, drawn outside it."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from lth import READOUTS, simple
from lth.evaluation import tokenizer as get_tok
from lth.store import readings_dir
from lth.terms import expand_to_token_ids

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
SIZES, RANDOM_SIZES, N_RANDOM, RHO, N_HEADS, R, SEED = (1, 2, 5, 10, 20, 40), (10, 20), 2, 0.10, 32, READOUTS.index("jhead"), 20261106
tok = get_tok(); rng = np.random.default_rng(SEED)
lists = json.load(open("results/word_lists_union/word_lists.json"))["all"]
cc = {t: {tuple(x) for x in next(hs for _, n, hs in v["sets"] if n == "union_alllist_allid")}
      for t, v in json.load(open("results/word_lists_union/heads.json"))["simple"].items()}
out, info = {}, []
for t in simple.TASKS:
    top = np.concatenate([np.load(readings_dir("simple", t) / f"{w}.npz")["top"][:, R] for w in simple.WORDINGS["rule"]])
    ids = np.array(sorted(set(expand_to_token_ids(tok, lists[t]["union"]))))
    hit = np.isin(top, ids); rate, count = hit.any(-1).mean(0), hit.sum(-1).mean(0)
    passed = {(int(h) // N_HEADS, int(h) % N_HEADS) for h in np.flatnonzero(rate >= RHO)}
    assert passed == cc[t], (t, len(passed), len(cc[t]))  # same set as the CC-pooled run
    order = [int(h) for h in np.lexsort((np.arange(len(rate)), -count, -rate)) if rate[h] >= RHO]
    sets = []
    for n in SIZES:
        if n > len(order): continue
        T = {(h // N_HEADS, h % N_HEADS) for h in order[:n]}
        sets.append(["rule", f"cc_top{n}", sorted(map(list, T))])
        if n in RANDOM_SIZES:
            for i in range(N_RANDOM):
                pick = []
                for layer in sorted({l for l, _ in T}):
                    k = sum(1 for l, _ in T if l == layer)
                    pool = [(layer, x) for x in range(N_HEADS) if (layer, x) not in T]
                    pick += [pool[int(j)] for j in rng.choice(len(pool), k, replace=False)]
                sets.append(["rule", f"cc_top{n}_random{i}", sorted(map(list, pick))])
    out[t] = {"sets": sets}
    first = [(h // N_HEADS, h % N_HEADS) for h in order[:5]]
    info.append(dict(task=t, full=len(order), top5=" ".join(f"L{l}H{h}({rate[l*N_HEADS+h]:.2f})" for l, h in first),
                     rate_at_20=round(float(rate[order[19]]), 3), rate_at_40=round(float(rate[order[39]]), 3)))
(OUT / "heads.json").write_text(json.dumps({"simple": out}, indent=1))
import pandas as pd
pd.DataFrame(info).to_csv(OUT / "ranking.csv", index=False); print(pd.DataFrame(info).to_string(index=False))
