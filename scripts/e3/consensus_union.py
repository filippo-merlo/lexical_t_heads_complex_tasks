"""Simple tasks, J-head: combined word lists (top-n task-specific + top-n common words, counted
on all five rule wordings; n = 5, 10) and heads selected by consensus: a head matches a prompt
when one of the list's tokens is among its TOP_K decoded tokens, and it must match on at least
RHO of the prompts of every one of the five rule templates (k = 5). The set 'top20' keeps the 20
heads with the highest lowest-template rate (ties: lowest-template mean list-token count, layer,
head); N_RANDOM random sets match top20's heads per layer, drawn outside top20."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from lth import READOUTS, simple
from lth.evaluation import tokenizer as get_tok
from lth.store import readings_dir
from lth.terms import expand_to_token_ids

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
TOP_K, RHO, N_TOP, N_RANDOM, N_HEADS, R, SEED = 5, 0.15, 20, 3, 32, READOUTS.index("jhead"), 20261105
tok = get_tok()
lists = json.load(open("results/word_lists_allid/word_lists.json"))
rng = np.random.default_rng(SEED)
out, info, used = {}, [], {}
for task in simple.TASKS:
    tops = [np.load(readings_dir("simple", task) / f"{w}.npz")["top"][:, R, :, :TOP_K] for w in simple.WORDINGS["rule"]]
    sets = []
    for n in (5, 10):
        spec, common = lists["specific"][task]["jhead"][:n], lists["common"]["jhead"][:n]
        words = spec + [w for w in common if w not in spec]
        used[f"{task}|{n}+{n}"] = words
        ids = np.array(sorted(set(expand_to_token_ids(tok, words))))
        hit = [np.isin(t, ids) for t in tops]                      # prompts, head, TOP_K
        rate = np.stack([h.any(-1).mean(0) for h in hit])          # templates, head
        count = np.stack([h.sum(-1).mean(0) for h in hit])
        low, low_count = rate.min(0), count[rate.argmin(0), np.arange(rate.shape[1])]
        passed = np.flatnonzero(low >= RHO)
        order = [int(h) for h in np.lexsort((np.arange(len(low)), -low_count, -low)) if low[h] >= RHO]
        top = order[:N_TOP]
        as_lh = lambda hs: sorted([[int(h) // N_HEADS, int(h) % N_HEADS] for h in hs])
        sets += [["rule", f"k5_{n}+{n}", as_lh(passed)], ["rule", f"k5_{n}+{n}_top20", as_lh(top)]]
        T = {(int(h) // N_HEADS, int(h) % N_HEADS) for h in top}
        for i in range(N_RANDOM):
            pick = []
            for layer in sorted({l for l, _ in T}):
                k = sum(1 for l, _ in T if l == layer)
                others = [(layer, x) for x in range(N_HEADS) if (layer, x) not in T]
                pick += [others[int(j)] for j in rng.choice(len(others), k, replace=False)]
            sets.append(["rule", f"k5_{n}+{n}_top20_random{i}", sorted(map(list, pick))])
        info.append(dict(task=task, list=f"{n}+{n}", words=len(words), n_pass=len(passed), n_top=len(top),
                         lowest_rate_top=round(float(low[top[-1]]), 3) if top else None,
                         layers_top=f"{min(l for l, _ in T)}-{max(l for l, _ in T)}" if T else ""))
    out[task] = {"sets": sets}
(OUT / "heads.json").write_text(json.dumps({"simple": out}, indent=1))
(OUT / "word_lists.json").write_text(json.dumps(used, indent=1))
import pandas as pd
pd.DataFrame(info).to_csv(OUT / "set_sizes.csv", index=False)
print(pd.DataFrame(info).to_string(index=False))
