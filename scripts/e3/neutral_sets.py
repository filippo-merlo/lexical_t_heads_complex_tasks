"""J-head combined lists after the neutral cut (the K most frequent words on WikiText removed),
n + n words (n = 5, 10; specific and common rebuilt from the five rule wordings); heads selected
on the five wordings pooled: a list token among the head's top 10 on >= RHO of the prompts.
N_RANDOM random sets per list match the set's heads per layer, drawn outside it."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from lth import READOUTS, simple
from lth.evaluation import tokenizer as get_tok
from lth.store import readings_dir
from lth.terms import expand_to_token_ids

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
K, RHO, N_RANDOM, N_HEADS, R, SEED = 50, 0.10, 3, 32, READOUTS.index("jhead"), 20261105
tok = get_tok(); rng = np.random.default_rng(SEED)
sh = json.load(open("results/neutral_lists/shares_jhead.json")); neutral, task = sh["neutral"], sh["task"]
removed = set(sorted(neutral, key=lambda w: (-neutral[w], w))[:K])
s = {t: {w: v for w, v in task[t].items() if w not in removed} for t in simple.TASKS}
vocab = set.union(*(set(v) for v in s.values()))
common = sorted((w for w in vocab if all(w in s[t] for t in simple.TASKS)), key=lambda w: (-min(s[t][w] for t in simple.TASKS), w))
out, used, info = {}, {"removed": sorted(removed), "common": common[:20]}, []
for t in simple.TASKS:
    o = [x for x in simple.TASKS if x != t]
    spec = sorted(s[t], key=lambda w: (-(s[t][w] - np.mean([s[x].get(w, 0) for x in o])), w))
    top = np.concatenate([np.load(readings_dir("simple", t) / f"{w}.npz")["top"][:, R] for w in simple.WORDINGS["rule"]])
    sets = []
    for n in (5, 10):
        words = spec[:n] + [w for w in common[:n] if w not in spec[:n]]
        used[f"{t}|{n}+{n}"] = words
        ids = np.array(sorted(set(expand_to_token_ids(tok, words))))
        rate = np.isin(top, ids).any(-1).mean(0)
        S = {(int(h) // N_HEADS, int(h) % N_HEADS) for h in np.flatnonzero(rate >= RHO)}
        sets.append(["rule", f"neutral_{n}+{n}", sorted(map(list, S))])
        for i in range(N_RANDOM):
            pick = []
            for layer in sorted({l for l, _ in S}):
                k = sum(1 for l, _ in S if l == layer)
                others = [(layer, x) for x in range(N_HEADS) if (layer, x) not in S]
                assert len(others) >= k, (t, n, layer, k)
                pick += [others[int(j)] for j in rng.choice(len(others), k, replace=False)]
            sets.append(["rule", f"neutral_{n}+{n}_random{i}", sorted(map(list, pick))])
        info.append(dict(task=t, list=f"{n}+{n}", words=len(words), heads=len(S)))
    out[t] = {"sets": sets}
(OUT / "heads.json").write_text(json.dumps({"simple": out}, indent=1))
(OUT / "word_lists.json").write_text(json.dumps(used, indent=1))
import pandas as pd
pd.DataFrame(info).to_csv(OUT / "set_sizes.csv", index=False)
print("common 10:", common[:10]); print(pd.DataFrame(info).pivot(index="task", columns="list", values="heads").to_string())
print(used["antonym|10+10"])
