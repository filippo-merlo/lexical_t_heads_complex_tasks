"""J-head, per task: the N words heads read most during the task (share of prompt-head pairs over
the five rule wordings' correct prompts) after removing the K words heads read most on WikiText;
heads selected on the five wordings pooled (a list token among the top 10 on >= RHO of the
prompts); N_RANDOM random sets with the set's heads per layer, drawn outside it."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from lth import READOUTS, simple
from lth.evaluation import tokenizer as get_tok
from lth.store import readings_dir
from lth.terms import expand_to_token_ids

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
K, N, RHO, N_RANDOM, N_HEADS, R, SEED = 50, 10, 0.10, 3, 32, READOUTS.index("jhead"), 20261105
tok = get_tok(); rng = np.random.default_rng(SEED)
sh = json.load(open("results/neutral_lists/shares_jhead.json")); neutral, task = sh["neutral"], sh["task"]
cut = set(sorted(neutral, key=lambda w: (-neutral[w], w))[:K])
old = {t: {(l, h) for l, h in next(hs for _, n, hs in v["sets"] if n == "union_alllist_allid")}
       for t, v in json.load(open("results/word_lists_union/heads.json"))["simple"].items()}
e1 = {t: {(l, h) for l, h, _ in v["rule"]["jhead"]["heads"]} for t, v in json.load(open("results/heads.json"))["simple"].items()}
out, used, info = {}, {}, []
for t in simple.TASKS:
    words = [w for w in sorted(task[t], key=lambda w: (-task[t][w], w)) if w not in cut][:N]
    used[t] = words
    top = np.concatenate([np.load(readings_dir("simple", t) / f"{w}.npz")["top"][:, R] for w in simple.WORDINGS["rule"]])
    ids = np.array(sorted(set(expand_to_token_ids(tok, words))))
    rate = np.isin(top, ids).any(-1).mean(0)
    S = {(int(h) // N_HEADS, int(h) % N_HEADS) for h in np.flatnonzero(rate >= RHO)}
    sets = [["rule", "taskneutral_10", sorted(map(list, S))]]
    for i in range(N_RANDOM):
        pick = []
        for layer in sorted({l for l, _ in S}):
            k = sum(1 for l, _ in S if l == layer)
            others = [(layer, x) for x in range(N_HEADS) if (layer, x) not in S]
            pick += [others[int(j)] for j in rng.choice(len(others), k, replace=False)]
        sets.append(["rule", f"taskneutral_10_random{i}", sorted(map(list, pick))])
    out[t] = {"sets": sets}
    ls = [l for l, _ in S]
    info.append(dict(task=t, heads=len(S), layers=f"{min(ls)}-{max(ls)}", mean_layer=round(float(np.mean(ls)), 1),
                     in_recipe=f"{len(S & old[t])}/{len(old[t])}", in_E1=f"{len(S & e1[t])}/{len(e1[t])}"))
(OUT / "heads.json").write_text(json.dumps({"simple": out}, indent=1))
(OUT / "word_lists.json").write_text(json.dumps({"removed": sorted(cut), "lists": used}, indent=1))
import pandas as pd
pd.DataFrame(info).to_csv(OUT / "set_sizes.csv", index=False); print(pd.DataFrame(info).to_string(index=False))
