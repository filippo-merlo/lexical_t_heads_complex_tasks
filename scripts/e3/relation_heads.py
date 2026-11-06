"""Relation heads, J-head: a head reads the relation on a prompt when its top-10 at the last prompt
token holds both a token of the task's term list (Yang et al.'s, expanded as in Definition 1)
and a token of that prompt's own query (every token of the query's spellings, pieces of >= 3
characters). Relation rate = share of the five rule wordings' correct prompts (pooled) where both
hold. Sets: the top N heads by relation rate (ties: rate of the term list alone, layer, head),
the threshold set (rate >= RHO), and random sets matching the top-N's heads per layer."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from lth import READOUTS, simple
from lth.evaluation import tokenizer as get_tok
from lth.store import load, readings_dir
from lth.terms import expand_to_token_ids, simple_term_lists

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
SIZES, RANDOM_SIZES, N_RANDOM, RHO, N_HEADS, R, SEED = (1, 2, 5, 10, 20), (5, 10), 2, 0.10, 32, READOUTS.index("jhead"), 20261106
tok = get_tok(); rng = np.random.default_rng(SEED); terms = simple_term_lists()
def query_ids(text):
    ids = set()
    for v in {text, " " + text, text.capitalize(), " " + text.capitalize(), text.lower(), " " + text.lower()}:
        ids |= {i for i in tok(v, add_special_tokens=False).input_ids if len(tok.decode([i]).strip()) >= 3}
    return np.array(sorted(ids))
out, info = {}, []
for t in simple.TASKS:
    data = simple.dataset(t); rel = np.array(sorted(set(expand_to_token_ids(tok, terms[t]))))
    both, term_hit, query_hit = [], [], []
    for w in simple.WORDINGS["rule"]:
        run = load(readings_dir("simple", t), w); top = run["top"][:, R]  # prompts, heads, 10
        for p, item in enumerate(run["id_items"]):
            q = np.isin(top[p], query_ids(data[int(item)]["input"])).any(-1); r_ = np.isin(top[p], rel).any(-1)
            both.append(q & r_); term_hit.append(r_); query_hit.append(q)
    rate, trate, qrate = np.mean(both, 0), np.mean(term_hit, 0), np.mean(query_hit, 0)
    order = [int(h) for h in np.lexsort((np.arange(len(rate)), -trate, -rate)) if rate[h] > 0]
    sets, T_all = [], {(int(h) // N_HEADS, int(h) % N_HEADS) for h in np.flatnonzero(rate >= RHO)}
    for n in SIZES:
        if n > len(order): continue
        T = {(h // N_HEADS, h % N_HEADS) for h in order[:n]}
        sets.append(["rule", f"rel_top{n}", sorted(map(list, T))])
        if n in RANDOM_SIZES:
            for i in range(N_RANDOM):
                pick = []
                for layer in sorted({l for l, _ in T}):
                    k = sum(1 for l, _ in T if l == layer)
                    pool = [(layer, x) for x in range(N_HEADS) if (layer, x) not in T]
                    pick += [pool[int(j)] for j in rng.choice(len(pool), k, replace=False)]
                sets.append(["rule", f"rel_top{n}_random{i}", sorted(map(list, pick))])
    if T_all and len(T_all) not in SIZES: sets.append(["rule", "rel_threshold", sorted(map(list, T_all))])
    out[t] = {"sets": sets}
    info.append(dict(task=t, n_threshold=len(T_all), n_rate_pos=len(order),
                     top5=" ".join(f"L{h//N_HEADS}H{h%N_HEADS}({rate[h]:.2f}|t{trate[h]:.2f}|q{qrate[h]:.2f})" for h in order[:5])))
(OUT / "heads.json").write_text(json.dumps({"simple": out}, indent=1))
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
pd.DataFrame(info).to_csv(OUT / "ranking.csv", index=False); print(pd.DataFrame(info).to_string(index=False))
