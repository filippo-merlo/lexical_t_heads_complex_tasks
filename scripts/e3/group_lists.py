"""J-head, per task: 5 task words = highest (share in the task - highest share in another task of
its group), counted on the five rule wordings' correct prompts; plus the 5 words heads read most
on WikiText. Heads selected on the five wordings pooled (a list token among the top 10 on >= RHO
of the prompts); N_RANDOM random sets with the set's heads per layer, drawn outside it."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from lth import READOUTS, simple
from lth.evaluation import tokenizer as get_tok
from lth.store import readings_dir
from lth.terms import expand_to_token_ids

GROUPS = {"word relations": ["antonym", "synonym"], "morphology": ["singular-plural", "present-past"],
          "facts": ["country-capital", "country-currency", "landmark-country", "park-country", "product-company"],
          "sequences": ["next_item", "prev_item"], "translations": ["english-french", "english-german", "english-spanish"]}
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
N, N_GENERAL, RHO, N_RANDOM, N_HEADS, R, SEED = 5, 5, 0.10, 3, 32, READOUTS.index("jhead"), 20261105
tok = get_tok(); rng = np.random.default_rng(SEED)
sh = json.load(open("results/neutral_lists/shares_jhead.json")); neutral, task = sh["neutral"], sh["task"]
general = sorted(neutral, key=lambda w: (-neutral[w], w))[:N_GENERAL]
group_of = {t: g for g, ts in GROUPS.items() for t in ts}; assert set(group_of) == set(simple.TASKS)
out, used, info = {}, {"general": general, "groups": GROUPS}, []
for t in simple.TASKS:
    others = [o for o in GROUPS[group_of[t]] if o != t]
    spec = sorted(task[t], key=lambda w: (-(task[t][w] - max(task[o].get(w, 0) for o in others)), w))[:N]
    words = spec + [w for w in general if w not in spec]
    used[t] = {"specific": spec, "list": words}
    top = np.concatenate([np.load(readings_dir("simple", t) / f"{w}.npz")["top"][:, R] for w in simple.WORDINGS["rule"]])
    ids = np.array(sorted(set(expand_to_token_ids(tok, words))))
    rate = np.isin(top, ids).any(-1).mean(0)
    S = {(int(h) // N_HEADS, int(h) % N_HEADS) for h in np.flatnonzero(rate >= RHO)}
    sets = [["rule", "group5_wiki5", sorted(map(list, S))]]
    for i in range(N_RANDOM):
        pick = []
        for layer in sorted({l for l, _ in S}):
            k = sum(1 for l, _ in S if l == layer)
            pool = [(layer, x) for x in range(N_HEADS) if (layer, x) not in S]
            pick += [pool[int(j)] for j in rng.choice(len(pool), k, replace=False)]
        sets.append(["rule", f"group5_wiki5_random{i}", sorted(map(list, pick))])
    out[t] = {"sets": sets}
    info.append(dict(task=t, group=group_of[t], heads=len(S), specific=", ".join(spec)))
(OUT / "heads.json").write_text(json.dumps({"simple": out}, indent=1))
(OUT / "word_lists.json").write_text(json.dumps(used, indent=1))
import pandas as pd
pd.DataFrame(info).to_csv(OUT / "set_sizes.csv", index=False)
print("general:", general); print(pd.DataFrame(info).to_string(index=False))
