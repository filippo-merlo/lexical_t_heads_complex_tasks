"""Per simple task: the best J-head set of the word-list runs (highest check-1 accuracy at its
best alpha), and N_RANDOM random sets with the same number of heads per layer, drawn from the
layer's heads outside the best set (topped up with set heads when a layer has too few others)."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
N_RANDOM, N_HEADS, SEED = 5, 32, 20261105
a = pd.read_csv("results/word_lists_allid/summary_best_alpha_one_vs_all.csv")
a = a[a.heads.str.contains("jhead")]
a["source_dir"] = a.apply(lambda r: "results/heads.json" if (r.heads == "yang_jhead" and r.ident == "one") else
                          ("results/word_lists_5w/heads.json" if (r.ident == "one" and r.heads.endswith("_20")) else
                           ("results/word_lists_5w_len/heads.json" if r.ident == "one" else "results/word_lists_allid/heads.json")), axis=1)
u = pd.read_csv("results/word_lists_union/summary_best_alpha.csv").assign(ident="", source_dir="results/word_lists_union/heads.json")
c = pd.concat([a, u])
best = c.sort_values(["acc", "control"], ascending=[False, True]).groupby("task").head(1).set_index("task")
rng = np.random.default_rng(SEED)
out, info = {}, {}
for task, r in best.iterrows():
    h = json.load(open(r.source_dir))["simple"][task]
    if r.source_dir == "results/heads.json":
        heads = [[l, k] for l, k, _ in h["rule"]["jhead"]["heads"]]
    else:
        name = r.heads[:-3] if r.source_dir.endswith("word_lists_5w/heads.json") else r.heads
        heads = next(hs for _, n, hs in h["sets"] if n == name)
    S = {tuple(x) for x in heads}
    sets, overlaps = [["rule", "best", sorted(map(list, S))]], []
    for i in range(N_RANDOM):
        R = []
        for layer in sorted({l for l, _ in S}):
            k = sum(1 for l, _ in S if l == layer)
            others = [(layer, x) for x in range(N_HEADS) if (layer, x) not in S]
            inset = [(layer, x) for x in range(N_HEADS) if (layer, x) in S]
            pick = [others[j] for j in rng.choice(len(others), min(k, len(others)), replace=False)]
            if k > len(others):
                pick += [inset[j] for j in rng.choice(len(inset), k - len(others), replace=False)]
            R += pick
        overlaps.append(len(set(R) & S) / len(S))
        sets.append(["rule", f"random{i}", sorted(map(list, R))])
    out[task] = {"sets": sets}
    info[task] = dict(best=f"{r.heads}{'/' + r.ident if r.ident else ''}", acc=r.acc, alpha=r.alpha, control=r.control,
                      n=len(S), layers=f"{min(l for l, _ in S)}-{max(l for l, _ in S)}", overlap=round(float(np.mean(overlaps)), 3))
(OUT / "heads.json").write_text(json.dumps({"simple": out}, indent=1))
pd.DataFrame(info).T.to_csv(OUT / "best_sets.csv")
print(pd.DataFrame(info).T.to_string())
