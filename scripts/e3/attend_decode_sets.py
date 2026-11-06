"""Top-N heads by attend x decode score (results/attend_decode_scores.csv) per simple task, and
random sets matching the top-N's heads per layer (drawn outside it)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from lth import simple
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
SIZES, RANDOM_SIZES, N_RANDOM, SEED = (1, 2, 3, 5, 10, 20), (5, 10), 2, 20261106
rng = np.random.default_rng(SEED); d = pd.read_csv("results/attend_decode_scores.csv"); out = {}
for t in simple.TASKS:
    g = d[d.task == t].sort_values(["score", "attn", "layer", "head"], ascending=[False, False, True, True])
    order = list(zip(g.layer.astype(int), g["head"].astype(int))); sets = []
    for n in SIZES:
        T = set(order[:n]); sets.append(["rule", f"ad_top{n}", sorted(map(list, T))])
        if n in RANDOM_SIZES:
            for i in range(N_RANDOM):
                pick = []
                for layer in sorted({l for l, _ in T}):
                    k = sum(1 for l, _ in T if l == layer); pool = [(layer, x) for x in range(32) if (layer, x) not in T]
                    pick += [pool[int(j)] for j in rng.choice(len(pool), k, replace=False)]
                sets.append(["rule", f"ad_top{n}_random{i}", sorted(map(list, pick))])
    out[t] = {"sets": sets}
(OUT / "heads.json").write_text(json.dumps({"simple": out}, indent=1)); print({t: len(v["sets"]) for t, v in out.items()})
