"""Simple tasks, J-head: word list = top-5 task-specific + top-5 common words (union), with the
words counted on the best (E1 identification) wording or on all five rule wordings, and heads
selected (Definition 1, k=10, rho=0.10) on the best wording or on all five. Sets
union_<list source>list_<identification>id."""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from wordfreq import zipf_frequency
from lth import READOUTS, simple
from lth.evaluation import tokenizer as get_tok
from lth.store import readings_dir
from lth.terms import expand_to_token_ids

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
N, RHO, N_HEADS, R = 5, 0.10, 32, READOUTS.index("jhead")
tok = get_tok()
heads = json.load(open("results/heads.json"))["simple"]
STOP = set("the and for with that this from are was were you your has have had not but its".split())
WORD = re.compile(r"^[a-z]{3,}$")
cache = {}
def word(t):
    if t not in cache:
        w = tok.decode([t]).strip().lower()
        cache[t] = w if WORD.match(w) and w not in STOP and zipf_frequency(w, "en") >= 3 else None
    return cache[t]

def load(task, which):
    best = heads[task]["rule"]["jhead"]["wordings"][0]
    ws = [best] if which == "best" else list(simple.WORDINGS["rule"])
    return np.concatenate([np.load(readings_dir("simple", task) / f"{w}.npz")["top"][:, R] for w in ws])  # prompts, head, 10

tops = {(t, w): load(t, w) for t in simple.TASKS for w in ("best", "all")}
lists, out = {}, {t: {"sets": []} for t in simple.TASKS}
for src in ("best", "all"):
    s = {}
    for t in simple.TASKS:
        top = tops[(t, src)]; counts = Counter()
        for p in range(top.shape[0]):
            for h in range(top.shape[1]):
                counts.update({w for x in top[p, h] if (w := word(int(x)))})
        s[t] = {w: c / (top.shape[0] * top.shape[1]) for w, c in counts.items()}
    vocab = set.union(*(set(v) for v in s.values()))
    common = sorted((w for w in vocab if all(w in s[t] for t in simple.TASKS)),
                    key=lambda w: (-min(s[t][w] for t in simple.TASKS), w))[:N]
    for t in simple.TASKS:
        others = [o for o in simple.TASKS if o != t]
        spec = sorted(s[t], key=lambda w: (-(s[t][w] - np.mean([s[o].get(w, 0) for o in others])), w))[:N]
        words = spec + [w for w in common if w not in spec]
        lists.setdefault(src, {})[t] = {"specific": spec, "common": common, "union": words}
        ids = np.array(sorted(set(expand_to_token_ids(tok, words))))
        for ident in ("best", "all"):
            rate = np.isin(tops[(t, ident)], ids).any(-1).mean(0)
            hs = [[int(h) // N_HEADS, int(h) % N_HEADS] for h in np.flatnonzero(rate >= RHO)]
            out[t]["sets"].append(["rule", f"union_{src}list_{ident}id", hs])
(OUT / "heads.json").write_text(json.dumps({"simple": out}, indent=1))
(OUT / "word_lists.json").write_text(json.dumps(lists, indent=1))
for src in ("best", "all"): print(src, "common:", lists[src][simple.TASKS[0]]["common"])
for t in simple.TASKS:
    print(f"{t:17s}", " | ".join(f"{src}: {lists[src][t]['specific']}" for src in ("best", "all")),
          "| sizes", {n.replace("union_", ""): len(h) for _, n, h in out[t]["sets"]})
