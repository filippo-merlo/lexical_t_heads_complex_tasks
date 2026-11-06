"""J-head word shares on the neutral WikiText readings and on each simple task (all five rule
wordings), and specific/common lists rebuilt after removing the neutral top-K words."""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from wordfreq import zipf_frequency
from lth import READOUTS, simple
from lth.evaluation import tokenizer as get_tok
from lth.store import readings_dir

R = READOUTS.index("jhead"); tok = get_tok()
STOP = set("the and for with that this from are was were you your has have had not but its".split())
WORD = re.compile(r"^[a-z]{3,}$"); cache = {}
def word(t):
    if t not in cache:
        w = tok.decode([t]).strip().lower()
        cache[t] = w if WORD.match(w) and w not in STOP and zipf_frequency(w, "en") >= 3 else None
    return cache[t]
def shares(top):  # prompts, heads, 10
    c = Counter()
    for p in range(top.shape[0]):
        for h in range(top.shape[1]):
            c.update({w for x in top[p, h] if (w := word(int(x)))})
    return {w: n / (top.shape[0] * top.shape[1]) for w, n in c.items()}
neutral = shares(np.load("data/readings/neutral/wikitext.npz")["top"][:, R])
task = {t: shares(np.concatenate([np.load(readings_dir("simple", t) / f"{w}.npz")["top"][:, R] for w in simple.WORDINGS["rule"]])) for t in simple.TASKS}
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
(OUT / "shares_jhead.json").write_text(json.dumps({"neutral": neutral, "task": task}))
nrank = sorted(neutral, key=lambda w: (-neutral[w], w))
print("neutral words:", len(neutral)); print("neutral top 60:", ", ".join(f"{w} {100*neutral[w]:.2f}" for w in nrank[:60]))
def lists(removed, n=10):
    s = {t: {w: v for w, v in task[t].items() if w not in removed} for t in simple.TASKS}
    vocab = set.union(*(set(v) for v in s.values()))
    common = sorted((w for w in vocab if all(w in s[t] for t in simple.TASKS)), key=lambda w: (-min(s[t][w] for t in simple.TASKS), w))[:n]
    spec = {}
    for t in simple.TASKS:
        o = [x for x in simple.TASKS if x != t]
        spec[t] = sorted(s[t], key=lambda w: (-(s[t][w] - np.mean([s[x].get(w, 0) for x in o])), w))[:n]
    return common, spec
old = json.load(open("results/word_lists_allid/word_lists.json"))
print("\nold common (top 10):", old["common"]["jhead"][:10])
for K in (20, 50, 100):
    removed = set(nrank[:K]); common, spec = lists(removed)
    gone_c = [w for w in old["common"]["jhead"][:10] if w in removed]
    gone_s = {t: [w for w in old["specific"][t]["jhead"][:10] if w in removed] for t in simple.TASKS}
    print(f"\nK={K}: common -> {common}\n  removed from old common top10: {gone_c}")
    print("  removed from old specific top10:", {t: g for t, g in gone_s.items() if g})
    if K == 50:
        for t in simple.TASKS: print(f"  {t:17s} {spec[t]}")
