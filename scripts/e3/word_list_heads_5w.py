"""Simple tasks: a common and a task-specific word list per readout (from all heads' top-10
at the last prompt token, pooled over the prompts of all five rule wordings inst0-4); heads
identified on the E1 identification wording only, then Definition-1 head sets from
each list (k=10, rho=0.10), in the set-list layout of scripts/rank_heads.py."""
import json, os, re, sys
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
PATCH_TASKS = sys.argv[2:]
N_WORDS, RHO, N_HEADS = 20, 0.10, 32
LENGTHS = [int(x) for x in os.environ.get("LIST_LENGTHS", "20").split()]  # prefixes of the ranked 20
tok = get_tok()
heads = json.load(open("results/heads.json"))["simple"]
STOP = set("the and for with that this from are was were you your has have had not but its".split())
WORD = re.compile(r"^[a-z]{3,}$")

def word(t):
    w = tok.decode([int(t)]).strip().lower()
    return w if WORD.match(w) and w not in STOP and zipf_frequency(w, "en") >= 3 else None

tops, shares = {}, {r: {} for r in READOUTS}
for task in simple.TASKS:
    wording = heads[task]["rule"]["jhead"]["wordings"][0]
    tops[task] = np.load(readings_dir("simple", task) / f"{wording}.npz")["top"]  # prompts, readout, head, 10
    top = np.concatenate([np.load(readings_dir("simple", task) / f"{w}.npz")["top"] for w in simple.WORDINGS["rule"]])
    words = {int(t): word(t) for t in np.unique(top)}
    for r, readout in enumerate(READOUTS):
        counts = Counter()
        for p in range(top.shape[0]):
            for h in range(top.shape[2]):
                counts.update({words[int(t)] for t in top[p, r, h] if words[int(t)]})
        pairs = top.shape[0] * top.shape[2]
        shares[readout][task] = {w: c / pairs for w, c in counts.items()}

lists, out = {}, {}
for r, readout in enumerate(READOUTS):
    s = shares[readout]
    vocab = set.union(*(set(v) for v in s.values()))
    # common: words in every task's lists, ranked by their smallest share over the tasks
    common = sorted((w for w in vocab if all(w in s[t] for t in simple.TASKS)),
                    key=lambda w: (-min(s[t][w] for t in simple.TASKS), w))[:N_WORDS]
    lists.setdefault("common", {})[readout] = common
    for task in simple.TASKS:
        others = [t for t in simple.TASKS if t != task]
        # specific: share in this task minus the mean share in the other 13
        spec = sorted(s[task], key=lambda w: (-(s[task][w] - np.mean([s[t].get(w, 0) for t in others])), w))[:N_WORDS]
        lists.setdefault("specific", {}).setdefault(task, {})[readout] = spec
for task in PATCH_TASKS:
    sets = []
    for kind in ("common", "specific"):
        for r, readout in enumerate(READOUTS):
          for n in LENGTHS:
            words = (lists[kind][readout] if kind == "common" else lists[kind][task][readout])[:n]
            ids = np.array(sorted(set(expand_to_token_ids(tok, words))))
            rate = np.isin(tops[task][:, r], ids).any(-1).mean(0)
            hs = [[int(h) // N_HEADS, int(h) % N_HEADS] for h in np.flatnonzero(rate >= RHO)]
            sets.append(["rule", f"{kind}_{readout}" + ("" if LENGTHS == [N_WORDS] else f"_{n}"), hs])
            print(f"{task:16s} {kind:8s} {readout:9s} {n:2d} words {len(hs):4d} heads")
    out[task] = {"sets": sets}
(OUT / "heads.json").write_text(json.dumps({"simple": out}, indent=1))
(OUT / "word_lists.json").write_text(json.dumps(lists, indent=1))
r = {k: {t: {ro: {w: round(shares[ro][t].get(w, 0), 5) for w in (lists["common"][ro] if k == "common" else lists[k][t][ro])}
             for ro in READOUTS} for t in simple.TASKS} for k in ("common", "specific")}
(OUT / "word_shares.json").write_text(json.dumps(r, indent=1))
print(json.dumps(lists["common"]))
for t in simple.TASKS:
    print(t, {ro: lists["specific"][t][ro] for ro in READOUTS})
