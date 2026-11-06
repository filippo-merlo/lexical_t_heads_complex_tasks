"""Prototype: per task, the words most often in the rule heads' top-10 at the last prompt token."""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from lth.evaluation import tokenizer as get_tok  # noqa
from lth.store import readings_dir
from lth import fu
from lth.terms import complex_term_lists, expand_to_token_ids, simple_term_lists

tok = get_tok()
heads = json.load(open("results/heads.json"))
STOP = set("the and for with that this from are was were you your has have had not but its".split())
WORD = re.compile(r"^[A-Za-z]{3,}$")

from wordfreq import zipf_frequency

def meaningful(w):
    # an English word of 3+ letters, not a function word, common enough (Zipf >= 3, ~1 per million)
    return bool(WORD.match(w)) and w not in STOP and zipf_frequency(w, "en") >= 3

def term_list(family, task):
    if family == "simple":
        return simple_term_lists()[task]
    return complex_term_lists()[fu.TASK_LEVELS[task].original]

def top_words(family, task, readout, n=15):
    term_ids = set(expand_to_token_ids(tok, term_list(family, task)))
    listed = set()
    entry = heads[family][task]["rule"][readout]
    wording = entry["wordings"][0]
    top = np.load(readings_dir(family, task) / f"{wording}.npz")["top"]  # prompts, readout, site, 10
    r = 0 if readout == "logithead" else 1
    n_heads = 32
    sites = [l * n_heads + h for l, h, *_ in entry["heads"]]
    counts = Counter()
    for p in range(top.shape[0]):
        for s in sites:
            # a word once per head and prompt (case and leading-space variants merged)
            words = {}
            for t in top[p, r, s]:
                if meaningful(w := tok.decode([int(t)]).strip().lower()):
                    words[w] = True
                    if int(t) in term_ids:
                        listed.add(w)
            counts.update(words)
    pairs = top.shape[0] * len(sites)  # share of (prompt, head) pairs whose top 10 holds the word
    return wording, len(sites), [(w + ("*" if w in listed else ""), c / pairs) for w, c in counts.most_common(n)]

family = sys.argv[1]
for task in sys.argv[2:]:
    print(f"{task} term list: {term_list(family, task)}")
    for readout in ("logithead", "jhead"):
        wording, k, words = top_words(family, task, readout)
        print(f"{task:22s} {readout:9s} {wording} {k:3d} heads: " + ", ".join(f"{w} {s:.2f}" for w, s in words))
