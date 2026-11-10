"""E8 D-filter design check (no GPU), run before the D rerun: list length 10, 30 or 50 words, with and
without the leading-space capitalised / upper forms (``lth.terms.expand_lowercase_to_token_ids``).
Per fold (choosing half) and task: D size, E4 FV-20 heads in D, DB size (D and BOS <= 0.5), FV-20
heads in DB, and how many list words are among the 50 words heads decode most on WikiText
(E7's ``removed``). No patching; FV-20 = results/heldout/fold<f>/heads.json, chosen on the same half.

Writes results/e8_filters/list_length.csv.

    .venv/bin/python scripts/e8_list_length.py
"""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path.cwd()))
from lth import DATA, READOUTS, RESULTS, simple
from lth.evaluation import tokenizer as get_tok
from lth.store import load
from lth.terms import expand_lowercase_to_token_ids, expand_to_token_ids
from wordfreq import zipf_frequency
R = READOUTS.index("jhead"); NH = 32
STOP = set("the and for with that this from are was were you your has have had not but its".split())
WORD = re.compile(r"^[a-z]{3,}$"); tok = get_tok(); cache = {}
def word(t):
    if t not in cache:
        w = tok.decode([t]).strip().lower()
        cache[t] = w if WORD.match(w) and w not in STOP and zipf_frequency(w, "en") >= 3 else None
    return cache[t]
def counts(top):
    c = Counter()
    for p in range(top.shape[0]):
        for h in range(top.shape[1]):
            c.update({w for x in top[p, h] if (w := word(int(x)))})
    return c
def ids_of(words, fix):
    return np.array(expand_lowercase_to_token_ids(tok, words) if fix else sorted(set(expand_to_token_ids(tok, words))))
neutral = counts(np.load(DATA / "readings" / "neutral" / "wikitext.npz")["top"][:, R])
generic = set(sorted(neutral, key=lambda w: (-neutral[w], w))[:50])
rows = []
for fold, half in ((1, "A"), (2, "B")):
    fv = {t: {h[0] * NH + h[1] for s in json.loads((RESULTS / "heldout" / f"fold{fold}" / "heads.json").read_text())["simple"][t]["sets"]
              if s[1] == "fv20" for h in s[2]} for t in simple.TASKS}
    for t in simple.TASKS:
        pooled = np.concatenate([load(DATA / "readings" / "heldout" / half / "simple" / t, w)["top"][:, R] for w in simple.WORDINGS["rule"]])
        c = counts(pooled); ranked = sorted(c, key=lambda w: (-c[w], w))
        bos = np.load(DATA / "readings" / "heldout" / f"attention_{half}" / f"{t}.npy").astype(np.float32).mean(0)[:, 3]
        for n in (10, 30, 50):
            for fix in (False, True):
                D = set(np.flatnonzero(np.isin(pooled, ids_of(ranked[:n], fix)).any(-1).mean(0) >= 0.10))
                DB = {h for h in D if bos[h] <= 0.5}
                rows.append(dict(fold=fold, task=t, n=n, fix=fix, D=len(D), D_fv=len(D & fv[t]), DB=len(DB),
                                 DB_fv=len(DB & fv[t]), generic=sum(w in generic for w in ranked[:n])))
        print(fold, t, "done", flush=True)
d = pd.DataFrame(rows); d.to_csv(RESULTS / "e8_filters" / "list_length.csv", index=False)
