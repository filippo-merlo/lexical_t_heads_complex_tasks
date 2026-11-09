"""The Davidson FV heads in the CC-pooled sets (``results/word_lists_union``, ``union_alllist_allid``).

Per task: how many of the 30 FV heads (14-task run) and of the 20 (4-task run) are in the
task's CC-pooled set, in its top 20 / 40 by match rate (``results/cc_sizes``), and the chance
counts (uniform over the 1,024 heads; layer-matched: per FV head, the share of its layer's
heads in the CC set). Per FV head and task: which CC list words it matches (J-head, the five
rule templates' P+ pooled, top 10), with their rates.

    .venv/bin/python scripts/e3/fv_in_cc.py results/davidson_fv_diff/fv_in_cc.json
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from lth import READOUTS, simple
from lth.evaluation import tokenizer as get_tok
from lth.store import readings_dir
from lth.terms import expand_to_token_ids
tok = get_tok(); R = READOUTS.index("jhead")
lists = json.load(open(ROOT / "results/word_lists_union/word_lists.json"))["all"]
cc = {t: {tuple(h) for s in v["sets"] if s[1] == "union_alllist_allid" for h in s[2]}
      for t, v in json.load(open(ROOT / "results/word_lists_union/heads.json"))["simple"].items()}
fv30 = [tuple(h[:2]) for h in json.load(open(ROOT / "results/davidson_fv_all14_30/heads.json"))["heads"]]
fv20 = [tuple(h[:2]) for h in json.load(open(ROOT / "results/davidson_fv/heads.json"))["heads"]]
heads = list(dict.fromkeys(fv30 + fv20)); sites = [l * 32 + h for l, h in heads]
out = {}
for t in simple.TASKS:
    top = np.concatenate([np.load(readings_dir("simple", t) / f"{w}.npz")["top"][:, R][:, sites]
                          for w in simple.WORDINGS["rule"]])
    words = lists[t]["union"]
    ids = {w: np.array(sorted(set(expand_to_token_ids(tok, [w])))) for w in words}
    allids = np.array(sorted(set(expand_to_token_ids(tok, words))))
    total = np.isin(top, allids).any(-1).mean(0)
    for j, h in enumerate(heads):
        rates = {w: float(np.isin(top[:, j], ids[w]).any(-1).mean()) for w in words}
        out.setdefault(f"L{h[0]}H{h[1]}", {})[t] = {"in_cc": h in cc[t], "rate": float(total[j]),
            "words": {w: round(r, 2) for w, r in sorted(rates.items(), key=lambda kv: -kv[1]) if r >= 0.05},
            "specific": lists[t]["specific"]}
from collections import Counter  # noqa: E402

def sized(name):
    return {t: {tuple(h[:2]) for s in v["sets"] if s[1] == name for h in s[2]}
            for t, v in json.load(open(ROOT / "results/cc_sizes/heads.json"))["simple"].items()}

top20, top40 = sized("cc_top20"), sized("cc_top40")
counts = {}
for t in simple.TASKS:
    per_layer = Counter(l for l, _ in cc[t])
    counts[t] = dict(cc=len(cc[t]), fv30=len(set(fv30) & cc[t]), fv20=len(set(fv20) & cc[t]),
                     chance_uniform=30 * len(cc[t]) / 1024, chance_layer=sum(per_layer[l] / 32 for l, _ in fv30),
                     fv30_top20=len(set(fv30) & top20[t]), fv30_top40=len(set(fv30) & top40[t]))
    print(t, counts[t])
json.dump({"common": lists[simple.TASKS[0]]["common"], "counts": counts, "heads": out}, open(sys.argv[1], "w"))
print("common:", lists[simple.TASKS[0]]["common"])
for t in simple.TASKS: print(t, lists[t]["specific"])
