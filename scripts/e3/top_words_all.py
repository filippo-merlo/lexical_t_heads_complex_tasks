"""Per task, the words most often in ANY head's top-10 at the last prompt token (rule wording the heads were found on)."""
import json, re, sys
from collections import Counter
import numpy as np
sys.path.insert(0, ".")
from wordfreq import zipf_frequency
from lth import fu
from lth.evaluation import tokenizer as get_tok
from lth.store import readings_dir
from lth.terms import complex_term_lists, expand_to_token_ids, simple_term_lists

tok = get_tok()
heads = json.load(open("results/heads.json"))
STOP = set("the and for with that this from are was were you your has have had not but its".split())
WORD = re.compile(r"^[a-z]{3,}$")
N = int(sys.argv[1])

def word(t):
    w = tok.decode([int(t)]).strip().lower()
    return w if WORD.match(w) and w not in STOP and zipf_frequency(w, "en") >= 3 else None

out = []
for family in ("simple", "complex"):
    lists = simple_term_lists() if family == "simple" else complex_term_lists()
    for task, styles in heads[family].items():
        terms = lists[task] if family == "simple" else lists[fu.TASK_LEVELS[task].original]
        term_ids = set(expand_to_token_ids(tok, terms))
        wording = styles["rule"]["jhead"]["wordings"][0]
        assert wording == styles["rule"]["logithead"]["wordings"][0]
        top = np.load(readings_dir(family, task) / f"{wording}.npz")["top"]  # prompts, readout, head, 10
        ids = np.unique(top)
        words = {int(t): word(t) for t in ids}
        out.append(f"## {task} ({family}, {wording}, {top.shape[0]} prompts x {top.shape[2]} heads)")
        for r, readout in enumerate(("Logit-head", "J-head")):
            counts, listed = Counter(), set()
            for p in range(top.shape[0]):
                for h in range(top.shape[2]):
                    seen = {words[int(t)] for t in top[p, r, h] if words[int(t)]}
                    listed |= {words[int(t)] for t in top[p, r, h] if words[int(t)] and int(t) in term_ids}
                    counts.update(seen)
            pairs = top.shape[0] * top.shape[2]
            out.append(f"**{readout}:** " + ", ".join(f"{w}{'*' if w in listed else ''} {100 * c / pairs:.1f}"
                                                   for w, c in counts.most_common(N)))
        out.append("")
open(sys.argv[2], "w").write("\n".join(out))
print("\n".join(out))
