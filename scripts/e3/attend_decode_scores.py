"""Per task and head (J-head): attn = mean share of the last token's attention on task-word
positions (data/readings/attention_rule), decode = share of the same prompts whose top-10 holds a
task-word token (Definition-1 expansion). Both over the five rule wordings' correct prompts."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from lth import READOUTS, simple
from lth.evaluation import tokenizer as get_tok
from lth.store import load, readings_dir
from lth.terms import expand_to_token_ids
tok = get_tok(); R = READOUTS.index("jhead"); info = json.load(open("data/readings/attention_rule/info.json"))
rows = []
for t in simple.TASKS:
    att = np.load(f"data/readings/attention_rule/{t}.npy").astype(np.float32)  # prompts, heads, 4
    top = np.concatenate([load(readings_dir("simple", t), w)["top"][:, R] for w in simple.WORDINGS["rule"]])
    assert len(top) == len(att), t
    ids = np.array(sorted(set(expand_to_token_ids(tok, info[t]["words"]))))
    dec = np.isin(top, ids).any(-1).mean(0)
    for h in range(att.shape[1]):
        rows.append(dict(task=t, layer=h // 32, head=h % 32, attn=att[:, h, 0].mean(), attn_query=att[:, h, 1].mean(), attn_bos=att[:, h, 3].mean(), decode=dec[h]))
d = pd.DataFrame(rows); d["score"] = d.attn * d.decode
d.to_csv(sys.argv[1], index=False)
x = d[(d.layer == 16) & (d["head"] == 25)].set_index("task")[["attn", "attn_query", "attn_bos", "decode", "score"]]
x["attn_rank"] = [int((d[d.task == t].attn > x.loc[t, "attn"]).sum()) + 1 for t in x.index]
x["score_rank"] = [int((d[d.task == t].score > x.loc[t, "score"]).sum()) + 1 for t in x.index]
print("L16H25:\n", x.round(3).to_string())
print("\nheads with attn>=0.3 and decode>=0.5 per task:", d[(d.attn >= 0.3) & (d.decode >= 0.5)].groupby("task").size().reindex(simple.TASKS, fill_value=0).to_dict())
for t in simple.TASKS:
    g = d[d.task == t].sort_values("score", ascending=False).head(6)
    print(f"{t:17s}", " ".join(f"L{r.layer}H{r.head}(a{r.attn:.2f},d{r.decode:.2f})" for r in g.itertuples()))
