"""What sets the Davidson FV task heads apart from heads that decode the task's words but are
not FV heads (no GPU). The motivation of E7 (``scripts/e7_sets.py``).

Per task and head (all 1,024): the J-head match rate of the task's CC list (``results/
word_lists_union``, five rule templates' P+ pooled, top 10), the FV score (``results/
davidson_fv_all14_30/cie.npz``, mean over the three corruption kinds) and its rank in the task,
and the per-head measures of ``results/davidson_fv_diff/generic_heads.json`` (attention at the
last token on task words, the query's input, the rest of the template and BOS; output cosine
across tasks; instruction change; J-head genericity). Groups per task: ``task24`` (FV heads
minus the 6 generic ones, ``results/davidson_fv_split/sets.json``), split into decoders (rate
>= 0.3) and non-decoders; ``generic6``; ``cc20_only`` / ``cc40_only`` (in the task's CC top 20 /
40 by match rate, ``results/cc_sizes``, not FV); the rest of layers 13-31.

Also: Spearman correlations of the FV score rank with each measure over the decoding (task,
head) pairs of layers 13-31 (rate >= 0.4); how many of them are in the task's FV top 30 with
BOS attention below 0.3 / 0.4 / 0.5 and above; and the CC-only heads that rank in a task's FV
top 30 with their number of such tasks and their rank by the mean score over the 14 tasks.

Writes ``results/davidson_fv_diff/fv_vs_cc.json`` (summary) and ``fv_vs_cc_rows.json.gz`` (rows).

    .venv/bin/python scripts/e3/fv_vs_cc_profile.py
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lth import READOUTS, RESULTS, simple  # noqa: E402
from lth.evaluation import tokenizer as get_tok  # noqa: E402
from lth.store import readings_dir  # noqa: E402
from lth.terms import expand_to_token_ids  # noqa: E402

OUT = RESULTS / "davidson_fv_diff"
MEASURES = ("att_taskword", "att_input", "att_rest", "att_bos", "out_cos", "instr_change",
            "jhead_same_as_diff", "jhead_generic", "breadth")


def spearman(a, b) -> float:
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def main() -> None:
    tok = get_tok()
    R = READOUTS.index("jhead")
    g = json.loads((OUT / "generic_heads.json").read_text())["heads"]
    S = json.loads((RESULTS / "davidson_fv_split/sets.json").read_text())
    t24, gen6 = ({tuple(h) for h in S[k]} for k in ("task24", "generic6"))
    lists = json.loads((RESULTS / "word_lists_union/word_lists.json").read_text())["all"]
    cz = np.load(RESULTS / "davidson_fv_all14_30/cie.npz")
    sized = {n: {t: {tuple(h[:2]) for s in v["sets"] if s[1] == n for h in s[2]}
                 for t, v in json.loads((RESULTS / "cc_sizes/heads.json").read_text())["simple"].items()}
             for n in ("cc_top20", "cc_top40")}
    rows = []
    for t in simple.TASKS:
        cie = np.mean([cz[f"{t}/{k}"] for k in ("equiprobable", "wikitext", "other_task")], 0)
        rank = (-cie.ravel()).argsort().argsort().reshape(32, 32)
        top = np.concatenate([np.load(readings_dir("simple", t) / f"{w}.npz")["top"][:, R]
                              for w in simple.WORDINGS["rule"]])
        ids = np.array(sorted(set(expand_to_token_ids(tok, lists[t]["union"]))))
        rate = np.isin(top, ids).any(-1).mean(0)
        for l in range(32):
            for h in range(32):
                k = (l, h)
                grp = ("task24" if k in t24 else "generic6" if k in gen6 else "cc20_only" if k in sized["cc_top20"][t]
                       else "cc40_only" if k in sized["cc_top40"][t] else "other_L13_31" if l >= 13 else "other")
                if grp == "task24":
                    grp = "task24_dec" if rate[l * 32 + h] >= 0.3 else "task24_nodec"
                m = g[f"L{l}H{h}"]
                rows.append(dict(task=t, head=f"L{l}H{h}", grp=grp, layer=l, rate=float(rate[l * 32 + h]),
                                 cie=float(cie[l, h]), cierank=int(rank[l, h]), **{x: m[x] for x in MEASURES}))
    d = pd.DataFrame(rows)
    cols = ["layer", "rate", "cierank", *MEASURES]
    groups = d.groupby("grp")
    summary = {"medians": groups[cols].median().round(3).to_dict(orient="index"),
               "pairs_per_task": (groups.size() / len(simple.TASKS)).round(2).to_dict()}
    dec = d[(d.rate >= 0.4) & (d.layer >= 13)]
    summary["decoders_L13_31"] = {"n_pairs": len(dec), "fv_pairs": int(dec.grp.str.startswith(("task24", "generic6")).sum()),
                                  "spearman_minus_rank": {c: round(spearman(dec[c], -dec.cierank), 3)
                                                          for c in ("rate", "layer", *MEASURES[:-1])}}
    summary["bos_marker"] = {str(x): {"below": int((dec.att_bos < x).sum()),
                                      "below_top30": int(((dec.att_bos < x) & (dec.cierank < 30)).sum()),
                                      "above": int((dec.att_bos >= x).sum()),
                                      "above_top30": int(((dec.att_bos >= x) & (dec.cierank < 30)).sum())}
                             for x in (0.3, 0.4, 0.5)}
    mean_rank = d.groupby("head").cie.mean().rank(ascending=False)
    in_top30 = d[d.cierank < 30].groupby("head").task.nunique()
    cc = d[d.grp == "cc20_only"]
    hit = cc[cc.cierank < 30]
    summary["cc_only_in_task_top30"] = {
        "pairs": len(hit), "of": len(cc),
        "heads": {h: {"tasks": int(in_top30[h]), "mean_rank": int(mean_rank[h]),
                      "best": hit[hit["head"] == h].sort_values("cierank").iloc[0][["task", "cierank"]].tolist()}
                  for h in sorted(hit["head"].unique(), key=lambda h: mean_rank[h])}}
    fv_heads = d[d.grp.isin(["task24_dec", "task24_nodec", "generic6"])]["head"].unique()
    summary["fv_tasks_in_top30_median"] = float(in_top30.reindex(fv_heads).fillna(0).median())
    (OUT / "fv_vs_cc.json").write_text(json.dumps(summary, indent=1, default=str))
    with gzip.open(OUT / "fv_vs_cc_rows.json.gz", "wt") as f:
        json.dump(rows, f)
    print(json.dumps(summary, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
