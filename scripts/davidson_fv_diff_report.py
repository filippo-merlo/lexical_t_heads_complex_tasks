"""Tables of ``scripts/davidson_fv_diff.py``'s readings for the Davidson FV heads (no GPU).

For each FV head of ``--run`` (and the heads of ``--extra`` runs not in it), readout and task:
the answer share and input share (share of the 60 corrupted prompts whose own answer's,
or input's, first token with its leading space is among the head's top 10 of
``mean_z - z``), the tokens most often in that top 10 with their share of prompts, and the
top tokens of the mean difference ``mean_z - mean(z)`` (kinds pooled). Pooled over tasks:
the mean of the per-task shares.

Also, over all 1,024 heads: the heads ranked by their answer share under each readout
(mean over tasks), and how many of the run's FV heads are among the top ``len(FV)``.

Writes ``results/<--out>/fv_heads.json`` and ``fv_heads.txt``.

    .venv/bin/python scripts/davidson_fv_diff_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import DATA, READOUTS, RESULTS  # noqa: E402

N_HEADS = 32
N_TOKENS = 12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", default="davidson_fv_all14_30")
    parser.add_argument("--extra", nargs="*", default=["davidson_fv"])
    parser.add_argument("--out", default="davidson_fv_diff")
    args = parser.parse_args()

    from lth.evaluation import tokenizer as get_tokenizer

    tok = get_tokenizer()
    cache: dict[int, str] = {}

    def dec(t: int) -> str:
        if t not in cache:
            cache[t] = tok.decode([int(t)])
        return cache[t]

    def heads_of(run):
        return [(l, h) for l, h, _ in json.loads((RESULTS / run / "heads.json").read_text())["heads"]]

    fv = heads_of(args.run)
    extra = [x for run in args.extra for x in heads_of(run) if x not in fv]
    heads = fv + list(dict.fromkeys(extra))
    out = RESULTS / args.out
    tasks = list(json.loads((out / "prompts.json").read_text()))
    mean_top = np.load(out / "mean_top.npz")

    table = {"fv": [f"L{l}H{h}" for l, h in fv], "extra": [f"L{l}H{h}" for l, h in heads[len(fv):]],
             "tasks": {}, "answer_rank": {}}
    answer_all = {r: [] for r in READOUTS}
    for task in tasks:
        z = np.load(DATA / "readings" / "davidson_diff" / f"{task}.npz")
        top, answers, inputs = z["top"], z["answers"], z["inputs"]  # [P, 2, 1024, k]
        per = {}
        for r, readout in enumerate(READOUTS):
            answer = (top[:, r] == answers[:, None, None]).any(-1).mean(0)  # [1024]
            inp = (top[:, r] == inputs[:, None, None]).any(-1).mean(0)
            answer_all[readout].append(answer)
            for l, h in heads:
                site = l * N_HEADS + h
                counts = Counter()
                for row in top[:, r, site]:
                    counts.update(set(row.tolist()))
                best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:N_TOKENS]
                per[f"L{l}H{h}/{readout}"] = dict(
                    answer=round(float(answer[site]), 3), input=round(float(inp[site]), 3),
                    tokens=[[dec(t), round(c / len(top), 3)] for t, c in best],
                    mean_diff=[dec(t) for t in mean_top[f"{task}/diff_all"][r, site][:N_TOKENS]],
                    mean_task=[dec(t) for t in mean_top[f"{task}/mean_task"][r, site][:N_TOKENS]])
        table["tasks"][task] = {"n_prompts": int(len(top)), "heads": per}

    lines = []
    for readout in READOUTS:
        score = np.mean(answer_all[readout], axis=0)
        order = np.argsort(-score, kind="stable")
        top_n = {(int(i) // N_HEADS, int(i) % N_HEADS) for i in order[:len(fv)]}
        ranked = [[int(i) // N_HEADS, int(i) % N_HEADS, round(float(score[i]), 3)] for i in order[:len(fv)]]
        table["answer_rank"][readout] = dict(top=ranked, fv_in_top=len(top_n & set(fv)))
        lines.append(f"{readout}: heads by answer share of the difference (mean over tasks); "
                     f"FV heads among the top {len(fv)}: {len(top_n & set(fv))}")
        lines.append("  " + ", ".join(f"L{l}H{h} {s:.2f}" for l, h, s in ranked))
    lines.append("")

    for l, h in heads:
        name = f"L{l}H{h}"
        rank = fv.index((l, h)) + 1 if (l, h) in fv else "-"
        lines.append(f"===== {name} (FV rank {rank})")
        for readout in READOUTS:
            key = f"{name}/{readout}"
            rows = [table["tasks"][t]["heads"][key] for t in tasks]
            pooled = defaultdict(float)
            for row in rows:
                for s, v in row["tokens"]:
                    pooled[s] += v / len(tasks)
            best = sorted(pooled.items(), key=lambda kv: -kv[1])[:N_TOKENS]
            lines.append(f"  {readout:9s} answer {np.mean([x['answer'] for x in rows]):.2f} "
                         f"input {np.mean([x['input'] for x in rows]):.2f} | pooled: "
                         + ", ".join(f"{s!r} {v:.2f}" for s, v in best))
            for t, row in zip(tasks, rows):
                lines.append(f"     {t:17s} a{row['answer']:.2f} i{row['input']:.2f} "
                             + ", ".join(f"{s!r} {v:.2f}" for s, v in row["tokens"][:6])
                             + " || mean diff: " + ", ".join(repr(s) for s in row["mean_diff"][:6]))
    (out / "fv_heads.json").write_text(json.dumps(table, ensure_ascii=False))
    (out / "fv_heads.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:5]))


if __name__ == "__main__":
    main()
