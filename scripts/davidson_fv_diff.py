"""Readings of what the instruction adds to each head: the difference between a head's mean
output on correct task prompts and its output on Davidson et al.'s corrupted prompts.

Davidson et al.'s head score (``scripts/davidson_fv.py``) sets a head, at the last token of a
prompt whose instruction was replaced by an uninformative one, to its mean output over
correct instruction prompts. The change it makes to the head is ``mean_z - z``. Here that
difference is read through the two readouts (``LM.top_tokens``: Logit-head ``W_O^h``, J-head
``J_{l,h}``, both through the final norm and the unembedding), for every head:

- the mean ``z`` of a task: as ``davidson_fv.py``, over the mean items of ``items.json``
  (25 correct prompts for each of inst1-inst4), checked against the saved FV (``fv.npz``);
- the corrupted prompts: the saved uninformative instructions (``uninformative.json``, kept as
  text, so re-encoded here) of the three kinds, each with its CIE item's query, 5 per
  instruction and kind (60 per task), as in the CIE;
- per corrupted prompt, the top ``--k`` tokens of ``mean_z - z``;
- per kind and pooled, the top ``--k-mean`` tokens of ``mean_z - mean(z)``; also those of
  ``mean_z`` itself and of the corrupted prompts' mean ``z``.

Also logs, per corrupted prompt, the model's greedy next token and whether it is the
answer's first token.

Writes ``data/readings/davidson_diff/{task}.npz`` (per-prompt tops, ``[P, 2, 1024, k]``) and
``results/<--out>/``: ``prompts.json`` (every corrupted prompt, its item, answer and the
model's next token), ``mean_top.npz`` (``{task}/{diff_<kind>|diff_all|mean_task|mean_corrupt}``,
``[2, 1024, k_mean]``) and ``fv_check.json``.

    .venv/bin/python scripts/davidson_fv_diff.py --gpu 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS, gpu, simple  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--run", default="davidson_fv_all14_30", help="Davidson FV run under results/")
    parser.add_argument("--out", default="davidson_fv_diff", help="output folder under results/")
    parser.add_argument("--k", type=int, default=10, help="tokens per head and corrupted prompt")
    parser.add_argument("--k-mean", type=int, default=20, help="tokens per head for the mean differences")
    args = parser.parse_args()
    gpu.pin(args.gpu)

    import numpy as np
    import torch

    from davidson_fv import INSTRUCTIONS, KINDS, QUERY, function_vector
    from patch import simple_mean

    from lth.model import LM

    run = RESULTS / args.run
    plan = json.loads((run / "items.json").read_text())
    corrupted = json.loads((run / "uninformative.json").read_text())
    fv_saved = np.load(run / "fv.npz")
    heads = [(l, h) for l, h, _ in json.loads((run / "heads.json").read_text())["heads"]]
    tasks = list(plan)
    out = RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)
    per_prompt = DATA / "readings" / "davidson_diff"
    per_prompt.mkdir(parents=True, exist_ok=True)

    lm = LM.load()
    tok = lm.tokenizer
    prompts_log, mean_top, fv_check = {}, {}, {}
    for task in tasks:
        data = simple.dataset(task)
        # Mean z over the mean items, as davidson_fv.py.
        parts = []
        for wording in INSTRUCTIONS:
            ids = simple.token_ids(tok, simple.prompts(task, wording)[0])
            items = plan[task]["items"][wording]["mean"]
            parts.append((simple_mean(lm, ids, items), len(items)))
        mean = sum(m * n for m, n in parts) / sum(n for _, n in parts)  # [n_layers, d_source], float32
        fv = function_vector(lm, heads, mean).cpu().numpy()
        saved = fv_saved[f"{task}/task"]
        fv_check[task] = dict(max_abs_diff=float(np.abs(fv - saved).max()), saved_norm=float(np.linalg.norm(saved)),
                              cosine=float(fv @ saved / np.linalg.norm(fv) / np.linalg.norm(saved)))

        # Corrupted prompts, in the CIE's order: kind, then instruction, then its 5 items.
        rows, meta = [], []
        for kind in KINDS:
            for wording in INSTRUCTIONS:
                for text, item in zip(corrupted[task][wording][kind], plan[task]["items"][wording]["cie"]):
                    c = tok(text, add_special_tokens=False).input_ids
                    query = tok(QUERY.format(input=data[item]["input"]), add_special_tokens=False).input_ids
                    rows.append([tok.bos_token_id] + c + query)
                    meta.append(dict(kind=kind, wording=wording, item=item, instruction=text,
                                     input=data[item]["input"], expected=data[item]["output"],
                                     answer_token=lm.first_token(" " + data[item]["output"])))
        z_rows, next_tokens = [], []
        for b in range(0, len(rows), simple.BATCH):
            t, z = lm.next_tokens_batch(rows[b:b + simple.BATCH])
            next_tokens += t
            z_rows.append(z.float().cpu())
        z_rows = torch.cat(z_rows)  # [P, n_layers, d_source]
        for m, t in zip(meta, next_tokens):
            m.update(predicted_token=int(t), predicted=tok.decode([int(t)]), correct=int(t) == m["answer_token"])
        prompts_log[task] = meta

        mean_dev = mean.to(lm.device)
        tops = torch.stack([lm.top_tokens(mean_dev - z_rows[p].to(lm.device), k=args.k) for p in range(len(rows))])
        np.savez_compressed(per_prompt / f"{task}.npz", top=tops.numpy(),
                            answers=np.array([m["answer_token"] for m in meta]),
                            inputs=np.array([lm.first_token(" " + m["input"]) for m in meta]),
                            kinds=np.array([m["kind"] for m in meta]))
        kinds = np.array([m["kind"] for m in meta])
        for kind in (*KINDS, "all"):
            sel = torch.tensor(kinds == kind) if kind != "all" else torch.ones(len(meta), dtype=torch.bool)
            mean_top[f"{task}/diff_{kind}"] = lm.top_tokens(mean_dev - z_rows[sel].mean(0).to(lm.device),
                                                            k=args.k_mean).numpy()
        mean_top[f"{task}/mean_task"] = lm.top_tokens(mean_dev, k=args.k_mean).numpy()
        mean_top[f"{task}/mean_corrupt"] = lm.top_tokens(z_rows.mean(0).to(lm.device), k=args.k_mean).numpy()
        acc = np.mean([m["correct"] for m in meta])
        print(f"{task}: {len(rows)} corrupted prompts, next token = answer {acc:.2f}; FV check "
              f"max |diff| {fv_check[task]['max_abs_diff']:.2e} (norm {fv_check[task]['saved_norm']:.1f})", flush=True)

    (out / "prompts.json").write_text(json.dumps(prompts_log, indent=1, ensure_ascii=False))
    (out / "fv_check.json").write_text(json.dumps(fv_check, indent=1))
    np.savez_compressed(out / "mean_top.npz", **mean_top)


if __name__ == "__main__":
    main()
