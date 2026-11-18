"""Rebuilt design, stage 3 (paper/head-sets-simple-design.tex, Sec. 6.2): the causal score of example
prompts on the P example formats, on the choosing half.

As ``scripts/e27_demo_cie.py`` (Todd et al.'s causal indirect effect, the score Davidson et al. use for
demonstrations), with the prompts of this design: five shots in each P format (``lth.simple.
EXAMPLE_FORMATS``), the demonstrations those of ``lth.simple.demo_items`` (other items of the same half).
Correct prompts: those answered correctly in the format's records (flexible score, as the means of the
patching). Davidson et al.'s totals are spread evenly over the formats: the mean of each head's ``z`` at
the last token over 100 correct prompts (34/33/33), the CIE over 25 further correct prompts (9/8/8) with
the demonstrations' answers shuffled among them; target: the first token of the query's answer. A format
with fewer than ten correct prompts on the half is not used. The patched (check-1) items are in the
other half and never used.

Writes ``results/rebuild/examples/fold<f>/cie/``: ``cie.npz`` (one ``[n_layers, n_heads]`` array per
task) and ``items.json`` (the mean and CIE prompts: query, demonstrations, shuffled answers).

    .venv/bin/python scripts/rebuild_demo_cie.py --gpu 0 --fold 1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import RESULTS, gpu, simple  # noqa: E402

N_MEAN, N_CIE, MIN_CORRECT = 100, 25, 10
CIE_CHUNK = 5
FOLDS = {1: "A", 2: "B"}


def spread(total: int, k: int) -> list[int]:
    return [total // k + (j < total % k) for j in range(k)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS))
    args = parser.parse_args()
    gpu.pin(args.gpu)

    import numpy as np

    from davidson_fv import cie
    from patch import simple_mean

    from lth.model import LM
    from lth.store import load, readings_dir

    half = FOLDS[args.fold]
    split = json.loads((RESULTS / "heldout" / "split.json").read_text())
    out = RESULTS / "rebuild" / "examples" / f"fold{args.fold}" / "cie"
    out.mkdir(parents=True, exist_ok=True)
    lm = LM.load()
    formats = simple.split_formats("P")
    scores, plan = {}, {}
    for task in args.tasks:
        data = simple.dataset(task)
        members = set(split[task][half])
        free = {f: [i for i in simple.correct_items(load(readings_dir("simple", task), f)["records"]) if i in members]
                for f in formats}
        used = [f for f in formats if len(free[f]) >= MIN_CORRECT]
        rng = random.Random(f"rebuild-demo-cie-{task}-{half}")
        mean_parts, rows, targets, plan[task] = [], [], [], {}
        for f, n_mean, n_cie in zip(used, spread(N_MEAN, len(used)), spread(N_CIE, len(used))):
            items = free[f]
            mean_items, cie_items = ((items[:n_mean], items[n_mean:n_mean + n_cie]) if len(items) >= n_mean + n_cie
                                     else (items[:-n_cie], items[-n_cie:]))
            prompts = simple.prompts(task, f)[0]
            mean_parts.append((simple_mean(lm, simple.token_ids(lm.tokenizer, prompts), mean_items), len(mean_items)))
            shuffled = []
            for q in cie_items:
                demos = simple.demo_items(task, q, simple.item_halves(task)[q])
                outs = [data[d]["output"] for d in demos]
                rng.shuffle(outs)
                fake = [dict(input=data[d]["input"], output=o) for d, o in zip(demos, outs)] + [data[q]]
                text, answer = simple.example_prompt(fake, f[len(simple.FORMAT_PREFIX):], list(range(len(demos))), len(demos))
                assert answer == simple.EXAMPLE_FORMATS[f[len(simple.FORMAT_PREFIX):]][2] + data[q]["output"]
                rows.append(simple.token_ids(lm.tokenizer, [text])[0])
                targets.append(lm.first_token(answer))
                shuffled.append(dict(query=q, demos=demos, shuffled_outputs=outs))
            plan[task][f] = dict(mean=mean_items, cie=shuffled)
        mean_z = sum(m * n for m, n in mean_parts) / sum(n for _, n in mean_parts)
        total = 0
        for b in range(0, len(rows), CIE_CHUNK):
            part = cie(lm, rows[b:b + CIE_CHUNK], targets[b:b + CIE_CHUNK], mean_z).numpy()
            total = total + part * len(rows[b:b + CIE_CHUNK])
        scores[task] = total / len(rows)
        print(f"fold {args.fold} {task}: formats {used}, mean {sum(n for _, n in mean_parts)}, CIE {len(rows)} prompts",
              flush=True)
    np.savez(out / "cie.npz", **scores)
    (out / "items.json").write_text(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
