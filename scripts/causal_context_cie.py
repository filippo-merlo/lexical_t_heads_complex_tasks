"""Causal score measured in the prompt the sets are tested in (paper/head-sets-simple-design.tex, Sec. 4.2,
Causal (zero-shot); design in paper/archive/causal-context-design.tex).

Davidson et al.'s causal indirect effect as ``scripts/davidson_fv.py`` computes it for Causal, with the
uninformative instruction left out. Per fold, everything but the patched prompt is Causal's:

- items: ``results/heldout/fold<f>/fv/items.json``, per question-answer template of the choosing half 25
  prompts for the mean and 5 for the indirect effect (20 per task);
- mean: the head outputs at the last token averaged over the 100 mean prompts (``patch.simple_mean`` per
  template, weighted by its number of prompts, as ``davidson_fv.py``);
- target: the first token of " " + answer; the head set to its mean at the last token, one head at a time
  (``davidson_fv.cie``).

Contexts (the patched prompt of each of the 20 items):

- ``zero``: the zero-shot prompt " x:" with BOS, the prompt every set is tested in;
- ``frame``: " Q: x\\nA:" with BOS, Causal's prompt without the uninformative text before it.

``--check`` first recomputes Causal's WikiText-kind score of the first task (the uninformative text
re-encoded from ``uninformative.json``) and prints its largest difference from ``cie.npz``, which tests that
the items, mean and target are Causal's.

Writes ``results/heldout/fold<f>/fv/cie_context.npz`` (``<task>/<context>``, ``[n_layers, n_heads]``),
updated after every task; tasks already in it are skipped.

    .venv/bin/python scripts/causal_context_cie.py --gpu 0 --fold 1 --tasks antonym --check
    .venv/bin/python scripts/causal_context_cie.py --gpu 0 --fold 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import RESULTS, gpu, simple  # noqa: E402

CONTEXTS = ("zero", "frame")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS))
    parser.add_argument("--check", action="store_true",
                        help="first recompute Causal's WikiText score of the first task and compare with cie.npz")
    args = parser.parse_args()
    gpu.pin(args.gpu)

    import numpy as np

    from davidson_fv import QUERY, cie
    from patch import simple_mean

    from lth.model import LM

    d = RESULTS / "heldout" / f"fold{args.fold}" / "fv"
    plan = json.loads((d / "items.json").read_text())
    out = d / "cie_context.npz"
    scores = dict(np.load(out)) if out.exists() else {}
    lm = LM.load()
    tok, bos = lm.tokenizer, lm.tokenizer.bos_token_id

    for k, task in enumerate(args.tasks):
        if not args.check and all(f"{task}/{c}" in scores for c in CONTEXTS):
            print(f"{task}: done before, skipped", flush=True)
            continue
        start = time.time()
        data = simple.dataset(task)
        per = plan[task]["items"]
        parts = []
        for wording, it in per.items():
            ids = simple.token_ids(tok, simple.prompts(task, wording)[0])
            parts.append((simple_mean(lm, ids, it["mean"]), len(it["mean"])))
        mean = sum(m * n for m, n in parts) / sum(n for _, n in parts)
        items = [i for w in per for i in per[w]["cie"]]
        targets = [lm.first_token(" " + data[i]["output"]) for i in items]
        query = {i: tok(QUERY.format(input=data[i]["input"]), add_special_tokens=False).input_ids for i in items}

        if args.check and k == 0:
            texts = json.loads((d / "uninformative.json").read_text())[task]
            rows = [[bos] + tok(c, add_special_tokens=False).input_ids + query[i]
                    for w in per for c, i in zip(texts[w]["wikitext"], per[w]["cie"])]
            again = cie(lm, rows, targets, mean).numpy()
            saved = np.load(d / "cie.npz")[f"{task}/wikitext"]
            print(f"check {task}/wikitext: max |recomputed - saved| = {np.abs(again - saved).max():.2e} "
                  f"(saved max {np.abs(saved).max():.2e}); same top 20: "
                  f"{len(set(np.argsort(-again.ravel())[:20]) & set(np.argsort(-saved.ravel())[:20]))}/20", flush=True)

        zero = simple.token_ids(tok, simple.prompts(task, simple.ZERO)[0])
        rows = {"zero": [zero[i] for i in items], "frame": [[bos] + query[i] for i in items]}
        for c in CONTEXTS:
            scores[f"{task}/{c}"] = cie(lm, rows[c], targets, mean).numpy()
        partial = d / "cie_context.partial.npz"
        np.savez(partial, **scores)
        partial.replace(out)
        print(f"{task}: {len(items)} items, {time.time() - start:.0f} s; top-20 mean CIE "
              + ", ".join(f"{c} {np.sort(scores[f'{task}/{c}'].ravel())[-20:].mean():.4f}" for c in CONTEXTS),
              flush=True)


if __name__ == "__main__":
    main()
