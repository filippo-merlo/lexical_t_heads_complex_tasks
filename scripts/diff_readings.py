"""E6: identification readings of the instruction's contribution to each head.

For every simple task, half of the E4 split (``results/heldout/split.json``) and instruction
template inst1-inst4 (inst0, the cloze template, has no instruction): the E4 identification
prompts (P+ of the half, ``data/readings/heldout/{half}``), each with two corrupted versions in
which only the instruction is replaced, by uninformative text matched to it in token length
and log-probability (Davidson et al. 2025, as ``scripts/davidson_fv.py``):

- ``equiprobable``: tokens drawn one at a time among those as probable, given the sequence so
  far, as the instruction's own token at that position;
- ``wikitext``: WikiText-103 prefixes of about the instruction's length, the closest in
  log-probability.

Davidson et al.'s third kind, other tasks' instructions, is left out: related tasks (the three
translations, the country tasks) share processes, and subtracting a related task's instruction
would remove them. ``N_POOL`` corrupted instructions of each kind are made per template, once
(``results/diff_heads/uninformative.json``, token ids and text); prompt ``j`` of a template uses
the ``j mod N_POOL``-th of each kind.

The reading of a prompt: every head's ``z`` at the last token of the clean prompt minus the
mean of its ``z`` on the two corrupted versions, through both readouts (``LM.top_tokens``), top
10. Clean and corrupted prompts are encoded alike: BOS + instruction + query (`` Q: {input}\\nA:``);
the share of clean prompts whose encoding equals the saved prompt's is printed.

Writes ``data/readings/diff/{half}/simple/{task}/{wording}`` (``lth.store`` format, E4's records
and P+ items) and ``results/diff_heads/next_tokens/{half}_{task}.jsonl.gz`` (per prompt: item,
answer, the model's next token on the clean and on each corrupted version).

    .venv/bin/python scripts/diff_readings.py --gpu 2
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import DATA, RESULTS, gpu, simple  # noqa: E402

KINDS = ("equiprobable", "wikitext")
WORDINGS = ("inst1", "inst2", "inst3", "inst4")
N_POOL = 20
SEED = 0
OUT = DATA / "readings" / "diff"
RES = RESULTS / "diff_heads"


def make_pools(lm, tasks) -> dict:
    """Per task and template: the instruction, its length and log-probability, and ``N_POOL``
    uninformative instructions of each kind (token ids)."""
    from davidson_fv import WIKI_POOL, closest, equiprobable, instruction, token_logprobs, wiki_prefixes

    rng = random.Random(SEED)
    wiki = wiki_prefixes(lm, rng)
    wiki_all = [c for cs in wiki.values() for c in cs]
    pools = {}
    for task in tasks:
        pools[task] = {}
        for wording in WORDINGS:
            text = instruction(task, wording)
            q = lm.tokenizer(text).input_ids  # with BOS
            target = token_logprobs(lm, [q])[0]
            logprob, length = sum(target), len(q) - 1
            near = [c for k in range(length - 3, length + 4) for c in wiki.get(k, [])]
            ids = dict(equiprobable=equiprobable(lm, q, target, N_POOL, rng),
                       wikitext=closest(lm, near or wiki_all, length, logprob, WIKI_POOL, N_POOL, rng))
            pools[task][wording] = dict(instruction=text, length=length, logprob=logprob, ids=ids,
                                        text={k: [lm.tokenizer.decode(c) for c in v] for k, v in ids.items()})
        print(f"{task}: pools ready", flush=True)
    return pools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(simple.TASKS))
    args = parser.parse_args()
    gpu.pin(args.gpu)

    from davidson_fv import QUERY, instruction

    from lth.model import LM
    from lth.store import done, load, load_outputs, outputs_path, readings_dir, save

    lm = LM.load()
    tok = lm.tokenizer
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "next_tokens").mkdir(exist_ok=True)
    pool_file = RES / "uninformative.json"
    if pool_file.exists():
        pools = json.loads(pool_file.read_text())
        assert all(t in pools for t in args.tasks), "pools exist for other tasks; extend them first"
    else:
        pools = make_pools(lm, args.tasks)
        pool_file.write_text(json.dumps(pools, indent=1, ensure_ascii=False))

    for task in args.tasks:
        data = simple.dataset(task)
        for half in ("A", "B"):
            log = []
            for wording in WORDINGS:
                directory = OUT / half / "simple" / task
                if done(directory, wording):
                    continue
                start = time.time()
                e4 = load(DATA / "readings" / "heldout" / half / "simple" / task, wording)
                items = [int(i) for i in e4["id_items"]]
                saved = {int(r["item"]): r["prompt"] for r in load_outputs(outputs_path("simple", task, wording))}
                head = [tok.bos_token_id] + tok(instruction(task, wording), add_special_tokens=False).input_ids
                clean, corrupt, same = [], [], 0
                for j, item in enumerate(items):
                    query = tok(QUERY.format(input=data[item]["input"]), add_special_tokens=False).input_ids
                    clean.append(head + query)
                    same += clean[-1] == tok(saved[item]).input_ids
                    corrupt.append([[tok.bos_token_id] + pools[task][wording]["ids"][k][j % N_POOL] + query
                                    for k in KINDS])
                top = []
                for b in range(0, len(items), simple.BATCH):
                    rows = clean[b:b + simple.BATCH] + [c[k] for c in corrupt[b:b + simple.BATCH]
                                                        for k in range(len(KINDS))]
                    n = len(clean[b:b + simple.BATCH])
                    next_tokens, z = lm.next_tokens_batch(rows)
                    zc = z[n:].float().view(n, len(KINDS), *z.shape[1:]).mean(1)
                    for i in range(n):
                        top.append(lm.top_tokens(z[i].float() - zc[i], k=10))
                        item = items[b + i]
                        log.append(dict(wording=wording, item=item, expected=data[item]["output"],
                                        answer_token=lm.first_token(" " + data[item]["output"]),
                                        clean=int(next_tokens[i]),
                                        **{k: int(next_tokens[n + i * len(KINDS) + c]) for c, k in enumerate(KINDS)}))
                save(directory, wording, load(readings_dir("simple", task), wording)["records"],
                     [str(i) for i in items], top)
                print(f"{half} {task} {wording}: {len(items)} prompts, encoding = saved prompt {same}/{len(items)} "
                      f"({time.time() - start:.0f} s)", flush=True)
            if log:
                with gzip.open(RES / "next_tokens" / f"{half}_{task}.jsonl.gz", "at") as f:
                    for row in log:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
