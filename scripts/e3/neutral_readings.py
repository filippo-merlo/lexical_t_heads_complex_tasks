"""Head readings on unrelated text: N WikiText-103 prefixes (train split, headings and empty
lines left out), each cut before a space to a token length drawn from the lengths of the
simple tasks' rule identification prompts; raw text with BOS, last token, every head's top-10
under both readouts, in Yang-style masked batches of 20 (as scripts/build_readings.py)."""
import json, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from lth import gpu, simple
from lth.store import load, readings_dir

N, SEED, GPU = 1000, 0, int(sys.argv[1])
OUT = Path("data/readings/neutral"); OUT.mkdir(parents=True, exist_ok=True)
gpu.pin(GPU)
from lth.model import LM
import datasets
lm = LM.load()
lengths = []
for t in simple.TASKS:
    for w in simple.WORDINGS["rule"]:
        prompts = simple.prompts(t, w)[0]
        items = [int(i) for i in load(readings_dir("simple", t), w)["id_items"]]
        lengths += [len(x) for x in simple.token_ids(lm.tokenizer, [prompts[i] for i in items])]
rng = random.Random(SEED)
lines = [t.strip() for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")["text"]]
lines = [t for t in lines if t and not t.startswith("=")]
rows, texts = [], []
for text in rng.sample(lines, 5 * N):
    if len(rows) == N:
        break
    target = rng.choice(lengths)  # with BOS
    for cut in [i for i, ch in enumerate(text) if ch == " "]:
        ids = lm.tokenizer(text[:cut]).input_ids
        if len(ids) == target:
            rows.append(ids); texts.append(text[:cut]); break
        if len(ids) > target:
            break
assert len(rows) == N, len(rows)
top = []
for b in range(0, N, simple.BATCH):
    _, z = lm.next_tokens_batch(rows[b:b + simple.BATCH])
    top += [lm.top_tokens(r).numpy() for r in z]
np.savez(OUT / "wikitext.npz", top=np.stack(top).astype(np.int32))
(OUT / "wikitext.json").write_text(json.dumps([{"text": t, "n_tokens": len(r)} for t, r in zip(texts, rows)], ensure_ascii=False))
print("saved", np.stack(top).shape, "length percentiles", np.percentile([len(r) for r in rows], [5, 50, 95]), "vs task", np.percentile(lengths, [5, 50, 95]))
