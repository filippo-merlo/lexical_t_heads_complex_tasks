"""Where chosen heads look from the last token, by prompt part, on the simple tasks' inst1-4 prompts.

For each head of ``HEADS`` and each prompt (the first ``N`` correct prompts per template and task,
as in ``data/readings/simple``), the last-token attention summed over: BOS, the instruction's
task words (Yang et al.'s terms and the CC-pooled task words, as ``scripts/e3/attention_pass.py``),
the instruction's other tokens, the template's `` Q`` and its ``:``, the query's input, the newline,
``A`` and the final ``:`` (the last token itself). Eager attention, prompts run alone, as
``attention_pass.py``.

Writes ``results/davidson_fv_diff/head_attention.json`` (mean per head, task and part) and prints
the means over tasks.

    .venv/bin/python scripts/e3/fv_head_attention.py 2
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lth import RESULTS, gpu, simple  # noqa: E402

gpu.pin(int(sys.argv[1]))
import torch  # noqa: E402

from lth.headvis import _eager  # noqa: E402
from lth.model import LM  # noqa: E402
from lth.store import load, readings_dir  # noqa: E402
from lth.terms import simple_term_lists  # noqa: E402

GENERIC = ["L13H31", "L14H11", "L14H29", "L15H5", "L18H22", "L31H14"]
TASK = ["L16H25", "L17H4", "L13H22", "L14H13", "L15H21", "L15H14"]
HEADS = GENERIC + TASK
PARTS = ("BOS", "task words", "other instruction", "Q", "Q-colon", "input", "newline", "A", "last colon")
WORDINGS, N = ("inst1", "inst2", "inst3", "inst4"), 25
sites = [(int(h[1:].split("H")[0]), int(h.split("H")[1])) for h in HEADS]
spec = {t: v["jhead"][:10] for t, v in json.loads((RESULTS / "word_lists_allid/word_lists.json").read_text())["specific"].items()}
words_of = {t: sorted({w.lower() for w in simple_term_lists()[t] if re.fullmatch(r"[A-Za-z]+", w)} | set(spec[t])) for t in simple.TASKS}

lm = LM.load()
tok = lm.tokenizer
rows = {}


def hook(layer):
    def fn(module, args, output):
        rows[layer] = output[1][0, :, -1].float().cpu().numpy()  # [n_heads, T]
        return output[0], None
    return fn


layers = sorted({l for l, _ in sites})
handles = [lm.model.layers[l].self_attn.register_forward_hook(hook(l)) for l in layers]
out = {}
try:
    with _eager(lm.model.hf_model), torch.no_grad():
        for t in simple.TASKS:
            data = simple.dataset(t)
            pat = re.compile(r"\b(" + "|".join(map(re.escape, words_of[t])) + r")\b", re.I)
            masses = []
            for w in WORDINGS:
                prompts = simple.prompts(t, w)[0]
                for item in load(readings_dir("simple", t), w)["id_items"][:N]:
                    text = prompts[int(item)]
                    q_at = text.rfind(" Q: ")
                    inp = (q_at + 4, q_at + 4 + len(data[int(item)]["input"]))
                    enc = tok(text, return_offsets_mapping=True)
                    ids, off = enc.input_ids, enc.offset_mapping
                    assert text.endswith("\nA:"), text
                    kw = [m.span() for m in pat.finditer(text[:q_at])]
                    label = []
                    for i, (a, b) in enumerate(off):
                        piece = text[a:b]
                        if i == 0:
                            label.append("BOS")
                        elif b <= q_at:
                            label.append("task words" if any(a < e and b > s for s, e in kw) else "other instruction")
                        elif i == len(ids) - 1:
                            label.append("last colon")
                        elif piece.strip() == "A" and i == len(ids) - 2:
                            label.append("A")
                        elif "\n" in piece:
                            label.append("newline")
                        elif a < inp[1] and b > inp[0]:
                            label.append("input")
                        elif piece.strip() == "Q":
                            label.append("Q")
                        elif piece.strip() == ":":
                            label.append("Q-colon")
                        else:
                            label.append("other instruction")
                    lm.model.hf_model(input_ids=torch.tensor([ids], device=lm.device), use_cache=False, logits_to_keep=1)
                    lab = np.array(label)
                    masses.append([[float(rows[l][h][lab == p].sum()) for p in PARTS] for l, h in sites])
            out[t] = np.mean(masses, 0).tolist()  # heads x parts
            print(t, len(masses), "prompts", flush=True)
finally:
    for h in handles:
        h.remove()
(RESULTS / "davidson_fv_diff" / "head_attention.json").write_text(json.dumps(dict(heads=HEADS, parts=PARTS, tasks=out), indent=1))
mean = np.mean([out[t] for t in out], 0)
print("head    " + " ".join(f"{p[:10]:>10s}" for p in PARTS))
for h, row in zip(HEADS, mean):
    print(f"{h:7s} " + " ".join(f"{v:10.2f}" for v in row) + ("   (generic)" if h in GENERIC else ""))
