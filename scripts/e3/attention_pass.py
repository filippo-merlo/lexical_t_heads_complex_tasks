"""Last-token attention of every head on the simple tasks' rule prompts (five wordings, the
correct prompts stored in the readings, pooled), reduced to masses on: task-word positions
(tokens overlapping a task word in the prompt, outside the query), the query, the rest of the
instruction, and BOS. Task words = Yang et al.'s term list + the task's top-10 CC-pooled
specific words. Prompts run alone (no padding) under eager attention, as lth.headvis.scan."""
import json, re, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, ".")
from lth import gpu, simple
from lth.store import load, readings_dir
from lth.terms import simple_term_lists

gpu.pin(int(sys.argv[1])); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
from lth.model import LM
from lth.headvis import _eager
lm = LM.load(); tok = lm.tokenizer; L = lm.spec.n_layers
spec = json.load(open("results/word_lists_allid/word_lists.json"))["specific"]
words_of = {t: sorted({w.lower() for w in simple_term_lists()[t] if re.fullmatch(r"[A-Za-z]+", w)} | set(spec[t]["jhead"][:10])) for t in simple.TASKS}
rows = {}
def hook(layer):
    def fn(module, args, output):
        rows[layer] = output[1][0, :, -1].float().cpu().numpy()  # [n_heads, T]
        return output[0], None
    return fn
handles = [lm.model.layers[l].self_attn.register_forward_hook(hook(l)) for l in range(L)]
info = {}
try:
    with _eager(lm.model.hf_model), torch.no_grad():
        for t in simple.TASKS:
            data = simple.dataset(t); pat = re.compile(r"\b(" + "|".join(map(re.escape, words_of[t])) + r")\b", re.I)
            masses, n_kw = [], []
            for w in simple.WORDINGS["rule"]:
                prompts = simple.prompts(t, w)[0]
                for item in load(readings_dir("simple", t), w)["id_items"]:
                    text = prompts[int(item)]; q = data[int(item)]["input"]
                    qs = text.rfind(q); qspan = (qs, qs + len(q))
                    enc = tok(text, return_offsets_mapping=True); ids, off = enc.input_ids, enc.offset_mapping
                    kw = [m.span() for m in pat.finditer(text) if not (m.start() < qspan[1] and m.end() > qspan[0])]
                    pos_kw = [i for i, (a, b) in enumerate(off) if i > 0 and any(a < e and b > s for s, e in kw)]
                    pos_q = [i for i, (a, b) in enumerate(off) if i > 0 and a < qspan[1] and b > qspan[0]]
                    pos_rest = [i for i in range(1, len(ids)) if i not in pos_kw and i not in pos_q]
                    lm.model.hf_model(input_ids=torch.tensor([ids], device=lm.device), use_cache=False, logits_to_keep=1)
                    a = np.stack([rows[l] for l in range(L)]).reshape(L * rows[0].shape[0], -1)  # [heads, T]
                    masses.append(np.stack([a[:, pos_kw].sum(1), a[:, pos_q].sum(1), a[:, pos_rest].sum(1), a[:, 0]], 1).astype(np.float16))
                    n_kw.append(len(pos_kw))
            np.save(OUT / f"{t}.npy", np.stack(masses))  # prompts, heads, [task words, query, rest, BOS]
            info[t] = dict(words=words_of[t], n_prompts=len(masses), prompts_without_task_word=int(sum(k == 0 for k in n_kw)))
            print(t, info[t]["n_prompts"], "prompts;", info[t]["prompts_without_task_word"], "without a task word", flush=True)
finally:
    for h in handles: h.remove()
(OUT / "info.json").write_text(json.dumps(info, indent=1))
