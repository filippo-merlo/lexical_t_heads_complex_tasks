"""Every head's mean output z at the last token, per simple task, on the prompts of the Davidson FV
run (``results/davidson_fv_all14_30``): the task mean (its mean items, correct inst1-4 prompts,
as ``scripts/davidson_fv.py``; checked against the saved FV) and the mean over the 60 corrupted
prompts of its CIE (``scripts/davidson_fv_diff.py``).

Writes ``data/estimates/fv_task_means.npz``: ``{task}/task`` and ``{task}/corrupt``, ``[n_layers, d_source]``
float32.

    .venv/bin/python scripts/e3/fv_task_means.py 2
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from lth import DATA, RESULTS, gpu, simple  # noqa: E402

gpu.pin(int(sys.argv[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from davidson_fv import function_vector  # noqa: E402
from davidson_fv_diff import corrupted_rows, task_mean  # noqa: E402
from lth.model import LM  # noqa: E402

run = RESULTS / "davidson_fv_all14_30"
plan = json.loads((run / "items.json").read_text())
corrupted = json.loads((run / "uninformative.json").read_text())
fv_saved = np.load(run / "fv.npz")
heads = [(l, h) for l, h, _ in json.loads((run / "heads.json").read_text())["heads"]]
lm = LM.load()
out = {}
with torch.no_grad():
    for task in plan:
        mean = task_mean(lm, task, plan)
        diff = float(np.abs(function_vector(lm, heads, mean).cpu().numpy() - fv_saved[f"{task}/task"]).max())
        rows, _ = corrupted_rows(lm, task, plan, corrupted)
        z = torch.cat([lm.next_tokens_batch(rows[b:b + simple.BATCH])[1].float().cpu()
                       for b in range(0, len(rows), simple.BATCH)])
        out[f"{task}/task"], out[f"{task}/corrupt"] = mean.numpy(), z.mean(0).numpy()
        print(f"{task}: FV check {diff:.1e}, {len(rows)} corrupted prompts", flush=True)
(DATA / "estimates").mkdir(parents=True, exist_ok=True)
np.savez(DATA / "estimates" / "fv_task_means.npz", **out)
