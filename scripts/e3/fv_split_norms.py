"""Norms and directions of the FV parts of scripts/e3/fv_split_check.py, per task (GPU for W_O).

Per task: |FV| of every set (sum of the heads' task-mean outputs through W_O, from
data/estimates/fv_task_means.npz), the cosine of the generic6 and task24 parts, and the mean
cosine of each part across task pairs. Writes results/davidson_fv_split/norms.json.

    .venv/bin/python scripts/e3/fv_split_norms.py 2
"""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / 'scripts'))
from lth import gpu; gpu.pin(int(sys.argv[1]))
import numpy as np, torch
from lth.model import LM
from davidson_fv import function_vector
m=np.load(R / 'data/estimates/fv_task_means.npz'); sets=json.load(open(R / 'results/davidson_fv_split/sets.json'))
lm=LM.load(); tasks=sorted({k.split('/')[0] for k in m.files})
res={}
for t in tasks:
    z=torch.tensor(m[f'{t}/task']); v={n: function_vector(lm,[tuple(x) for x in h], z) for n,h in sets.items()}
    res[t]={n: float(v[n].norm()) for n in v}
    res[t]['cos_generic_task24']=float(torch.cosine_similarity(v['generic6'],v['task24'],0))
    res[t]['g']=v['generic6'].cpu(); res[t]['t']=v['task24'].cpu()
print('task              |all30| |generic6| |task24| cos(g,t)')
for t in tasks: print(f"{t:17s} {res[t]['all30']:6.1f} {res[t]['generic6']:9.1f} {res[t]['task24']:8.1f} {res[t]['cos_generic_task24']:7.2f}")
import itertools
cg=np.mean([float(torch.cosine_similarity(res[a]['g'],res[b]['g'],0)) for a,b in itertools.combinations(tasks,2)])
ct=np.mean([float(torch.cosine_similarity(res[a]['t'],res[b]['t'],0)) for a,b in itertools.combinations(tasks,2)])
print(f'generic6 FV across tasks: mean cosine {cg:.2f}; task24 FV across tasks: {ct:.2f}')
json.dump(dict(tasks={t: {k: v for k, v in r.items() if k not in ("g", "t")} for t, r in res.items()},
               generic6_across_tasks=float(cg), task24_across_tasks=float(ct)),
          open(R / 'results/davidson_fv_split/norms.json', 'w'), indent=1)
