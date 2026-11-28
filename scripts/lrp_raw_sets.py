"""LRP-T shared with raw scores averaged over tasks (Pham et al.'s way of combining tasks) at every size (user
2026-11-28, after scripts/pham_frame_sets.py found it ahead of the rank mean at 20 heads). No GPU.

Per fold, each task's LRP-T score (``results/rebuild/additions/fold<f>/lrp_features.npz``, ``<task>/T``: positive
AttnLRP relevance of the last position's attention to the task-word segment, averaged over the choosing half's pool
prompts) is averaged over the 14 tasks as it is, not ranked first; the set is the top N of that mean
(``rebuild_sets`` ties: lower layer, then head). Named ``shared_lrpTraw<N>`` with ``N_RANDOM`` random sets of the
same layer counts (``rebuild_sets.random_sets``, key ``lrp_raw_n/<fold>/shared/<name>``), listed under every task.
The 20-head set was patched in ``results/rebuild/pham_frame`` and is not written again.

Checks before anything is written: (a) the top 20 equal ``shared_lrpTraw20`` of ``results/rebuild/pham_frame``;
(b) the same scores with within-task ranks averaged give the stored ``shared_lrpT<N>`` at every size.

Writes ``results/rebuild/lrp_raw_n/fold<f>/heads.json``. Patched by ``scripts/lrp_raw_patch.sh``, reported by
``scripts/rebuild_report.py --dir lrp_raw_n --stage raw``.

    .venv/bin/python scripts/lrp_raw_sets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lth import RESULTS, simple  # noqa: E402
from rebuild_sets import FOLDS, N_RANDOM, OUT as ADD, SIZES, order_of, random_sets, rankdata, ranks, top  # noqa: E402

R = RESULTS / "rebuild"
OUT = R / "lrp_raw_n"
NEW = tuple(n for n in SIZES if n != 20)


def stored(path: Path, name: str) -> list[list[int]]:
    sets = json.loads(path.read_text())["simple"][simple.TASKS[0]]["sets"]
    (heads,) = [hs for _, nm, hs in sets if nm == name]
    return heads


def main() -> None:
    for fold in FOLDS:
        z = np.load(ADD / f"fold{fold}" / "lrp_features.npz")
        score = {t: z[f"{t}/T"] for t in simple.TASKS}
        raw = order_of(ranks(np.mean([score[t] for t in simple.TASKS], axis=0)))
        rank = order_of(rankdata(np.mean([ranks(score[t]) for t in simple.TASKS], axis=0), method="average"))
        assert top(raw, 20) == stored(R / "pham_frame" / f"fold{fold}" / "heads.json", "shared_lrpTraw20"), f"(a) fold {fold}"
        for n in SIZES:
            assert top(rank, n) == stored(ADD / f"fold{fold}" / "heads.json", f"shared_lrpT{n}"), f"(b) fold {fold} N={n}"
        entries, common = [], {}
        for n in NEW:
            heads, name = top(raw, n), f"shared_lrpTraw{n}"
            entries += [["rule", name, heads]] + [["rule", f"random_{name}_{i}", r]
                                                  for i, r in enumerate(random_sets(heads, f"lrp_raw_n/{fold}/shared/{name}"))]
            common[n] = len(set(map(tuple, heads)) & set(map(tuple, top(rank, n))))
        assert len(entries) == len(NEW) * (N_RANDOM + 1)
        d = OUT / f"fold{fold}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "heads.json").write_text(json.dumps({"simple": {t: {"sets": entries} for t in simple.TASKS}}))
        print(f"fold {fold}: checks (a) and (b) pass; heads in common with the rank-mean LRP-T set: "
              + ", ".join(f"{n}: {c}/{n}" for n, c in common.items()), flush=True)


if __name__ == "__main__":
    main()
