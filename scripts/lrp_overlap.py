"""Overlap of the LRP-T sets (``scripts/lrp_sets.py``) with the other rules' sets, by the measure of
paper/head-sets-simple-design.tex Sec. 7 (``rebuild_figures``'s ``ov_*`` functions, unchanged): the share of heads in
common, a set of one fold against a set of the other (both orders), its layer-matched chance level, and the
normalized overlap q = (s - c) / sqrt((rho_A - c_A)(rho_B - c_B)), rho being each rule's fold-1 vs fold-2 stability.
Sets identified on Yang et al.'s five templates; per task and shared, every size. No GPU.

Writes ``results/rebuild/overlap/lrp_overlap.csv`` and prints q, s and c per scope and size.

    .venv/bin/python scripts/lrp_overlap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rebuild_figures as rf  # noqa: E402

OTHERS = ("selected", "T", "D", "causal", "causalzs")
ALL = ("lrpT",) + OTHERS


def main() -> None:
    rows = []
    for scope in rf.SCOPES:
        for n in rf.SIZES:
            stab = {r: rf.ov_share(("I", scope, r), ("I", scope, r), n, rf.ov_units(scope, stability=True)) for r in ALL}
            for r in ALL:
                if stab[r] is not None:
                    rows.append(dict(kind="stability", scope=scope, n=n, rule_a=r, share=stab[r][0], chance=stab[r][1]))
            a = ("I", scope, "lrpT")
            for r in OTHERS:
                s = rf.ov_share(a, ("I", scope, r), n, rf.ov_units(scope))
                rows.append(dict(kind="rules", scope=scope, n=n, rule_a="lrpT", rule_b=r,
                                 share=None if s is None else s[0], chance=None if s is None else s[1],
                                 q=rf.ov_q(s, stab["lrpT"], stab[r])))
    df = pd.DataFrame(rows)
    rf.OV_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(rf.OV_DIR / "lrp_overlap.csv", index=False)
    for scope in rf.SCOPES:
        print(f"\n== {scope}: normalized overlap q of LRP-T with each rule (share in common s, chance c), by N")
        d = df[(df.scope == scope) & (df.kind == "rules")]
        for r in OTHERS:
            cells = [f"{x.q:5.2f} ({x.share:.2f}/{x.chance:.2f})" if pd.notna(x.q) else "  --" for x in d[d.rule_b == r].itertuples()]
            print(f"{r:10s} " + "  ".join(cells))
        print(f"-- stability rho (chance), by N")
        d = df[(df.scope == scope) & (df.kind == "stability")]
        for r in ALL:
            cells = [f"{x.share:.2f} ({x.chance:.2f})" for x in d[d.rule_a == r].itertuples()]
            print(f"{r:10s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
