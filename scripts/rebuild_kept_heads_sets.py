"""Step 1 diagnostic (user 2026-11-22): the heads common to the old-list and the TF-IDF-list D sets ("kept"), at 40
and 120 heads, for country-capital, present-past, singular-plural, both folds. Sets from the archived and the
new additions heads.json (D<k>, per task). Writes results/rebuild/diagnostics/kept_heads/fold<f>/heads.json."""
import json
from pathlib import Path
R = Path("results/rebuild")
TASKS = ("country-capital", "present-past", "singular-plural")
for f in (1, 2):
    old = json.loads((R / "archive/share_lists/additions" / f"fold{f}" / "heads.json").read_text())["simple"]
    new = json.loads((R / "additions" / f"fold{f}" / "heads.json").read_text())["simple"]
    out = {}
    for t in TASKS:
        o = {nm: hs for _, nm, hs in old[t]["sets"]}; n = {nm: hs for _, nm, hs in new[t]["sets"]}
        sets = []
        for k in (40, 120):
            kept = sorted({tuple(h) for h in o[f"D{k}"]} & {tuple(h) for h in n[f"D{k}"]})
            sets.append(["rule", f"keptD{k}", [list(h) for h in kept]])
            print(f"fold {f} {t} D{k}: kept {len(kept)}, left {k - len(kept)}, entered {k - len(kept)}")
        out[t] = {"sets": sets}
    d = R / "diagnostics" / "kept_heads" / f"fold{f}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "heads.json").write_text(json.dumps({"simple": out}))
