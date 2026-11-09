"""Tables of scripts/e3/fv_split_check.py (no GPU): check of all30 against the saved FV run,
accuracy / answer words / line-break stops by set and alpha (mean over the 14 tasks), each
set at its best alpha per task (chosen on the same items, as the FV runs report it), the most
common answers at high alpha, and the FV part norms (norms.json). Writes
results/davidson_fv_split/report.md.

    .venv/bin/python scripts/e3/fv_split_report.py
"""
import collections, glob, gzip, json
from pathlib import Path
import pandas as pd


def md(df):  # a DataFrame as a markdown table
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [f"{v:g}" if isinstance(v, float) else str(v) for v in r]) + " |")
    return "\n".join(out)

OUT = Path(__file__).resolve().parents[2] / "results" / "davidson_fv_split"
d = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "patching_simple.part-*.csv")))], ignore_index=True)
saved = pd.read_csv(OUT.parent / "davidson_fv_all14_30" / "patching_simple.csv")
m = d[d.set == "all30"].merge(saved[saved.source != "none"], on=["task", "source", "alpha"], suffixes=("", "_saved"))
x = (m.accuracy - m.accuracy_saved).abs()
order = ["all30", "task24", "drop6_r0", "drop6_r1", "drop6_r2", "generic6"]
t = d[d.source == "task"].copy()
L = ["# Davidson FV split: generic heads vs task heads (check 1)", "",
     f"Check: all30 vs the saved FV run at the same alphas: {(x > 0).sum()}/{len(x)} rows differ, mean |diff| {x.mean():.4f}, "
     f"max {x.max():.2f} (the batches differ with the alpha grid, bf16).", ""]
for col, title in (("accuracy", "Accuracy"), ("mean_words", "Words in the answer"), ("stopped", "Share stopping at a line break")):
    p = t.pivot_table(index="set", columns="alpha", values=col).reindex(order)
    L += [f"## {title}, task means, mean over 14 tasks", "", md(p.round(2)), ""]
c = d[d.source == "control"].pivot_table(index="set", columns="alpha", values="accuracy")
L += ["## Accuracy with the control mean", "", md(c.round(2)), ""]
t["key"] = list(zip(-t.accuracy, t.alpha))
b = t.sort_values("key").drop_duplicates(["task", "set"])
acc = b.pivot(index="task", columns="set", values="accuracy")[order]
al = b.pivot(index="task", columns="set", values="alpha")[["all30", "task24"]].add_prefix("alpha ")
L += ["## Best alpha per task (on the same items)", "", md(pd.concat([acc.round(2), al], axis=1)),
      "", "Mean: " + ", ".join(f"{k} {v:.2f}" for k, v in acc.mean().items()), ""]
rows = [json.loads(l) for p in glob.glob(str(OUT / "answers" / "*.jsonl.gz")) for l in gzip.open(p, "rt")]
L += ["## Most common answers (task means)", ""]
for s, a in (("all30", 1), ("all30", 3), ("generic6", 5), ("task24", 3)):
    cnt = collections.Counter(r["predicted"].strip()[:20] for r in rows if r["set"] == s and r["alpha"] == a and r["source"] == "task")
    L.append(f"- {s}, alpha {a}: " + ", ".join(f"`{k}` {n}" for k, n in cnt.most_common(5)))
n = json.loads((OUT / "norms.json").read_text())
L += ["", "## FV part norms", "", "| task | all30 | generic6 | task24 | cos(generic6, task24) |", "|---|---|---|---|---|"]
for k, v in n["tasks"].items():
    L.append(f"| {k} | {v['all30']:.1f} | {v['generic6']:.1f} | {v['task24']:.1f} | {v['cos_generic_task24']:.2f} |")
L += ["", f"Mean cosine across task pairs: generic6 {n['generic6_across_tasks']:.2f}, task24 {n['task24_across_tasks']:.2f}."]
(OUT / "report.md").write_text("\n".join(L) + "\n")
print("\n".join(L[:4]))
