"""Step 1 diagnostic report: old D<k>, new (TF-IDF) D<k> and the kept heads alone; accuracy and share of replies
that name the task (a word of Yang et al.'s term list for the task, whole word, any case), per alpha, mean over
both folds; also the most common replies of the kept sets."""
import csv, gzip, json, re, sys, collections
from pathlib import Path
sys.path.insert(0, ".")
from lth.terms import simple_term_lists
R = Path("results/rebuild")
TASKS = ("country-capital", "present-past", "singular-plural")
terms = simple_term_lists()
def lines(path, names):
    with gzip.open(path, "rt") as f:
        for x in f:
            r = json.loads(x)
            if r.get("source") == "task" and r.get("heads") in names:
                yield r
rows, common = [], {}
for t in TASKS:
    pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in terms[t]) + r")\b", re.I)
    for k in (40, 120):
        src = {"old": (R / "archive/share_lists/additions", f"patching_simple.part-s2-{t}.jsonl.gz", f"D{k}"),
               "new": (R / "additions", f"patching_simple.part-s2-L-{t}.jsonl.gz", f"D{k}"),
               "kept": (R / "diagnostics/kept_heads", f"patching_simple.part-kept-{t}.jsonl.gz", f"keptD{k}")}
        for label, (d, fname, name) in src.items():
            acc = collections.defaultdict(list)
            for f in (1, 2):
                for r in lines(d / f"fold{f}" / "answers" / fname, {name}):
                    a = float(r["alpha"])
                    acc[a].append((bool(r["correct"]), bool(pat.search(r["reply"]))))
                    if label == "kept":
                        common.setdefault((t, k, a), collections.Counter())[r["predicted"].strip()[:30]] += 1
            for a, v in sorted(acc.items()):
                rows.append(dict(task=t, size=k, set=label, alpha=a, n=len(v),
                                 accuracy=round(sum(c for c, _ in v) / len(v), 3),
                                 names_task=round(sum(m for _, m in v) / len(v), 3)))
out = R / "diagnostics" / "kept_heads" / "summary.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for t in TASKS:
    for k in (40, 120):
        print(f"{t} D-{k}: " + " | ".join(
            f"a={a:.0f} " + "  ".join(f"{s} {r['accuracy']:.2f}/{r['names_task']:.2f}" for s in ("old", "new", "kept")
                                       for r in rows if r["task"] == t and r["size"] == k and r["set"] == s and r["alpha"] == a)
            for a in (1.0, 2.0)))
        for a in (1.0, 2.0):
            print(f"     kept a={a:.0f} top replies: {common[(t, k, a)].most_common(4)}")
