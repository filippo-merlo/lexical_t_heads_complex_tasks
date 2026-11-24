"""Repeated replies for the Limitations of paper/head-sets-simple-design (2026-11-24).
Repeated replies (one reply >= half of a condition's answers) in the test conditions of Secs. 5-6, keeping the
latest run of each condition (an -L part replaces the base part); for correct repeated replies, their contribution
to the mean over 14 tasks and 2 folds of the set."""
import gzip, json, glob, re, collections
from pathlib import Path
R = str(Path(__file__).resolve().parents[1] / "results" / "rebuild")
TASKS = ["antonym", "synonym", "singular-plural", "present-past", "country-capital", "country-currency",
         "landmark-country", "park-country", "product-company", "next_item", "prev_item", "english-french",
         "english-german", "english-spanish"]
DIRS = {"scenarios/wordings_transfer": "wt", "scenarios/examples_transfer": "et", "scenarios/summed": "sum",
        "wordings": "wq", "examples": "eq", "transfer_n": None, "additions": "s5"}
for n in (10, 20, 60, 80, 100, 120):
    DIRS[f"examples_n{n}"] = "eq"; DIRS[f"wordings_n{n}"] = "wq"; DIRS[f"shared_n{n}"] = "s5"
SIZED = ("transfer_n", "examples_n", "wordings_n", "shared_n")
SKIP = re.compile(r"^(quick|full|screen|s3w-screen|s3e-screen|s1)")
best = {}  # condition -> (is_L, rows)
for d, test in DIRS.items():
    for f in glob.glob(f"{R}/{d}/fold*/answers/*.jsonl.gz"):
        part = re.search(r"part-(.*)\.jsonl", f).group(1)
        task = next((t for t in TASKS if part.endswith(t)), None)
        if task is None: continue
        kind = part[: -len(task) - 1]
        if SKIP.match(kind) or "screen" in kind: continue
        is_L = kind.endswith("-L"); base = kind[:-2] if is_L else kind
        prio = 2 if d.startswith(SIZED) else int(is_L)
        t2 = test or {"wt": "wt", "et": "et", "tsum": "sum"}[base]
        if "sum" in base and t2 == "eq": t2 = "esum"
        if "sum" in base and t2 == "s5": t2 = "s5sum"
        fold = re.search(r"fold(\d)", f).group(1)
        cond = collections.defaultdict(list)
        for l in gzip.open(f, "rt"):
            r = json.loads(l)
            if r["source"] != "task" or r["heads"].startswith("random") or "screen" in r["heads"]: continue
            cond[(t2, fold, task, r["heads"], r["mean_wording"], str(r["alpha"]))].append(r)
        for k, rs in cond.items():
            if k not in best or prio > best[k][0]: best[k] = (prio, rs)
nw = collections.Counter((t2, fold, task, h, a) for (t2, fold, task, h, w, a) in best)
agg = collections.defaultdict(float); rep = collections.defaultdict(set); n_rep = 0
for (t2, fold, task, h, w, a), (_, rs) in best.items():
    c = collections.Counter(x["predicted"].strip() for x in rs); top, k = c.most_common(1)[0]
    if k / len(rs) < 0.5: continue
    n_rep += 1
    corr = sum(x["predicted"].strip() == top and x["correct"] for x in rs) / len(rs)
    if corr: agg[(t2, h, a)] += corr / (nw[(t2, fold, task, h, a)] * 14 * 2); rep[(t2, h, a)].add((task, top))
print(n_rep, "conditions with a repeated reply")
for k, v in sorted(agg.items(), key=lambda x: (x[0][0], -x[1])):
    print(f"{k[0]:5s} {k[1]:26s} a={k[2]:3s} +{v:.4f}  {sorted(rep[k])}")
