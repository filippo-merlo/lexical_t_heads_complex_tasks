import glob, sys
DIR = sys.argv[1] if len(sys.argv) > 1 else "results/word_lists"
import pandas as pd
new = pd.concat(pd.read_csv(p) for p in glob.glob(DIR + "/patching_simple.part-*.csv"))
old = pd.read_csv("results/patching_simple.csv")
old = old[(old["style"] == "rule") & (old.mean_style == "rule") & old.heads.isin(["logithead", "jhead"])]
old = old.assign(heads="E1_" + old.heads)
d = pd.concat([new[new.source != "none"], old[old.source != "none"]])
d = d[d.task.isin(new.task.unique())]
d["alpha"] = d.alpha.astype(float)
d = d[d.alpha.isin([1, 2, 4, 8])]
m = d.groupby(["task", "heads", "source", "alpha"]).agg(acc=("accuracy", "mean"), n=("n_heads", "first")).reset_index()
t = m.pivot_table(index=["task", "heads", "n"], columns=["source", "alpha"], values="acc")
order = ["E1_logithead", "common_logithead", "specific_logithead", "E1_jhead", "common_jhead", "specific_jhead"]
t = t.reindex(sorted(t.index, key=lambda i: (i[0], order.index(i[1]))))
pd.set_option("display.width", 200)
print(t.round(2).to_string())
