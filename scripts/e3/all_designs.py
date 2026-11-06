import pandas as pd, glob, sys
pd.set_option("display.width",250); pd.set_option("display.max_rows",500)
rows=[]
for d in "word_lists word_lists_5w word_lists_5w_len word_lists_allid word_lists_union random_vs_best k5_union neutral_union task_neutral group_lists cc_sizes attend_decode".split():
    df=pd.concat([pd.read_csv(f) for f in glob.glob(f"results/{d}/patching_simple.part-*.csv")])
    base=df[df.source=="none"].groupby("task").accuracy.first()
    p=df[df.source!="none"].groupby(["heads","task","source","alpha"]).agg(acc=("accuracy","mean"),n=("n_heads","first")).reset_index()
    t=p[p.source=="task"]; c=p[p.source=="control"]
    best=t.loc[t.groupby(["heads","task"]).acc.idxmax()]
    best=best.merge(c[["heads","task","alpha","acc"]].rename(columns={"acc":"ctl"}),on=["heads","task","alpha"],how="left")
    best["base"]=best.task.map(base); best["dir"]=d
    rows.append(best)
B=pd.concat(rows); B.to_csv(sys.argv[1],index=False)
s=B.groupby(["dir","heads"]).agg(ntask=("task","size"),mean=("acc","mean"),ctl=("ctl","mean"),nh=("n","mean")).round(2)
print(s.to_string())
