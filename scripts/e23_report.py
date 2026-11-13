"""E23 report: the 15 best E22 combinations per scope at every size, next to E13's sets.

Accuracies as ``scripts/e13_report.py`` (check 1, alpha in {1, 2} chosen on the other fold, mean over
the five rule means, mean of the two folds, mean of 14 tasks). Sources: E23 (20, 60, 80, 100, 120
heads), E22's all_combos.csv (40 heads, all three runs), E13 (its seven filter combinations at
every size, and the reference sets). The 15 were chosen on the same held-out items at 40 heads
(``results/e23_best_sizes/chosen.csv``), so their curves are optimistic.

Writes ``results/e23_best_sizes/{per_task,summary,curves}.csv``, ``report.md`` and ``tables.tex``, and
``paper/tables/e23_{task,shared}.tex``. Figures: ``scripts/e23_figures.py``.

    .venv/bin/python scripts/e23_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e13_report as e13  # noqa: E402
from e21_report import md  # noqa: E402
from e22_sets import COMBOS  # noqa: E402
import results_tex as rt  # noqa: E402
from lth import RESULTS  # noqa: E402

OUT = RESULTS / "e23_best_sizes"
SIZES = e13.SIZES
SCOPES = {"task": "per task", "shared": "shared"}
REF = {"causal": "Causal", "causal_nogen": "Causal − generic", "fvsum": "FV-sum", "fvsum_nogen": "FV-sum − generic",
       "yang": "Yang terms", "T": "T (E13)", "BT": "BT (E13)", "DBT_nogen": "DBT − generic",
       "random": "random (layer-matched to DBT)"}


TEX_REF = ("Causal", "Causal − generic", "T (E13)", "BT (E13)", "random (layer-matched to DBT)")


def tex_table(both: pd.DataFrame, scope: str) -> str:
    """Rows 1-15 and E13's main references; bold: equal to the best of the column at two decimals."""
    t = both.loc[[i for i in both.index if ". " in i or i in TEX_REF], list(SIZES)]
    best = t.max()
    head = "per-task sets (one set per task)" if scope == "task" else "shared sets (one set for all 14 tasks)"
    rows = [r"\begin{tabular}{@{}l" + "r" * len(SIZES) + "@{}}", r"\toprule",
            r"\multicolumn{" + str(len(SIZES) + 1) + r"}{@{}l}{\textbf{" + head + r"}} \\",
            "Set & " + " & ".join(map(str, SIZES)) + r" \\", r"\midrule"]
    for i, (name, r) in enumerate(t.iterrows()):
        if i and ". " not in name and ". " in t.index[i - 1]:
            rows.append(r"\midrule")
        label = (name.replace("−", "$-$").replace(" (layer-matched to DBT)", "").replace(" (E13)", " (E13)"))
        cells = ["--" if pd.isna(v) else (r"\textbf{%.2f}" % v if round(v, 2) == round(best[n], 2) else "%.2f" % v)
                 for n, v in r.items()]
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(rows)


def main() -> None:
    e13.OUT = OUT
    curves, _ = e13.load()
    per, summ = e13.summarise(e13.crossfit(curves))
    per.to_csv(OUT / "per_task.csv", index=False)
    summ.to_csv(OUT / "summary.csv", index=False)

    chosen = pd.read_csv(OUT / "chosen.csv")
    e22s = pd.read_csv(RESULTS / "e22_all_combos" / "all_combos.csv")
    e13s = pd.read_csv(RESULTS / "e13_capped" / "summary.csv")
    e22p = pd.concat([pd.read_csv(RESULTS / x / "per_task.csv") for x in ("e13_capped", "e21_bai_rules", "e22_all_combos")])
    e13p = pd.read_csv(RESULTS / "e13_capped" / "per_task.csv")
    cols = ["scope", "condition", "n", "check1_task", "check1_control", "fold1", "fold2"]
    s = pd.concat([summ[cols].assign(run="E23"),
                   e22s[e22s.n == 40][cols].assign(run=e22s.run),
                   e13s[e13s.condition.isin(COMBOS) & (e13s.n != 40)][cols].assign(run="E13")], ignore_index=True)
    s = s.merge(chosen.rename(columns={"combo": "condition"})[["scope", "condition", "rank"]], on=["scope", "condition"])
    assert not s.duplicated(["scope", "condition", "n"]).any()
    assert s.groupby(["scope", "condition"]).n.apply(lambda x: tuple(sorted(x)) == tuple(SIZES)).all(), "missing sizes"
    assert s.groupby("scope").condition.nunique().eq(15).all()
    assert summ.tasks.eq(14).all() and e22s[e22s.n == 40].tasks.eq(14).all()
    ref = e13s[e13s.condition.isin(REF) & (e13s.tasks == 14)][cols].assign(run="E13")  # unfillable sets: "--"
    s.to_csv(OUT / "curves.csv", index=False)

    keys = ["scope", "condition", "n"]
    p = pd.concat([per[per.condition.isin(COMBOS)], e22p[e22p.n == 40], e13p[e13p.condition.isin(COMBOS) & (e13p.n != 40)]])
    p = p.drop_duplicates(keys + ["task", "source"]).merge(s[keys], on=keys)
    p = pd.concat([p, e13p[e13p.condition.isin(REF)]])
    groups = {t: g for g, ts in e13.GROUPS.items() for t in ts}

    tex = []
    lines = ["# E23: the 15 best E22 combinations per scope at every size, check 1", "",
             "Held-out items, cross-fitted alpha in {1, 2}, mean of the two folds, mean of 14 tasks. "
             "40 heads: E13/E21/E22; other sizes: E23, except E13's own filter combinations (run there at every size). "
             "The 15 per scope are the best of 79 at 40 heads on these same items (results/e23_best_sizes/chosen.csv), "
             "so their accuracy is optimistic by selection; E13's reference sets were fixed beforehand. "
             "Letters: D decodes task words; B low BOS; T task words; L last token; M template markers; "
             "S punctuation + stopwords; N whole instruction.", ""]
    for scope, label in SCOPES.items():
        a = s[s.scope == scope]
        tab = a.pivot(index="condition", columns="n", values="check1_task")
        tab = tab.loc[a.drop_duplicates("condition").sort_values("rank").condition]
        tab.index = [f"{r}. {c}" for r, c in zip(a.drop_duplicates("condition").sort_values("rank")["rank"], tab.index)]
        r = ref[ref.scope == scope].pivot(index="condition", columns="n", values="check1_task")
        r = r.reindex([k for k in REF if k in r.index])
        r.index = [REF[k] for k in r.index]
        both = pd.concat([tab, r])
        both["peak"] = both[list(SIZES)].max(axis=1)
        both["at"] = both[list(SIZES)].idxmax(axis=1)
        both.index.name = "set"
        tex.append(tex_table(both, scope))
        lines += [f"## {label.capitalize()}: task accuracy by size", "",
                  "Rows 1-15: chosen at 40 heads (rank there). Below: E13 references. peak/at: the best size, "
                  "picked on the test items. --: a set some task cannot fill (too few eligible heads).", "", md(both.round(3)), ""]

        c = a.pivot(index="condition", columns="n", values="check1_control").max(axis=1)
        f = a.assign(gap=(a.fold1 - a.fold2).abs()).groupby("condition").gap.max()
        lines += [f"Largest control accuracy over sizes ({label}): {c.max():.3f} ({c.idxmax()}); "
                  f"largest fold gap: {f.max():.3f} ({f.idxmax()}).", ""]

        pk = both.reset_index()[["set", "at"]].rename(columns={"set": "label", "at": "n"})
        pk["condition"] = [x.split(". ", 1)[1] if ". " in x else {v: k for k, v in REF.items()}[x] for x in pk.label]
        q = p[(p.scope == scope) & (p.source == "task")].merge(pk, on=["condition", "n"])
        assert q.groupby("label").task.nunique().eq(14).all() and q.label.nunique() == len(pk)
        g = q.assign(group=q.task.map(groups)).groupby(["label", "group"]).accuracy.mean().unstack("group")
        g = g.reindex(both.index)[list(e13.GROUPS)]
        g.insert(0, "heads", both["at"])
        g.index.name = "set"
        lines += [f"By task type at each set's peak size ({label})", "", md(g.round(2)), ""]
    (OUT / "report.md").write_text("\n".join(lines))
    (OUT / "tables.tex").write_text("\n".join(tex))
    for scope, t in zip(SCOPES, tex):  # for paper/head-sets-simple
        (rt.TABLES / f"e23_{scope}.tex").write_text("% Generated by scripts/e23_report.py.\n" + t)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
