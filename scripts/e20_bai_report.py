"""E20 step 2 (CPU): where the heads of each E13 set look, in Bai et al.'s token types.

Reads the masses of ``scripts/e20_bai_attention.py`` (choosing half of each fold) and reports, with
no patching and no test-half item:

1. check: the split reproduces E13's stored masses (task words, query, BOS, and rest = the sum of
   self, template, punct, function, content), and ranking heads on the task-word mass reproduces
   E13's T sets;
2. profiles: mean attention mass per category of E13's sets (per task: Causal, Causal - generic,
   DBT, BT, T, D; shared: Causal, Causal - generic, DBT), the fold's generic heads and all heads, at
   N = 40 and 80; five templates pooled, and the four question--answer templates alone;
3. Spearman correlation over the 1,024 heads between each category's mass and Davidson et al.'s
   causal score on the choosing half (per task: the task's score; shared: the score and masses
   averaged over tasks);
4. overlap with Causal-N of a top N ranked on one category's mass (BOS: lowest first, as B).

Means are over a task's prompts, then over tasks, then over the two folds.

    .venv/bin/python scripts/e20_bai_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lth import DATA, RESULTS, simple  # noqa: E402

from e20_bai_attention import CATS  # noqa: E402

HALF = {1: "A", 2: "B"}
N_HEADS = 32
KINDS = ("equiprobable", "wikitext", "other_task")
E13 = {"task": 0, "query": 1, "rest": 2, "bos": 3}
REST = ("self", "template", "punct", "function", "content")
SIZES = (40, 80)
PER_TASK = ("causal{n}", "causal{n}_nogen", "DBT{n}", "BT{n}", "T{n}", "D{n}")
SHARED = ("shared_causal{n}", "shared_causal{n}_nogen", "shared_DBT{n}")
OUT = RESULTS / "e20_bai_attention"


# Bai et al.'s stopword type (punctuation and function words together), and the whole instruction
# (every token outside BOS, query, scaffold and the last token): T without the word list.
DERIVED = {"punct+function": ("punct", "function"), "instruction": ("task", "punct", "function", "content")}


def derived(m: np.ndarray, parts) -> np.ndarray:
    return m[:, [CATS.index(c) for c in parts]].sum(1)


def idx(heads) -> np.ndarray:
    return np.array([l * N_HEADS + h for l, h in heads])


def top(score: np.ndarray, n: int) -> set[int]:
    """Top n by score, ties to the lower layer, then head (E13's rule)."""
    return set(np.argsort(-score, kind="stable")[:n].tolist())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def md(df: pd.DataFrame) -> str:
    """A DataFrame as a Markdown table (no tabulate in the venv)."""
    cols = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    rows = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    rows += ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks, prof, corr, over = [], [], [], []
    for fold, half in HALF.items():
        att_dir = DATA / "readings" / "heldout" / f"attention20_{half}"
        e13_dir = DATA / "readings" / "heldout" / f"attention13_{half}"
        heads = json.loads((RESULTS / "e13_capped" / f"fold{fold}" / "heads.json").read_text())["simple"]
        generic = [tuple(int(x) for x in g[1:].split("H")) for g in
                   json.loads((RESULTS / "e13_capped" / f"fold{fold}" / "generic.json").read_text())["generic"]]
        cie = np.load(RESULTS / "heldout" / f"fold{fold}" / "fv" / "cie.npz")
        causal = {t: np.mean([cie[f"{t}/{k}"] for k in KINDS], axis=0).ravel() for t in simple.TASKS}
        mean_all, mean_qa = {}, {}
        for t in simple.TASKS:
            m = np.load(att_dir / f"{t}.npy").astype(np.float64)        # prompts x 1024 x 8
            tm = np.load(att_dir / f"{t}.templates.npy")
            old = np.load(e13_dir / f"{t}.npy").astype(np.float64)       # prompts x 1024 x 4
            assert m.shape[:2] == old.shape[:2], (t, m.shape, old.shape)
            rest = m[..., [CATS.index(c) for c in REST]].sum(-1)
            checks.append(dict(fold=fold, task=t, prompts=m.shape[0],
                               sum_minus_1=np.abs(m.sum(-1) - 1).max(),
                               **{f"d_{k}": np.abs((rest if k == "rest" else m[..., CATS.index(k)]) - old[..., j]).max()
                                  for k, j in E13.items()}))
            mean_all[t] = m.mean(0)
            mean_qa[t] = m[tm > 0].mean(0)
            sets = {name: idx(h) for _, name, h in heads[t]["sets"]}
            for n in SIZES:  # T ranked on the new task-word mass = E13's T set
                checks[-1][f"T{n}_same"] = top(mean_all[t][:, CATS.index("task")], n) == set(sets[f"T{n}"].tolist())

            # 2. profiles
            groups = {"all heads": np.arange(N_HEADS * 32), "generic heads": idx(generic)}
            for n in SIZES:
                for g in PER_TASK + SHARED:
                    groups[g.format(n=n)] = sets[g.format(n=n)]
            for g, ix in groups.items():
                for templates, mm in (("all five", mean_all[t]), ("question-answer", mean_qa[t])):
                    prof.append(dict(fold=fold, dataset=t, group=g, templates=templates,
                                     **dict(zip(CATS, mm[ix].mean(0)))))

            # 3. correlation with the causal score, per task
            for c in CATS:
                corr.append(dict(fold=fold, scope="per task", task=t, category=c,
                                 rho=spearman(mean_all[t][:, CATS.index(c)], causal[t])))
            for d, parts in DERIVED.items():
                corr.append(dict(fold=fold, scope="per task", task=t, category=d,
                                 rho=spearman(derived(mean_all[t], parts), causal[t])))

            # 4. overlap of a one-category top N with Causal-N
            for n in SIZES:
                ref = set(sets[f"causal{n}"].tolist())
                for c in CATS:
                    s = mean_all[t][:, CATS.index(c)] * (-1 if c == "bos" else 1)
                    over.append(dict(fold=fold, scope="per task", task=t, n=n, ranking=c, overlap=len(top(s, n) & ref)))
                for d, parts in DERIVED.items():
                    over.append(dict(fold=fold, scope="per task", task=t, n=n, ranking=d,
                                     overlap=len(top(derived(mean_all[t], parts), n) & ref)))
                for g in ("DBT", "BT", "T"):
                    over.append(dict(fold=fold, scope="per task", task=t, n=n, ranking=f"E13 {g}",
                                     overlap=len(set(sets[f"{g}{n}"].tolist()) & ref)))

        # shared scope: masses and causal score averaged over tasks
        cs = np.mean([causal[t] for t in simple.TASKS], axis=0)
        ms = np.mean([mean_all[t] for t in simple.TASKS], axis=0)
        sets0 = {name: idx(h) for _, name, h in heads[simple.TASKS[0]]["sets"]}
        for c in CATS:
            corr.append(dict(fold=fold, scope="shared", task="all", category=c, rho=spearman(ms[:, CATS.index(c)], cs)))
            for n in SIZES:
                s = ms[:, CATS.index(c)] * (-1 if c == "bos" else 1)
                over.append(dict(fold=fold, scope="shared", task="all", n=n, ranking=c,
                                 overlap=len(top(s, n) & set(sets0[f"shared_causal{n}"].tolist()))))
        for d, parts in DERIVED.items():
            corr.append(dict(fold=fold, scope="shared", task="all", category=d, rho=spearman(derived(ms, parts), cs)))
            for n in SIZES:
                over.append(dict(fold=fold, scope="shared", task="all", n=n, ranking=d,
                                 overlap=len(top(derived(ms, parts), n) & set(sets0[f"shared_causal{n}"].tolist()))))
        for n in SIZES:
            for g in ("DBT", "BT", "T"):
                over.append(dict(fold=fold, scope="shared", task="all", n=n, ranking=f"E13 {g}",
                                 overlap=len(set(sets0[f"shared_{g}{n}"].tolist()) & set(sets0[f"shared_causal{n}"].tolist()))))

    checks, prof, corr, over = map(pd.DataFrame, (checks, prof, corr, over))
    for name, df in (("checks", checks), ("profiles", prof), ("spearman", corr), ("overlap", over)):
        df.to_csv(OUT / f"{name}.csv", index=False)

    lines = ["# E20: attention of E13's head sets by Bai et al.'s token types", "",
             "Choosing half of each fold; last-token attention mass; means over prompts, tasks, folds. "
             "No patching, no test-half item.", "", "## 1. Checks", "",
             f"- max |sum of the 8 masses - 1| = {checks.sum_minus_1.max():.2e}",
             *[f"- max |{k} - E13 {k}| = {checks[f'd_{k}'].max():.2e}" for k in E13],
             f"- T ranked on the new task-word mass = E13's T set: "
             f"{int(checks[[f'T{n}_same' for n in SIZES]].values.sum())}/{2 * len(checks)} (task x fold x N)", ""]
    for templates in ("all five", "question-answer"):
        p = prof[prof.templates == templates].groupby(["fold", "group"])[list(CATS)].mean().groupby("group").mean()
        for d, parts in DERIVED.items():
            p[d] = p[list(parts)].sum(1)
        order = ["all heads", "generic heads"] + [g.format(n=n) for n in SIZES for g in PER_TASK + SHARED]
        lines += [f"## 2. Mean attention mass per category ({templates} templates)", "",
                  md(p.loc[order].round(3)), ""]
    r = corr.groupby(["scope", "category", "fold"]).rho.mean().groupby(["scope", "category"]).mean().unstack(0)
    lines += ["## 3. Spearman rho over 1,024 heads: category mass vs causal score", "",
              "Per task: mean over tasks and folds; shared: masses and score averaged over tasks, mean of folds.", "",
              md(r.round(3)), ""]
    rt = corr[corr.scope == "per task"].groupby(["task", "category"]).rho.mean().unstack(1)
    lines += ["Per task (mean of folds):", "", md(rt.round(2)), ""]
    o = over.groupby(["scope", "n", "ranking", "fold"]).overlap.mean().groupby(["scope", "n", "ranking"]).mean().unstack([0, 1])
    lines += ["## 4. Heads shared with Causal-N, top N on one category (BOS lowest first)", "",
              "Mean over tasks and folds (shared: mean of folds).", "", md(o.round(1)), ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
