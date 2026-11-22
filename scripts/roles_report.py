"""Head roles from readings: classes across tasks and agreement between halves
(paper/head-roles-design.tex, stage 1). Reads ``results/roles/readings/scores.npz``
(scripts/roles_readings.py) and writes ``head_roles.csv``, ``head_classes.csv``,
``summary.json`` and ``pairs.csv`` next to it.

A head has role r in task t when its score is at least TAU (rate minus control). Across the
tasks where r can be measured:
  constant  r in at least 12 of 14 tasks (85%)
  broad     r in tasks of at least two families, fewer than 12
  family    r in two or more tasks, all of one family
  single    r in one task only
A head's primary role in a task is its highest-scoring role there (if >= TAU). A head with
different primary roles in different tasks is "multi-role".

    .venv/bin/python scripts/roles_report.py [--tau 0.3] [--control] [--sets specific|shared]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import RESULTS  # noqa: E402

DIR = RESULTS / "roles" / "readings"
FAMILIES = {"antonym": "word", "synonym": "word", "singular-plural": "morph", "present-past": "morph",
            "country-capital": "facts", "country-currency": "facts", "landmark-country": "facts",
            "park-country": "facts", "product-company": "facts", "next_item": "seq", "prev_item": "seq",
            "english-french": "transl", "english-german": "transl", "english-spanish": "transl"}
ITEM_ROLES = ("answer", "query", "answer_letter", "query_letter")


def lab(h: int) -> str:
    return f"L{h // 32}H{h % 32}"


def classify(tasks_with: list[str], measurable: int) -> str:
    if not tasks_with:
        return ""
    fams = {FAMILIES[t] for t in tasks_with}
    if len(tasks_with) >= int(np.ceil(0.85 * measurable)):
        return "constant"
    if len(fams) >= 2:
        return "broad"
    return "family" if len(tasks_with) >= 2 else "single"


def extra_stats(S, roles, tasks, tau, rows, class_rows, half) -> dict:
    """Counts quoted in the design doc: heads per role and task, constant heads, merged
    task-level / item-level constancy, layers, role combinations of multi-role heads."""
    X = np.nan_to_num(S, nan=-1.0) >= tau  # role, task, head
    item = [roles.index(r) for r in ITEM_ROLES]
    task_level = [i for i in range(len(roles)) if i not in item]
    out = {"heads_per_task": {r: dict(zip(tasks, X[ri].sum(-1).tolist())) for ri, r in enumerate(roles)},
           "max_heads_in_a_task": {r: int(X[ri].sum(-1).max()) for ri, r in enumerate(roles)}}
    out["constant_heads"] = {r: [x["head"] for x in rows if x["half"] == half and x["role"] == r
                                 and x["class"] == "constant"] for r in roles}
    for name, idx in (("task_level", task_level), ("item_level", item)):
        n = X[idx].any(0).sum(0)  # tasks per head
        heads = np.nonzero(n >= 12)[0]
        out[f"{name}_in_12_of_14"] = {lab(h): int(n[h]) for h in heads[np.argsort(-n[heads], kind="stable")]}
    n_any = X.any(0).sum(0)
    out["any_role_in_14"] = int((n_any == 14).sum())
    out["any_role_in_12_or_more"] = int((n_any >= 12).sum())
    out["heads_by_tasks_with_a_role"] = {int(k): int((n_any == k).sum()) for k in range(1, 15)}
    both = X[task_level].any(0) & X[item].any(0)  # task, head
    t_any, i_any = X[task_level].any(0).any(0), X[item].any(0).any(0)
    out["heads_with_an_item_level_role"] = int(i_any.sum())
    out["heads_with_a_task_level_role"] = int(t_any.sum())
    out["heads_with_item_and_task_roles"] = int((t_any & i_any).sum())
    out["of_which_both_in_one_task"] = int((t_any & i_any & both.any(0)).sum())
    out["layers"] = {}
    for r in roles:
        L = [x["layer"] for x in rows if x["half"] == half and x["role"] == r]
        if L:
            out["layers"][r] = {"heads": len(L), "median": float(np.median(L)),
                                "q25": float(np.percentile(L, 25)), "q75": float(np.percentile(L, 75))}
    mine = [x for x in class_rows if x["half"] == half]
    combos = collections.Counter(x["primary_roles"] for x in mine if len(x["primary_roles"].split()) > 1)
    out["role_combinations_of_multi_role_heads"] = dict(combos.most_common(8))
    same = 0
    for x in mine:
        pt = [y.split(":") for y in x["per_task"].split()]
        if len({b for _, b in pt}) < 2:
            continue
        fam = collections.defaultdict(set)
        for a, b in pt:
            fam[FAMILIES[a]].add(b)
        same += all(len(v) == 1 for v in fam.values())
    out["multi_role_heads_with_one_role_per_family"] = same
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--control", action="store_true", help="read the failure control's scores")
    parser.add_argument("--sets", choices=("specific", "shared"), default="specific",
                        help="token sets of scripts/roles_readings.py --sets")
    args = parser.parse_args()
    tau = args.tau
    name = ("scores" if args.sets == "specific" else "scores_shared") + ("_control" if args.control else "")
    z = np.load(DIR / f"{name}.npz")
    meta = json.loads((DIR / "meta.json").read_text())
    S, tasks, roles, halves = z["scores"], meta["tasks"], meta["roles"], meta["halves"]
    suffix = (("" if args.sets == "specific" else "_shared") + ("" if tau == 0.3 else f"_tau{tau}")
              + ("_control" if args.control else ""))
    summary = {"tau": tau, "halves": {}}
    rows, class_rows = [], []
    for hi, half in enumerate(halves):
        other = 1 - hi
        hs = {}
        for ri, r in enumerate(roles):
            meas = [ti for ti in range(len(tasks)) if not np.isnan(S[hi, ri, ti]).all()]
            counts = {"constant": 0, "broad": 0, "family": 0, "single": 0}
            kept = held = 0
            for h in range(1024):
                tw = [tasks[ti] for ti in meas if S[hi, ri, ti, h] >= tau]
                c = classify(tw, len(meas))
                if not c:
                    continue
                counts[c] += 1
                ok = [S[other, ri, tasks.index(t), h] >= tau for t in tw]
                kept += len(ok)
                held += sum(ok)
                rows.append({"half": half, "head": lab(h), "layer": h // 32, "role": r, "class": c,
                             "n_tasks": len(tw), "tasks": " ".join(tw),
                             "families": " ".join(sorted({FAMILIES[t] for t in tw})),
                             "mean_score": round(float(np.mean([S[hi, ri, tasks.index(t), h] for t in tw])), 3),
                             "held_other_half": round(float(np.mean(ok)), 3)})
            hs[r] = {"heads": sum(counts.values()), **counts, "measurable_tasks": len(meas),
                     "role_task_pairs": kept, "held_on_other_half": round(held / kept, 3) if kept else None}
        # primary roles and multi-role heads
        Sh = np.nan_to_num(S[hi], nan=-1.0)  # role, task, head
        best = Sh.argmax(0)                  # task, head
        has = Sh.max(0) >= tau
        n_one = n_multi = n_item_task = 0
        for h in range(1024):
            prim = {roles[best[ti, h]] for ti in range(len(tasks)) if has[ti, h]}
            if not prim:
                continue
            if len(prim) == 1:
                n_one += 1
            else:
                n_multi += 1
                if prim & set(ITEM_ROLES) and prim - set(ITEM_ROLES):
                    n_item_task += 1
            class_rows.append({"half": half, "head": lab(h), "layer": h // 32,
                               "primary_roles": " ".join(sorted(prim)),
                               "per_task": " ".join(f"{t}:{roles[best[ti, h]]}" for ti, t in enumerate(tasks) if has[ti, h]),
                               "generic": round(float(z["generic"][hi, h]), 3)})
        summary["halves"][half] = {"roles": hs, "heads_with_a_role": n_one + n_multi, "one_primary_role": n_one,
                                   "several_primary_roles": n_multi, "of_which_item_and_task_roles": n_item_task}
        # agreement of the scores between halves (role-task-head triples active in either half)
        corr = {}
        for ri, r in enumerate(roles):
            a, b = S[hi, ri].ravel(), S[other, ri].ravel()
            m = ~np.isnan(a) & ~np.isnan(b) & ((a >= tau) | (b >= tau))
            corr[r] = round(float(np.corrcoef(a[m], b[m])[0, 1]), 3) if m.sum() > 2 else None
        summary["halves"][half]["score_correlation_active"] = corr
        summary["halves"][half].update(extra_stats(S[hi], roles, tasks, tau, rows, class_rows, half))
    for hi, half in enumerate(halves):  # heads that read the same tokens on both tasks of every pair
        sharp = (z["pair_peak"][hi].min(1) >= 0.5).all(0)
        same = np.nonzero(sharp & (z["pair_cos"][hi] >= 0.99).all(0))[0]
        summary["halves"][half]["same_reading_on_all_pairs"] = {
            lab(h): round(float(z["generic"][hi, h]), 3) for h in same}
    pair_rows = []
    for k, (a, b) in enumerate(meta["pairs"]):
        for hi, half in enumerate(halves):
            sharp = (z["pair_peak"][hi, k, 0] >= 0.5) & (z["pair_peak"][hi, k, 1] >= 0.5)
            cos = z["pair_cos"][hi, k]
            low = np.nonzero(sharp & (cos < 0.2))[0]
            pair_rows.append({"pair": f"{a} | {b}", "half": half, "sharp_heads": int(sharp.sum()),
                              "median_cos": round(float(np.median(cos[sharp])), 3),
                              "switching_heads": len(low),
                              "switching_examples": " ".join(lab(h) for h in low[np.argsort(cos[low])][:12])})
    for name, rr in (("head_roles", rows), ("head_classes", class_rows), ("pairs", pair_rows)):
        with open(DIR / f"{name}{suffix}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    (DIR / f"summary{suffix}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    for r in pair_rows:
        print(r)


if __name__ == "__main__":
    main()
