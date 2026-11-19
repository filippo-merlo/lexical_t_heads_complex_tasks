"""Rebuilt design (paper/head-sets-simple-design.tex): the shared 40-head sets of Sec. 5, unchanged, listed for
tasks that took no part in identifying them, and the item split of those tasks. No GPU.

Agreed with the user 2026-11-19: the unseen tasks are the four of Todd et al.'s datasets (as released by
Yang et al.) that have Yang et al.'s templates in our format and are not among the 14
(``lth.simple.UNSEEN_TASKS``: person-occupation, person-sport, person-instrument, capitalize). Every set
was identified on the 14 tasks only, so nothing of an unseen task enters the choice of heads or of the
combination; the unseen task only supplies its own means (``scripts/patch.py``, from its identification
half) and its test items. Control tasks: ``lth.simple.CONTROL``. Capitalize is scored with case kept
(``lth.evaluation.CASE_SENSITIVE_TASKS``).

Sets, per fold, copied from ``results/rebuild/additions/fold<f>/heads.json``: shared Selected (DBTML, first
on both folds), the other combinations among the five best shared ones of each fold (``shared_screen_<combo>``),
shared Causal, Causal - generic and D, all at 40 heads. Random sets: the existing ones where the rule has
them (Selected, Causal, Causal - generic, D); for the other combinations ``N_RANDOM`` new draws,
``rebuild_sets.random_sets`` with key ``newtasks/<fold>/shared/<name>``.

Item split: as ``scripts/heldout_split.py`` (seed 0 permutation, halves A and B), added to
``results/heldout/split.json`` for the unseen tasks; the entries of the 14 tasks are checked unchanged.

Writes ``results/rebuild/newtasks/fold<f>/heads.json`` (``scripts/patch.py --set-list`` format).

    .venv/bin/python scripts/rebuild_newtask_sets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from heldout_split import split  # noqa: E402
from lth import ROOT, simple  # noqa: E402
from rebuild_sets import FOLDS, N_RANDOM, OUT, SCREEN_N, random_sets  # noqa: E402

N_BEST = 5
SELECTED = "DBTML"
DEST = OUT.parent / "newtasks"
SPLIT = ROOT / "results" / "heldout" / "split.json"


def combos_used() -> list[str]:
    sel = json.loads((OUT / "selected.json").read_text())
    out = []
    for fold in FOLDS:
        assert sel[str(fold)]["shared"]["combo"] == SELECTED, fold
        for c, _ in sel[str(fold)]["shared"]["top10_other_fold"][:N_BEST]:
            if c not in out:
                out.append(c)
    return out


def extend_split() -> None:
    old = json.loads(SPLIT.read_text())
    new = dict(old)
    for t in simple.UNSEEN_TASKS:
        h = split(len(simple.dataset(t)))
        assert not set(h["A"]) & set(h["B"]) and len(h["A"]) + len(h["B"]) == len(simple.dataset(t))
        assert t not in old or old[t] == h, t
        new[t] = h
    assert all(new[t] == old[t] for t in simple.TASKS)
    if new != old:
        tmp = SPLIT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(new))
        tmp.replace(SPLIT)  # atomic: other runs read this file
    print("split: " + ", ".join(f"{t} {len(new[t]['A'])}/{len(new[t]['B'])}" for t in simple.UNSEEN_TASKS), flush=True)


def main() -> None:
    assert not set(simple.UNSEEN_TASKS) & set(simple.TASKS)
    extend_split()
    combos = combos_used()
    for fold in FOLDS:
        listed = json.loads((OUT / f"fold{fold}" / "heads.json").read_text())["simple"]
        shared = [e for e in listed[simple.TASKS[0]]["sets"] if e[1].startswith(("shared_", "random_shared_"))]
        for t in simple.TASKS[1:]:  # the shared entries are the same under every task
            assert [e for e in listed[t]["sets"] if e[1].startswith(("shared_", "random_shared_"))] == shared, (fold, t)
        by_name = {nm: hs for _, nm, hs in shared}
        assert by_name[f"shared_selected{SCREEN_N}"] == by_name[f"shared_screen_{SELECTED}"], fold
        sets = []
        for rule in (f"shared_selected{SCREEN_N}", f"shared_causal{SCREEN_N}", f"shared_causalnogen{SCREEN_N}", f"shared_D{SCREEN_N}"):
            sets.append(["rule", rule, by_name[rule]])
            sets += [["rule", f"random_{rule}_{i}", by_name[f"random_{rule}_{i}"]] for i in range(N_RANDOM)]
        for c in combos:
            if c == SELECTED:
                continue
            nm = f"shared_screen_{c}"
            sets.append(["rule", nm, by_name[nm]])
            sets += [["rule", f"random_{nm}_{i}", r] for i, r in enumerate(random_sets(by_name[nm], f"newtasks/{fold}/shared/{nm}"))]
        assert all(len(hs) == SCREEN_N for _, _, hs in sets)
        out = DEST / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "heads.json").write_text(json.dumps({"simple": {t: {"sets": sets} for t in simple.UNSEEN_TASKS}}))
        print(f"fold {fold}: {len(sets) // (N_RANDOM + 1)} sets (Selected = {SELECTED}, {len(combos) - 1} other combinations, "
              f"Causal, Causal - generic, D), {N_RANDOM} random each, for {', '.join(simple.UNSEEN_TASKS)}", flush=True)


if __name__ == "__main__":
    main()
