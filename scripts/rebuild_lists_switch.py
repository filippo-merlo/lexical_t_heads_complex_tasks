"""Switch of the rebuilt design's word lists to the smooth TF-IDF rule (user 2026-11-22; ``rebuild_sets.LIST_RULE``):
archive every result that depends on the lists, keep the rows that do not. No GPU.

The lists feed D, the attention segments (task words, and through their priority the others), hence the
combination screen, Selected and the screen candidates. Yang terms, Causal and Causal - generic do not use
them: their sets are checked identical before and after the switch, and their rows are kept. Patching the
same set with the same options gives the same rows (checked: the TF-IDF DBTM set patched twice, and
``rebuild_lists_rerun.sh check``).

Archive: ``results/rebuild/archive/share_lists/<same path>`` (the share ranking with the 50 most shared
WikiText words removed, used until 2026-11-21).

Steps, in order (``scripts/rebuild_lists_rerun.sh`` runs the GPU parts in between):

- ``stage2-archive``: additions' sets, lists, features, generic heads, screen, Selected and its runs,
  reports; then ``scripts/rebuild_sets.py`` rebuilds the sets with the new lists.
- ``stage2-split``: checks every set that does not depend on the lists unchanged (and the generic heads);
  splits each ``s2-<task>`` part into its rows of Yang terms (J) and Causal - generic (kept in place) and the
  rest (archived with the original); copies the quick and full passes of the screen of these lists, run
  before the switch in ``results/rebuild/tfidf_screen`` with the same options, after checking that every
  screen set and feature is the same.
- ``stage3-archive``: both stage-3 pools' sets and the stage-3 reports, screens and Selected; then
  ``scripts/rebuild_sets_s3.py`` for both pools.
- ``stage3-split``: checks the pools' list-independent sets unchanged; splits the parts of 3.1, 3.3, 3.5a
  (stage-2 sets) and 3.2, 3.4, 3.5b (re-identified) the same way; archives the screens of 3.2 and 3.4 and the
  3.1 Selected supplement in full.
- ``newtasks-archive``: the unseen-task sets and runs (re-run in full).

    .venv/bin/python scripts/rebuild_lists_switch.py stage2-archive
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import gzip
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lth import RESULTS, simple  # noqa: E402

R = RESULTS / "rebuild"
ARCHIVE = R / "archive" / "share_lists"
FOLDS = (1, 2)
KEEP = re.compile(r"^(random_)?(shared_)?(yangJ|causal|causalnogen)\d+(_\d+)?$")
DROP = re.compile(r"^(random_)?(shared_)?(D\d+|selected\d+|screen_[A-Z]+)(_\d+)?$")
LIST_DEPENDENT_SET = re.compile(r"^(random_)?(shared_)?(D\d+|selected\d+|screen_[A-Z]+)(_\d+)?$")
SET_FILES = ("heads.json", "word_lists.json", "features.npz", "sets.csv", "screen.json", "generic.json")
SPLITS = {"wt": ("scenarios/wordings_transfer", "s3wt"), "sum": ("scenarios/summed", "s3sum"),
          "et": ("scenarios/examples_transfer", "s3et"), "wq": ("wordings", "s3w-test"),
          "eq": ("examples", "s3e-test"), "esum": ("examples", "s3e-sum")}


def archive(path: Path, copy: bool = False) -> None:
    dest = ARCHIVE / path.relative_to(R)
    assert not dest.exists(), f"already archived: {dest}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    (shutil.copy2 if copy else shutil.move)(str(path), str(dest))


def archived(path: Path) -> Path:
    return ARCHIVE / path.relative_to(R)


def part_files(folder: Path, prefix: str) -> list[Path]:
    """``patching_simple.part-<prefix>-<task>.csv`` for the 14 tasks exactly (not ``<prefix>-sel40-...``)."""
    out = [folder / f"patching_simple.part-{prefix}-{t}.csv" for t in simple.TASKS]
    assert all(p.exists() for p in out), [p for p in out if not p.exists()]
    return out


def split_part(csv_path: Path) -> tuple[int, int]:
    """Archive the part (CSV and answers) and write back only its unpatched rows and list-independent sets."""
    answers = csv_path.parent / "answers" / (csv_path.stem + ".jsonl.gz")
    assert answers.exists(), answers
    rows = list(csv.DictReader(csv_path.open()))
    fields = list(rows[0])
    for r in rows:
        assert r["source"] == "none" or KEEP.match(r["heads"]) or DROP.match(r["heads"]), (csv_path, r["heads"])
    keep = [r for r in rows if r["source"] == "none" or KEEP.match(r["heads"])]
    with gzip.open(answers, "rt") as f:
        lines = [json.loads(x) for x in f]
    for x in lines:
        assert x["source"] == "none" or KEEP.match(x["heads"]) or DROP.match(x["heads"]), (answers, x["heads"])
    keep_lines = [x for x in lines if x["source"] == "none" or KEEP.match(x["heads"])]
    archive(csv_path)
    archive(answers)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
    with gzip.open(answers, "wt") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in keep_lines)
    return len(keep), len(rows)


def archive_part(csv_path: Path) -> None:
    archive(csv_path)
    archive(csv_path.parent / "answers" / (csv_path.stem + ".jsonl.gz"))


def check_sets_unchanged(folder: Path) -> int:
    """Every set whose name does not depend on the lists is the same in the archived and the new heads.json."""
    old = json.loads(archived(folder / "heads.json").read_text())["simple"]
    new = json.loads((folder / "heads.json").read_text())["simple"]
    assert set(old) == set(new), folder
    n = 0
    for t in old:
        o = {nm: hs for _, nm, hs in old[t]["sets"] if not LIST_DEPENDENT_SET.match(nm)}
        m = {nm: hs for _, nm, hs in new[t]["sets"] if not LIST_DEPENDENT_SET.match(nm)}
        assert o == m, (folder, t, sorted(set(o) ^ set(m))[:5], [k for k in o if k in m and o[k] != m[k]][:5])
        n += len(o)
    old_g, new_g = (json.loads(p.read_text()) for p in (archived(folder / "generic.json"), folder / "generic.json"))
    assert old_g == new_g, (folder, "generic heads changed")
    return n


def stage2_archive() -> None:
    add = R / "additions"
    for name in ("s2_per_task.csv", "s2_summary.csv", "selected.json"):
        archive(add / name)
    for f in FOLDS:
        d = add / f"fold{f}"
        for name in SET_FILES:
            archive(d / name)
        for p in part_files(d, "s2sel"):
            archive_part(p)
    print("stage2-archive: additions' sets, reports, Selected and its runs archived; now run scripts/rebuild_sets.py")


def stage2_split() -> None:
    add, ts = R / "additions", R / "tfidf_screen"
    for f in FOLDS:
        d = add / f"fold{f}"
        n = check_sets_unchanged(d)
        print(f"fold {f}: {n} list-independent set entries unchanged, generic heads unchanged", flush=True)
        new = json.loads((d / "heads.json").read_text())["simple"]
        scr = json.loads((ts / f"fold{f}" / "heads.json").read_text())["simple"]
        for t in simple.TASKS:
            a = {nm: hs for _, nm, hs in new[t]["sets"] if nm.startswith(("screen_", "shared_screen_"))}
            b = {nm: hs for _, nm, hs in scr[t]["sets"] if nm.startswith(("screen_", "shared_screen_"))}
            assert a == b and len(a) == 1022, (f, t)
        za, zb = np.load(d / "features.npz"), np.load(ts / f"fold{f}" / "features.npz")
        for k in zb.files:
            assert np.array_equal(za[k], zb[k]), (f, k)
        print(f"fold {f}: all 1022 screen sets per task and every feature equal to results/rebuild/tfidf_screen", flush=True)
        kept = total = 0
        for p in part_files(d, "s2"):
            k, n = split_part(p)
            kept, total = kept + k, total + n
        print(f"fold {f}: s2 parts split, {kept} of {total} rows kept (Yang terms (J), Causal - generic, unpatched)", flush=True)
        for prefix in ("quick-s2", "s2-screen"):
            for t in simple.TASKS:
                for src in (ts / f"fold{f}" / f"patching_simple.part-{prefix}-{t}.csv",
                            ts / f"fold{f}" / "answers" / f"patching_simple.part-{prefix}-{t}.jsonl.gz"):
                    dst = d / src.relative_to(ts / f"fold{f}")
                    assert not dst.exists(), dst
                    shutil.copy2(src, dst)
                    assert filecmp.cmp(src, dst, shallow=False)
        print(f"fold {f}: quick and full passes of the screen copied from tfidf_screen", flush=True)


def stage3_archive() -> None:
    for d in ("wordings", "examples"):
        for name in ("selected.json", "shortlist.json"):
            if (R / d / name).exists():
                archive(R / d / name)
        for f in FOLDS:
            for name in SET_FILES:
                archive(R / d / f"fold{f}" / name)
    for kind, (d, prefix) in SPLITS.items():
        for name in (f"{prefix}_per_task.csv", f"{prefix}_summary.csv"):
            if (R / d / name).exists():
                archive(R / d / name)
    print("stage3-archive: both pools' sets and Selected, stage-3 reports archived; now run scripts/rebuild_sets_s3.py "
          "--pool wordings and --pool examples")


def stage3_split() -> None:
    for d in ("wordings", "examples"):
        for f in FOLDS:
            n = check_sets_unchanged(R / d / f"fold{f}")
            print(f"{d} fold {f}: {n} list-independent set entries unchanged, generic heads unchanged", flush=True)
    for kind, (d, prefix) in SPLITS.items():
        for f in FOLDS:
            kept = total = 0
            for p in part_files(R / d / f"fold{f}", prefix):
                k, n = split_part(p)
                kept, total = kept + k, total + n
            print(f"{kind} fold {f}: {prefix} parts split, {kept} of {total} rows kept", flush=True)
    for f in FOLDS:
        for d, prefix in (("wordings", "s3w-screen"), ("examples", "s3e-screen")):
            for p in part_files(R / d / f"fold{f}", prefix):
                archive_part(p)
        for p in sorted((R / "scenarios" / "wordings_transfer" / f"fold{f}").glob("patching_simple.part-s3wt-sel40-*.csv")):
            archive_part(p)
    print("stage3-split: 3.2 and 3.4 screens and the 3.1 Selected supplement archived")


def newtasks_archive() -> None:
    d = R / "newtasks"
    for name in ("per_task.csv", "repeated.csv", "summary.csv"):
        archive(d / name)
    for f in FOLDS:
        archive(d / f"fold{f}" / "heads.json")
        for p in sorted((d / f"fold{f}").glob("patching_simple.part-nt-*.csv")):
            archive_part(p)
    print("newtasks-archive: unseen-task sets, runs and reports archived; now run scripts/rebuild_newtask_sets.py")


def main() -> None:
    steps = {"stage2-archive": stage2_archive, "stage2-split": stage2_split, "stage3-archive": stage3_archive,
             "stage3-split": stage3_split, "newtasks-archive": newtasks_archive}
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=tuple(steps))
    steps[parser.parse_args().step]()


if __name__ == "__main__":
    main()
