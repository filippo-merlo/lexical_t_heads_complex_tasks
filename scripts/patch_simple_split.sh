#!/usr/bin/env bash
# The simple-task causal check split over GPUs 0, 2, 3 and 5 by task, then merged into
# results/patching_simple.csv and results/answers/patching_simple.jsonl.gz, then the E0
# table and the Results tables and figures. Start it detached:
#   setsid nohup scripts/patch_simple_split.sh > logs/patch_simple_split.log 2>&1 < /dev/null &
# Writes logs/patch_simple_split.done at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID

say() { echo "[$(date '+%F %T')] $*"; }

part() {  # gpu part tasks...
    local gpu=$1 name=$2; shift 2
    $PY scripts/patch.py --gpu "$gpu" --family simple --tasks "$@" --part "$name" > "logs/patch_simple.part-$name.log" 2>&1
    say "part $name exit $?"
}

part 0 a antonym synonym singular-plural present-past &
PIDS=($!)
sleep 90
part 2 b country-capital country-currency landmark-country park-country &
PIDS+=($!)
sleep 90
part 3 c product-company next_item prev_item &
PIDS+=($!)
sleep 90
part 5 d english-french english-german english-spanish &
PIDS+=($!)
wait "${PIDS[@]}"
$PY - <<'PY'
import gzip, shutil
from pathlib import Path

import pandas as pd

results = Path("results")
parts = sorted(results.glob("patching_simple.part-*.csv"))
assert len(parts) == 4, parts
pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(results / "patching_simple.csv", index=False)
with gzip.open(results / "answers" / "patching_simple.jsonl.gz", "wb") as out:
    for p in sorted((results / "answers").glob("patching_simple.part-*.jsonl.gz")):
        with gzip.open(p, "rb") as f:
            shutil.copyfileobj(f, out)
for p in parts + sorted((results / "answers").glob("patching_simple.part-*.jsonl.gz")):
    p.unlink()
print("merged", len(parts), "parts")
PY
say "merge exit $?"
$PY scripts/e0_table.py && $PY scripts/results_tex.py; say "tables exit $?"
touch logs/patch_simple_split.done
