#!/usr/bin/env bash
# A simple-task causal check split over GPUs 0, 2, 3 and 5 by task, then merged into
# results/patching_<family>.csv and results/answers/patching_<family>.jsonl.gz, then the E0
# table and the Results tables and figures. The family is the first argument: simple
# (default) or simple-wrong. Start it detached:
#   setsid nohup scripts/patch_simple_split.sh [family] > logs/patch_<family>_split.log 2>&1 < /dev/null &
# Writes logs/patch_<family>_split.done at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export FAMILY=${1:-simple}

say() { echo "[$(date '+%F %T')] $*"; }

part() {  # gpu part tasks...
    local gpu=$1 name=$2; shift 2
    $PY scripts/patch.py --gpu "$gpu" --family "$FAMILY" --tasks "$@" --part "$name" > "logs/patch_$FAMILY.part-$name.log" 2>&1
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
import gzip, os, shutil
from pathlib import Path

import pandas as pd

results = Path("results")
family = os.environ["FAMILY"]
parts = sorted(results.glob(f"patching_{family}.part-*.csv"))
assert len(parts) == 4, parts
pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(results / f"patching_{family}.csv", index=False)
with gzip.open(results / "answers" / f"patching_{family}.jsonl.gz", "wb") as out:
    for p in sorted((results / "answers").glob(f"patching_{family}.part-*.jsonl.gz")):
        with gzip.open(p, "rb") as f:
            shutil.copyfileobj(f, out)
for p in parts + sorted((results / "answers").glob(f"patching_{family}.part-*.jsonl.gz")):
    p.unlink()
print("merged", len(parts), "parts")
PY
say "merge exit $?"
$PY scripts/e0_table.py && $PY scripts/results_tex.py; say "tables exit $?"
touch "logs/patch_${FAMILY}_split.done"
