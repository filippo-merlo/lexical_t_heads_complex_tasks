#!/usr/bin/env bash
# Head identification and both causal checks under the flexible score (2026-10-30). Run
# after scripts/build_readings.py has written data/readings for every simple task and
# complex level. Start it detached:
#   setsid nohup scripts/pipeline_flexible.sh > logs/flex/pipeline.log 2>&1 < /dev/null &
#
#  1. Head identification (results/heads.json, measures.csv, overlaps.csv).
#  2. One job queue on GPUs 0, 2, 3, 5, longest first: the complex checks one level each
#     (check 2 has no Operator Function: no item is answered wrongly under the original
#     rules prompt), and the simple checks in four shares of the tasks each.
#  3. The shares merged into results/patching_<family>.csv and its answers, then the
#     Results tables and figures.
# Model loads are spaced 90 s apart (the node's commit limit fails four at once). Logs in
# logs/flex/; writes logs/flex/pipeline.done at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
GPUS=(0 2 3 5)
LOGS=logs/flex
mkdir -p $LOGS

say() { echo "[$(date '+%F %T')] $*"; }

next_job() {  # pop the first line of queue file $1
    flock "$1.lock" bash -c 'head -n1 "$0"; sed -i 1d "$0"' "$1"
}

run_job() {  # gpu family share-or-level [tasks...]
    local gpu=$1 family=$2 share=$3 log
    shift 3
    if [ $# -gt 0 ]; then
        log=$LOGS/patch_$family.part-$share.log
        $PY scripts/patch.py --gpu "$gpu" --family "$family" --tasks "$@" --part "$share" > "$log" 2>&1
    else
        log=$LOGS/patch_${family}_${share//\//-}.log
        $PY scripts/patch.py --gpu "$gpu" --family "$family" --levels "$share" > "$log" 2>&1
    fi
    say "GPU $gpu: $family $share exit $?"
}

worker() {  # gpu queue
    local job
    while job=$(next_job "$2") && [ -n "$job" ]; do
        flock $LOGS/load.lock sleep 90
        run_job "$1" $job
    done
}

say "stage 1: head identification"
$PY scripts/identify_heads.py > $LOGS/identify.log 2>&1 || { say "identify failed"; exit 1; }

say "stage 2: causal checks"
cat > $LOGS/queue <<'EOF'
complex set_game/easy
complex tapatan/easy
complex-wrong set_game/easy
complex-wrong tapatan/easy
complex operator_func/easy
complex noun_class_agreement/easy
complex lexical_category_inference/easy
complex-wrong noun_class_agreement/easy
complex-wrong lexical_category_inference/easy
simple a antonym synonym singular-plural present-past
simple b country-capital country-currency landmark-country park-country
simple c product-company next_item prev_item
simple d english-french english-german english-spanish
simple-wrong a antonym synonym singular-plural present-past
simple-wrong b country-capital country-currency landmark-country park-country
simple-wrong c product-company next_item prev_item
simple-wrong d english-french english-german english-spanish
EOF
lanes=()
for gpu in "${GPUS[@]}"; do
    worker "$gpu" $LOGS/queue &
    lanes+=($!)
done
wait "${lanes[@]}"

say "stage 3: merge, tables, figures"
$PY - <<'PY'
import gzip, shutil
from pathlib import Path

import pandas as pd

results = Path("results")
for family in ("simple", "simple-wrong", "complex", "complex-wrong"):
    pattern = f"patching_{family}.part-*" if family.startswith("simple") else f"patching_{family}_*"
    parts = sorted(results.glob(pattern + ".csv"))
    answers = sorted((results / "answers").glob(pattern + ".jsonl.gz"))
    if not parts:
        print("no parts for", family)
        continue
    pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(results / f"patching_{family}.csv", index=False)
    with gzip.open(results / "answers" / f"patching_{family}.jsonl.gz", "wb") as out:
        for p in answers:
            with gzip.open(p, "rb") as f:
                shutil.copyfileobj(f, out)
    for p in parts + answers:
        p.unlink()
    print("merged", family, len(parts), "parts")
PY
say "merge exit $?"
$PY scripts/results_tex.py > $LOGS/results_tex.log 2>&1; say "tables exit $?"
touch $LOGS/pipeline.done