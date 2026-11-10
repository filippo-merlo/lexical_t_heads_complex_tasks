#!/usr/bin/env bash
# E9 (scripts/e9_sets.py): patching of E8's DB set on the five complex tasks (easy level), the
# complex checks of E2 (scripts/patch.py run_fu): rule style, each rule wording's own mean and the
# control task's orig mean, alphas 1 and 2 only, replies to EOS or 1,024 tokens, flexible score,
# every answer saved. Check 1 (complex): the bare prompts of the first 100 items correct under orig;
# check 2 (complex-wrong): the orig prompts of the first 100 items wrong under it (none on Operator
# Function). One job per task and check, a fixed queue per GPU. Writes
# results/e9_complex_db/patching_{complex,complex-wrong}_<task>-easy.csv and answers/.
# Start detached:
#   setsid nohup scripts/e9_patch.sh > logs/e9/patch.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LTH_RESULTS=results/e9_complex_db
ALPHAS="1 2"
mkdir -p logs/e9
say() { echo "[$(date '+%F %T')] $*"; }

run() {  # gpu family level
    local gpu=$1 family=$2 level=$3
    $PY scripts/patch.py --gpu "$gpu" --family "$family" --levels "$level" --set-list --alphas $ALPHAS \
        > "logs/e9/patch_${family}_${level%%/*}.log" 2>&1
    say "$family $level exit $?"
}

queue() {  # gpu (family level)...
    local gpu=$1; shift
    while [ $# -gt 0 ]; do run "$gpu" "$1" "$2"; shift 2; done
}

queue 0 complex set_game/easy complex-wrong set_game/easy &
PIDS=($!)
queue 2 complex tapatan/easy complex-wrong tapatan/easy &
PIDS+=($!)
queue 3 complex noun_class_agreement/easy complex-wrong noun_class_agreement/easy complex operator_func/easy &
PIDS+=($!)
queue 5 complex lexical_category_inference/easy complex-wrong lexical_category_inference/easy &
PIDS+=($!)
wait "${PIDS[@]}"
say "all done"
touch logs/e9/patch.done
