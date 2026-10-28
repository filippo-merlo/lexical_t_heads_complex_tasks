#!/usr/bin/env bash
# The simple tasks end to end: E0 and readings on GPUs 0, 2, 3 and 5, then head
# identification, then the causal check split over the same GPUs (patch_simple_split.sh,
# which also merges the parts and rebuilds the E0 table and the Results tables and
# figures). Start it detached:
#   setsid nohup scripts/run_simple_all.sh > logs/run_simple_all.log 2>&1 < /dev/null &
# Writes logs/run_simple_all.done at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID

say() { echo "[$(date '+%F %T')] $*"; }

lane() {  # gpu tasks...
    local gpu=$1; shift
    $PY scripts/run_simple.py --gpu "$gpu" --tasks "$@" > "logs/simple_g$gpu.log" 2>&1
    say "E0 GPU $gpu exit $?"
}

lane 0 english-spanish antonym present-past country-currency &
PIDS=($!)
sleep 90
lane 2 english-german synonym country-capital prev_item &
PIDS+=($!)
sleep 90
lane 3 english-french singular-plural landmark-country &
PIDS+=($!)
sleep 90
lane 5 park-country product-company next_item &
PIDS+=($!)
wait "${PIDS[@]}"
$PY scripts/identify_heads.py > logs/identify_heads.log 2>&1; say "identify exit $?"
scripts/patch_simple_split.sh > logs/patch_simple_split.log 2>&1; say "causal check exit $?"
touch logs/run_simple_all.done
