#!/usr/bin/env bash
# The saved outputs of the answer evaluation (proposal Sec. 5.5) for every task and wording,
# under the generation scheme of each group of tasks (scripts/run_outputs.py), on GPUs 0, 2,
# 3 and 5. Scoring (scripts/score_outputs.py) is run separately. Start it detached:
#   setsid nohup scripts/run_outputs_all.sh > logs/run_outputs_all.log 2>&1 < /dev/null &
# Per-lane logs in logs/outputs/; writes logs/run_outputs_all.done at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/outputs

say() { echo "[$(date '+%F %T')] $*"; }

lane() {  # gpu name args...
    local gpu=$1 name=$2; shift 2
    $PY scripts/run_outputs.py --gpu "$gpu" "$@" > "logs/outputs/$name.log" 2>&1
    say "$name (GPU $gpu) exit $?"
}

lane 0 simple --family simple &
PIDS=($!)
sleep 90
lane 2 set_game --family complex --levels set_game/easy &
PIDS+=($!)
sleep 90
lane 3 operator_tapatan --family complex --levels operator_func/easy tapatan/easy &
PIDS+=($!)
sleep 90
lane 5 nca_lci --family complex --levels noun_class_agreement/easy lexical_category_inference/easy &
PIDS+=($!)
wait "${PIDS[@]}"
touch logs/run_outputs_all.done
