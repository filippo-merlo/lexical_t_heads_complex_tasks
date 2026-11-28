#!/usr/bin/env bash
# Core heads (per-head credit analysis, 2026-11-28) on the four unseen tasks: core7 + random draws + top 7 of
# shared DBTL / LRP-T / Causal / Causal - generic (scripts/core_unseen_sets.py); options of
# scripts/rebuild_newtasks.sh step 5. Then scripts/core_unseen_report.py.
#   setsid nohup scripts/core_unseen.sh 0 > logs/newtasks_core/run.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/newtasks_core
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=.venv/bin/python
GPU=${1:?gpu}
say() { echo "[$(date '+%F %T')] $*"; }
TASKS=(person-occupation person-sport person-instrument capitalize)
COMMON=(--family simple --set-list --same-style --max-new-tokens 16 --alphas 1 2
        --mean-styles rule --control-same-wording --min-mean-prompts 10)
say "start (GPU $GPU)"
for fold in 1 2; do
    half=$([ "$fold" = 1 ] && echo A || echo B)
    dir=results/rebuild/newtasks_core/fold$fold
    for task in "${TASKS[@]}"; do
        part=core-$task
        [ -f "$dir/patching_simple.part-$part.csv" ] && { say "skip fold $fold $part (done)"; continue; }
        LTH_RESULTS=$dir $PY scripts/patch.py --gpu "$GPU" "${COMMON[@]}" --half "$half" --task-only random \
            --tasks "$task" --part "$part" > "logs/newtasks_core/${part}_f$fold.log" 2>&1
        code=$?
        say "fold $fold $part exit $code"
        [ "$code" = 0 ] || { say "stopped: patch fold $fold $task"; exit 1; }
    done
done
say "core unseen done"
