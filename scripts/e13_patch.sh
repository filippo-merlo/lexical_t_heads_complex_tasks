#!/usr/bin/env bash
# E13 (scripts/e13_sets.py, e13_fv_residual.py): the capped grid of paper/head-sets-simple.tex.
# Protocol of E8/E10/E11: both folds, alphas 1 and 2, rule means of the choosing half, control mean
# (random sets: task means only), replies of at most 16 tokens; check 1 (simple) only. Per GPU a
# fixed group of tasks; each lane then runs Davidson et al.'s summed intervention for its tasks.
# Writes results/e13_capped/fold{1,2}/patching_simple.part-<group>.csv, fv_residual.part-<tasks>.csv
# and answers/. Start detached:
#   setsid nohup scripts/e13_patch.sh > logs/e13/patch.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/e13
say() { echo "[$(date '+%F %T')] $*"; }

part() {  # gpu group tasks...
    local gpu=$1 group=$2; shift 2
    local tasks="$*"
    for fold in 1 2; do
        local half; half=$([ "$fold" = 1 ] && echo A || echo B)
        LTH_RESULTS=results/e13_capped/fold$fold $PY scripts/patch.py --gpu "$gpu" --family simple --set-list \
            --same-style --alphas 1 2 --half "$half" --max-new-tokens 16 --task-only random \
            --tasks $tasks --part "$group" > "logs/e13/patch_f${fold}_$group.log" 2>&1
        say "fold $fold patch $group exit $?"
    done
    for fold in 1 2; do
        $PY scripts/e13_fv_residual.py --gpu "$gpu" --fold "$fold" --tasks $tasks > "logs/e13/fv_f${fold}_$group.log" 2>&1
        say "fold $fold fv_residual $group exit $?"
    done
}

part 0 a antonym synonym singular-plural present-past &
PIDS=($!)
sleep 60
part 2 b country-capital country-currency landmark-country park-country &
PIDS+=($!)
part 3 c product-company next_item prev_item &
PIDS+=($!)
part 5 d english-french english-german english-spanish &
PIDS+=($!)
wait "${PIDS[@]}"
say "all parts done"
touch logs/e13/patch.done
