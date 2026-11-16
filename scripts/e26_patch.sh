#!/usr/bin/env bash
# E26 (scripts/e26_sets.py, e26_summed.py): the capped comparison's sets with the summed intervention.
# Protocol of E13: both folds, alphas 1 and 2, the five rule means of the choosing half, control mean
# (random sets: task means only), replies of at most 16 tokens; check 1 only. Per GPU a fixed group of
# tasks. Writes results/e26_summed/fold{1,2}/patching_simple.part-<group>.csv and answers/. Start detached:
#   setsid nohup scripts/e26_patch.sh > logs/e26/patch.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/e26
say() { echo "[$(date '+%F %T')] $*"; }

part() {  # gpu group tasks...
    local gpu=$1 group=$2; shift 2
    local tasks="$*"
    for fold in 1 2; do
        local half; half=$([ "$fold" = 1 ] && echo A || echo B)
        LTH_RESULTS=results/e26_summed/fold$fold $PY scripts/e26_summed.py --gpu "$gpu" --family simple --set-list \
            --same-style --alphas 1 2 --half "$half" --max-new-tokens 16 --task-only random \
            --tasks $tasks --part "$group" > "logs/e26/patch_f${fold}_$group.log" 2>&1
        say "fold $fold patch $group exit $?"
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
touch logs/e26/patch.done
