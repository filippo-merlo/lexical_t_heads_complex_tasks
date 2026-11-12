#!/usr/bin/env bash
# E16 (scripts/e16_sets.py): the AHON (LN) counterparts of E13's B, DB, BT, DBT, per task and shared,
# N 20/40/60/80. E13's protocol: both folds, alphas 1 and 2, rule means of the choosing half, control
# mean, replies of at most 16 tokens; check 1 (simple) only. Per GPU a fixed group of tasks.
# Writes results/e16_ahon/fold{1,2}/patching_simple.part-<group>.csv and answers/. Start detached:
#   setsid nohup scripts/e16_patch.sh > logs/e16/patch.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/e16
say() { echo "[$(date '+%F %T')] $*"; }

part() {  # gpu group tasks...
    local gpu=$1 group=$2; shift 2
    local tasks="$*"
    for fold in 1 2; do
        local half; half=$([ "$fold" = 1 ] && echo A || echo B)
        LTH_RESULTS=results/e16_ahon/fold$fold $PY scripts/patch.py --gpu "$gpu" --family simple --set-list \
            --same-style --alphas 1 2 --half "$half" --max-new-tokens 16 \
            --tasks $tasks --part "$group" > "logs/e16/patch_f${fold}_$group.log" 2>&1
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
touch logs/e16/patch.done
