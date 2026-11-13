#!/usr/bin/env bash
# E23 (scripts/e23_sets.py): the 15 best E22 combinations per scope (per task, shared) at 20, 60, 80, 100
# and 120 heads (40 run in E13/E21/E22; E13's D/B/T/DB/DT/BT/DBT not rerun), under E13's protocol exactly (scripts/e13_patch.sh): both folds, alphas
# 1 and 2, rule means of the choosing half, control mean, replies of at most 16 tokens, check 1.
# Writes results/e23_best_sizes/fold{1,2}/patching_simple.part-<group>.csv and answers/. Start detached:
#   setsid nohup scripts/e23_patch.sh > logs/e23/patch.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/e23
say() { echo "[$(date '+%F %T')] $*"; }

part() {  # gpu group tasks...
    local gpu=$1 group=$2; shift 2
    for fold in 1 2; do
        local half; half=$([ "$fold" = 1 ] && echo A || echo B)
        LTH_RESULTS=results/e23_best_sizes/fold$fold $PY scripts/patch.py --gpu "$gpu" --family simple --set-list \
            --same-style --alphas 1 2 --half "$half" --max-new-tokens 16 --task-only random \
            --tasks "$@" --part "$group" > "logs/e23/patch_f${fold}_$group.log" 2>&1
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
touch logs/e23/patch.done
