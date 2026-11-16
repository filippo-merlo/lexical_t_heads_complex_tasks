#!/usr/bin/env bash
# E24 step E (scripts/e24_sets_e.py): the other good filter combinations (12 of 4+ letters and DBS) at 20-120 heads,
# both scopes, with the example mean, and with rule means where E22/E23 never ran them; otherwise E13's protocol exactly
# (scripts/e13_patch.sh): both folds, alphas 1 and 2, control mean (the control task's 1-shot prompts at
# its items correct 1-shot, choosing half; random draws: task means only), replies of at most 16 tokens,
# check 1. Writes results/e24_cross_style/e/fold{1,2}/patching_simple.part-<group>.csv and answers/.
#   setsid nohup scripts/e24_patch_e.sh > logs/e24/patch_e.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/e24
say() { echo "[$(date '+%F %T')] $*"; }

part() {  # gpu group tasks...
    local gpu=$1 group=$2; shift 2
    for fold in 1 2; do
        local half; half=$([ "$fold" = 1 ] && echo A || echo B)
        LTH_RESULTS=results/e24_cross_style/e/fold$fold $PY scripts/patch.py --gpu "$gpu" --family simple --set-list \
            --same-style --example-wordings shot5 --alphas 1 2 --half "$half" --max-new-tokens 16 --task-only random \
            --tasks "$@" --part "$group" > "logs/e24/patch_e_f${fold}_$group.log" 2>&1
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
touch logs/e24/patch_e.done
