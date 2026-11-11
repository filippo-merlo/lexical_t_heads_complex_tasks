#!/usr/bin/env bash
# E12 step 5 (scripts/e12_generic_halves.py): the no-generic sets with the generic heads identified
# on each choosing half. E11's protocol unchanged (scripts/e11_patch.sh): both folds, alphas 1 and
# 2, rule means of the choosing half, control mean in both checks, replies of at most 16 tokens,
# check 1 (simple) and check 2 (simple-wrong), the same four task groups per GPU.
# - fvuniv20_nogen_half: patched head by head (scripts/patch.py, results/e12_generic_halves/fold<f>/heads.json);
# - fv30_nogen_half: Davidson et al.'s intervention (scripts/e11_fv_residual.py --sets-from generic.json).
# Writes results/e12_generic_halves/fold{1,2}/patching_{simple,simple-wrong}.part-<group>.csv,
# fv_residual.part-<tasks>.csv and answers/. Start detached:
#   setsid nohup scripts/e12_patch.sh > logs/e12/patch.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ALPHAS="1 2"
mkdir -p logs/e12
say() { echo "[$(date '+%F %T')] $*"; }

run() {  # gpu group fold family
    local gpu=$1 group=$2 fold=$3 family=$4
    local half; half=$([ "$fold" = 1 ] && echo A || echo B)
    LTH_RESULTS=results/e12_generic_halves/fold$fold $PY scripts/patch.py --gpu "$gpu" --family "$family" --set-list \
        --same-style --alphas $ALPHAS --half "$half" --wrong-control --max-new-tokens 16 \
        --tasks $TASKS --part "$group" --names fvuniv20_nogen_half > "logs/e12/patch_f${fold}_${family}_$group.log" 2>&1
    say "fold $fold $family $group exit $?"
}

part() {  # gpu group tasks...
    local gpu=$1 group=$2; shift 2
    TASKS="$*"
    for fold in 1 2; do
        for family in simple simple-wrong; do run "$gpu" "$group" "$fold" "$family"; done
        $PY scripts/e11_fv_residual.py --gpu "$gpu" --fold "$fold" --tasks $TASKS \
            --sets-from "results/e12_generic_halves/fold$fold/generic.json" \
            --out "results/e12_generic_halves/fold$fold" > "logs/e12/fv_f${fold}_$group.log" 2>&1
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
touch logs/e12/patch.done
