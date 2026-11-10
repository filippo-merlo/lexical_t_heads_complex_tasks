#!/usr/bin/env bash
# E11 (scripts/e11_sets.py, e11_fv_residual.py): the runs that complete paper/head-sets-simple.tex.
# E8/E10 protocol unchanged: both folds, alphas 1 and 2, rule means of the choosing half, control
# mean in both checks (random sets: task means only), replies of at most 16 tokens, check 1
# (simple) and check 2 (simple-wrong). Per GPU a fixed group of simple tasks (as E8/E10); each lane
# then runs Davidson et al.'s own intervention (fv30, fv30_nogen) for its tasks, both folds.
# Writes results/e11_complete/fold{1,2}/patching_{simple,simple-wrong}.part-<group>.csv,
# fv_residual.part-<tasks>.csv and answers/. Start detached:
#   setsid nohup scripts/e11_patch.sh > logs/e11/patch.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ALPHAS="1 2"
NAMES1=('fvuniv*' 'e10dbt*' 'randomuniv*' 'randomdbt*' 'random_layer_*' 'random_drop_*')
NAMES2=("${NAMES1[@]}" 'random20_*' 'random40_*' 'randomNt_*')  # E4's draws: check 2 only
mkdir -p logs/e11
say() { echo "[$(date '+%F %T')] $*"; }

run() {  # gpu group fold family
    local gpu=$1 group=$2 fold=$3 family=$4
    local half; half=$([ "$fold" = 1 ] && echo A || echo B)
    local names=("${NAMES1[@]}"); [ "$family" = simple-wrong ] && names=("${NAMES2[@]}")
    LTH_RESULTS=results/e11_complete/fold$fold $PY scripts/patch.py --gpu "$gpu" --family "$family" --set-list \
        --same-style --alphas $ALPHAS --half "$half" --wrong-control --max-new-tokens 16 --task-only random \
        --tasks $TASKS --part "$group" --names "${names[@]}" > "logs/e11/patch_f${fold}_${family}_$group.log" 2>&1
    say "fold $fold $family $group exit $?"
}

part() {  # gpu group tasks...
    local gpu=$1 group=$2; shift 2
    TASKS="$*"
    for fold in 1 2; do
        for family in simple simple-wrong; do run "$gpu" "$group" "$fold" "$family"; done
    done
    for fold in 1 2; do
        $PY scripts/e11_fv_residual.py --gpu "$gpu" --fold "$fold" --tasks $TASKS > "logs/e11/fv_f${fold}_$group.log" 2>&1
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
touch logs/e11/patch.done
