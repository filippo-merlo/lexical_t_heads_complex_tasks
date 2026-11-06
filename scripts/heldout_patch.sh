#!/usr/bin/env bash
# E4 (paper/e4_heldout.tex), patching of Phases 3 and 4a: per GPU a fixed group of simple tasks,
# in order (1) the non-FV sets, both folds and checks; (2) the FV sets, once
# scripts/heldout_fv_sets.py has added them (this script runs it when both folds' CIE exist);
# (3) the random sets, check 1, task means only. Secondary sets (ccclean, ad*, fvuniv*) in
# fold 1 only. Ten alphas, rule means of the choosing half, control in both checks, replies of
# at most 16 tokens. Random sets in check 2 (cross-fitted alpha) are run later, from the results.
# Writes results/heldout/fold{1,2}/patching_{simple,simple-wrong}.part-<stage>-<group>.csv and
# answers/. Start detached:
#   setsid nohup scripts/heldout_patch.sh > logs/heldout/patch.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ALPHAS="-10 -4 1 2 3 4 5 6 8 10"
say() { echo "[$(date '+%F %T')] $*"; }

run() {  # gpu group stage fold family names...
    local gpu=$1 group=$2 stage=$3 fold=$4 family=$5; shift 5
    local half; half=$([ "$fold" = 1 ] && echo A || echo B)
    LTH_RESULTS=results/heldout/fold$fold $PY scripts/patch.py --gpu "$gpu" --family "$family" --set-list \
        --same-style --alphas $ALPHAS --half "$half" --wrong-control --max-new-tokens 16 --task-only random \
        --tasks $TASKS --part "$stage-$group" --names "$@" > "logs/heldout/patch_f${fold}_${family}_${stage}-$group.log" 2>&1
    say "fold $fold $family $stage $group exit $?"
}

part() {  # gpu group wait_for tasks...
    local gpu=$1 group=$2 wait_for=$3; shift 3
    TASKS="$*"
    while [ -n "$wait_for" ] && [ ! -e "$wait_for" ]; do sleep 60; done
    for fold in 1 2; do
        local names=(e1 cc cc20 cc40)
        [ "$fold" = 1 ] && names+=(ccclean ad20 ad40)
        for family in simple simple-wrong; do run "$gpu" "$group" main "$fold" "$family" "${names[@]}"; done
    done
    while [ ! -e results/heldout/fv_sets.done ]; do sleep 60; done
    for fold in 1 2; do
        local names=(fv20 fv40 fvNt)
        [ "$fold" = 1 ] && names+=(fvuniv20 fvuniv40)
        for family in simple simple-wrong; do run "$gpu" "$group" fv "$fold" "$family" "${names[@]}"; done
    done
    for fold in 1 2; do run "$gpu" "$group" random "$fold" simple 'random*'; done
}

rm -f results/heldout/fv_sets.done
part 3 a "" antonym synonym singular-plural present-past &
PIDS=($!)
sleep 60
part 5 b "" country-capital country-currency landmark-country park-country &
PIDS+=($!)
part 2 c results/heldout/fold1/fv/patching_simple.csv product-company next_item prev_item &
PIDS+=($!)
part 0 d results/heldout/fold2/fv/patching_simple.csv english-french english-german english-spanish &
PIDS+=($!)
while [ ! -e results/heldout/fold1/fv/cie.npz ] || [ ! -e results/heldout/fold2/fv/cie.npz ]; do sleep 60; done
for fold in 1 2; do $PY scripts/heldout_fv_sets.py --fold $fold; done && touch results/heldout/fv_sets.done
say "fv sets added"
wait "${PIDS[@]}"
say "all parts done"
touch logs/heldout/patch.done
