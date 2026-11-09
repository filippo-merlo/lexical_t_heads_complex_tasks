#!/usr/bin/env bash
# E7 (scripts/e7_sets.py): patching of the most-active-word sets with and without the BOS filter,
# E4 protocol: both folds, ten alphas, rule means of the choosing half, control mean in both checks,
# replies of at most 16 tokens. Per GPU a fixed group of simple tasks: (1) raw, raw_att, clean,
# clean_att in check 1 (simple) and check 2 (simple-wrong); (2) the random controls, check 1, task
# means only. Writes results/e7_attfilter/fold{1,2}/patching_{simple,simple-wrong}.part-<stage>-<group>.csv
# and answers/. Start detached:
#   setsid nohup scripts/e7_patch.sh > logs/e7/patch.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ALPHAS="-10 -4 1 2 3 4 5 6 8 10"
mkdir -p logs/e7
say() { echo "[$(date '+%F %T')] $*"; }

run() {  # gpu group stage fold family names...
    local gpu=$1 group=$2 stage=$3 fold=$4 family=$5; shift 5
    local half; half=$([ "$fold" = 1 ] && echo A || echo B)
    LTH_RESULTS=results/e7_attfilter/fold$fold $PY scripts/patch.py --gpu "$gpu" --family "$family" --set-list \
        --same-style --alphas $ALPHAS --half "$half" --wrong-control --max-new-tokens 16 --task-only random \
        --tasks $TASKS --part "$stage-$group" --names "$@" > "logs/e7/patch_f${fold}_${family}_${stage}-$group.log" 2>&1
    say "fold $fold $family $stage $group exit $?"
}

part() {  # gpu group tasks...
    local gpu=$1 group=$2; shift 2
    TASKS="$*"
    for fold in 1 2; do
        for family in simple simple-wrong; do run "$gpu" "$group" main "$fold" "$family" raw raw_att clean clean_att; done
    done
    for fold in 1 2; do run "$gpu" "$group" random "$fold" simple 'random*'; done
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
touch logs/e7/patch.done
