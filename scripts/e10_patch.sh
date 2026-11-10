#!/usr/bin/env bash
# E10 (scripts/e10_sets.py): patching of the 93 shared 30-head sets (3 ways of combining E8's scores
# over the tasks x 31 filter combinations), E8's protocol unchanged: both folds, alphas 1 and 2, rule
# means of the choosing half, control mean in both checks, replies of at most 16 tokens, check 1
# (simple) and check 2 (simple-wrong). Per GPU a fixed group of simple tasks (as E8). Writes
# results/e10_shared/fold{1,2}/patching_{simple,simple-wrong}.part-<group>.csv and answers/.
# Start detached:
#   setsid nohup scripts/e10_patch.sh > logs/e10/patch.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ALPHAS="1 2"
mkdir -p logs/e10
say() { echo "[$(date '+%F %T')] $*"; }

run() {  # gpu group fold family
    local gpu=$1 group=$2 fold=$3 family=$4
    local half; half=$([ "$fold" = 1 ] && echo A || echo B)
    LTH_RESULTS=results/e10_shared/fold$fold $PY scripts/patch.py --gpu "$gpu" --family "$family" --set-list \
        --same-style --alphas $ALPHAS --half "$half" --wrong-control --max-new-tokens 16 \
        --tasks $TASKS --part "$group" > "logs/e10/patch_f${fold}_${family}_$group.log" 2>&1
    say "fold $fold $family $group exit $?"
}

part() {  # gpu group tasks...
    local gpu=$1 group=$2; shift 2
    TASKS="$*"
    for fold in 1 2; do
        for family in simple simple-wrong; do run "$gpu" "$group" "$fold" "$family"; done
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
touch logs/e10/patch.done
