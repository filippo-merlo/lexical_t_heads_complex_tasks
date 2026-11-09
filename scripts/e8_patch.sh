#!/usr/bin/env bash
# E8 (scripts/e8_sets.py): patching of the 31 filter-combination sets (D decoding, B BOS, T task
# words, I input, R rest), E4 protocol: both folds, alphas 1 and 2, rule means of the choosing half,
# control mean in both checks, replies of at most 16 tokens, check 1 (simple) and check 2
# (simple-wrong), no controls. Per GPU a fixed group of simple tasks. Writes
# results/e8_filters/fold{1,2}/patching_{simple,simple-wrong}.part-main-<group>.csv and answers/.
# Start detached:
#   setsid nohup scripts/e8_patch.sh > logs/e8/patch.log 2>&1 < /dev/null &
# D-filter rerun (30-word list + case fix; the old D rows were dropped from the main-* parts):
#   SETS="D DB DT DI DR DBT DBI DBR DTI DTR DIR DBTI DBTR DBIR DTIR DBTIR" PART=d30 \
#     setsid nohup scripts/e8_patch.sh > logs/e8/patch_d30.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ALPHAS="1 2"
SETS="${SETS:-D B T I R DB DT DI DR BT BI BR TI TR IR DBT DBI DBR DTI DTR DIR BTI BTR BIR TIR DBTI DBTR DBIR DTIR BTIR DBTIR}"
PART="${PART:-main}"
mkdir -p logs/e8
say() { echo "[$(date '+%F %T')] $*"; }

run() {  # gpu group fold family
    local gpu=$1 group=$2 fold=$3 family=$4
    local half; half=$([ "$fold" = 1 ] && echo A || echo B)
    LTH_RESULTS=results/e8_filters/fold$fold $PY scripts/patch.py --gpu "$gpu" --family "$family" --set-list \
        --same-style --alphas $ALPHAS --half "$half" --wrong-control --max-new-tokens 16 \
        --tasks $TASKS --part "$PART-$group" --names $SETS > "logs/e8/patch_f${fold}_${family}_$PART-$group.log" 2>&1
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
touch "logs/e8/patch_$PART.done"
