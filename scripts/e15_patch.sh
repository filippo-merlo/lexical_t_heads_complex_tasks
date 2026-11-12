#!/usr/bin/env bash
# E15 (scripts/e15_corpus_generic.py): shared Causal / DBT minus corpus-defined generic heads, N 40 and 80,
# and shared DBT minus E14's generic heads; E14's protocol (check 1, both folds, alphas 1 and 2, means of all
# 35 rule templates, replies of at most 16 tokens). The 28 (fold, task) jobs dealt round-robin to GPUs 0/2/3.
# Writes results/e15_corpus_generic/fold{1,2}/patching_simple.part-g<gpu>.csv and answers/. Start detached:
#   setsid nohup scripts/e15_patch.sh > logs/e15/run.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/e15
say() { echo "[$(date '+%F %T')] $*"; }
TASKS=(antonym synonym singular-plural present-past country-capital country-currency landmark-country park-country
       product-company next_item prev_item english-french english-german english-spanish)
GPUS=(0 2 3)
lane() {
    local i=$1 gpu=${GPUS[$1]}
    for fold in 1 2; do
        local half; half=$([ "$fold" = 1 ] && echo A || echo B)
        local mine=()
        for j in "${!TASKS[@]}"; do
            local k=$(( (fold - 1) * ${#TASKS[@]} + j ))
            [ $(( k % ${#GPUS[@]} )) = "$i" ] && mine+=("${TASKS[$j]}")
        done
        [ ${#mine[@]} = 0 ] && continue
        LTH_RESULTS=results/e15_corpus_generic/fold$fold $PY scripts/patch.py --gpu "$gpu" --family simple --set-list \
            --same-style --alphas 1 2 --half "$half" --max-new-tokens 16 --mean-wordings all \
            --tasks "${mine[@]}" --part "g$gpu" > "logs/e15/patch_f${fold}_g$gpu.log" 2>&1
        say "fold $fold patch lane g$gpu (${mine[*]}) exit $?"
    done
}
PIDS=()
for i in "${!GPUS[@]}"; do
    lane "$i" &
    PIDS+=($!)
    sleep 30
done
wait "${PIDS[@]}"
say "all patching done"
touch logs/e15/patch.done
