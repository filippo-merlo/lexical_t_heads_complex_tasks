#!/usr/bin/env bash
# E4 Phase 4b: random sets in check 2, task means only, each at the cross-fitted alpha of the CC
# set of its size (results/heldout/random_alphas_wrong_fold{f}.json, from scripts/heldout_report.py).
#   setsid nohup scripts/heldout_random_wrong.sh <gpu> <group> <tasks...> > logs/heldout/random_wrong_<group>.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
gpu=$1 group=$2; shift 2
for fold in 1 2; do
    half=$([ "$fold" = 1 ] && echo A || echo B)
    LTH_RESULTS=results/heldout/fold$fold .venv/bin/python scripts/patch.py --gpu "$gpu" --family simple-wrong \
        --set-list --same-style --alphas 1 --half "$half" --wrong-control --max-new-tokens 16 --task-only random \
        --set-alphas "results/heldout/random_alphas_wrong_fold$fold.json" --tasks "$@" --part "random-$group" \
        --names 'random*' > "logs/heldout/patch_f${fold}_simple-wrong_random-$group.log" 2>&1
    echo "[$(date '+%F %T')] fold $fold simple-wrong random $group exit $?"
done
