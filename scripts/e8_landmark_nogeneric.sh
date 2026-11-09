#!/usr/bin/env bash
# Landmark-country, check 1 only: E4's FV-20 with and without Davidson's 6 generic FV heads
# (results/davidson_fv_split/sets.json, generic6), both folds, alphas 1 and 2, E4 protocol as
# scripts/e8_patch.sh. Sets: results/e8_landmark_nogeneric/fold{1,2}/heads.json.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/e8
for fold in 1 2; do
    half=$([ "$fold" = 1 ] && echo A || echo B)
    LTH_RESULTS=results/e8_landmark_nogeneric/fold$fold .venv/bin/python scripts/patch.py --gpu 3 --family simple --set-list \
        --same-style --alphas 1 2 --half "$half" --wrong-control --max-new-tokens 16 \
        --tasks landmark-country --part main --names fv20 fv20_nogeneric > "logs/e8/landmark_nogeneric_f$fold.log" 2>&1
    echo "fold $fold exit $?"
done
