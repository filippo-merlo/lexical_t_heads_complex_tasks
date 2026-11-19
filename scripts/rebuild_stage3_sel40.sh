#!/usr/bin/env bash
# Rebuilt design, stage 3.1 supplement: control means of selected40 and shared_selected40.
# In the 3.1 run (scripts/rebuild_stage3.sh, job wt) these two sets were identical to candidate screen sets
# patched earlier with task means only, and patch.py's de-duplication copied those rows (fixed in 9963708),
# so they have no control rows. This re-runs both sets with the same settings as job wt; the task rows it
# writes repeat the existing ones (checked identical before the report averages them).
#   setsid nohup scripts/rebuild_stage3_sel40.sh 0 > logs/rebuild/stage3_sel40.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU=${1:-0}
say() { echo "[$(date '+%F %T')] $*"; }
TASKS=(antonym synonym singular-plural present-past country-capital country-currency landmark-country
       park-country product-company next_item prev_item english-french english-german english-spanish)
COMMON=(--family simple --set-list --same-style --max-new-tokens 16 --alphas 1 2
        --mean-styles rule --control-same-wording --min-mean-prompts 10)
for fold in 1 2; do
    half=$([ "$fold" = 1 ] && echo A || echo B)
    dir=results/rebuild/scenarios/wordings_transfer/fold$fold
    for task in "${TASKS[@]}"; do
        part=s3wt-sel40-$task
        [ -f "$dir/patching_simple.part-$part.csv" ] && { say "skip fold $fold $part (done)"; continue; }
        LTH_RESULTS=$dir $PY scripts/patch.py --gpu "$GPU" "${COMMON[@]}" --half "$half" --mean-wordings Q \
            --task-only random --names selected40 shared_selected40 --tasks "$task" --part "$part" \
            > "logs/rebuild/${part}_f$fold.log" 2>&1
        say "fold $fold $part gpu $GPU exit $?"
    done
done
say "sel40 done"
