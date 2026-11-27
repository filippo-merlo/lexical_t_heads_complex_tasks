#!/usr/bin/env bash
# Add vs replace (Pham et al. 2026, eval/utils.py): the 40-head sets of LRP-T and Selected, per task and shared,
# patched with alpha * mean ADDED to each head's own output at the last prompt token (patch.py --additive) instead
# of replacing it. Everything else as their replacement runs (scripts/rebuild_lrp.sh, scripts/rebuild_stage2.sh):
# head by head, control means and random sets (task means only), alphas 1 and 2, both folds, check 1, the choosing
# half's means of Yang et al.'s five templates (at least 10 correct prompts), replies of at most 16 tokens.
# One queue of (fold, task) jobs over the GPUs; a job whose output exists is skipped (restartable).
# Writes results/rebuild/additions/fold{1,2}/patching_simple.part-add-<task>.csv and answers/. Start detached:
#   setsid nohup scripts/rebuild_add.sh 0 3 5 > logs/rebuild/add_patch.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/rebuild
say() { echo "[$(date '+%F %T')] $*"; }
GPUS=("$@")
[ ${#GPUS[@]} -gt 0 ] || GPUS=(0 3 5)
TASKS=(${TASKS:-antonym synonym singular-plural present-past country-capital country-currency landmark-country
       park-country product-company next_item prev_item english-french english-german english-spanish})
FOLDS=(${FOLDS:-1 2})
COMMON=(--family simple --set-list --same-style --max-new-tokens 16 --task-only random
        --mean-styles rule --control-same-wording --min-mean-prompts 10 --additive)
SETS=('lrpT40' 'random_lrpT40_*' 'shared_lrpT40' 'random_shared_lrpT40_*'
      'selected40' 'random_selected40_*' 'shared_selected40' 'random_shared_selected40_*')
Q=logs/rebuild/add_queue
next_job() {  # pop the first line of the queue
    (
        flock 9
        line=$(head -n 1 "$Q")
        [ -n "$line" ] && sed -i 1d "$Q"
        echo "$line"
    ) 9> "$Q.lock"
}
job() {  # gpu fold task
    local gpu=$1 fold=$2 task=$3
    local half; half=$([ "$fold" = 1 ] && echo A || echo B)
    local dir=results/rebuild/additions/fold$fold
    if [ -f "$dir/patching_simple.part-add-$task.csv" ]; then
        say "skip fold $fold add-$task (done)"
    else
        LTH_RESULTS=$dir $PY scripts/patch.py --gpu "$gpu" "${COMMON[@]}" --alphas 1 2 --half "$half" \
            --names "${SETS[@]}" --tasks "$task" --part "add-$task" > "logs/rebuild/add_f${fold}_$task.log" 2>&1
        say "fold $fold additive $task gpu $gpu exit $?"
    fi
}
worker() {
    local gpu=$1 line
    while line=$(next_job) && [ -n "$line" ]; do
        job "$gpu" $line
    done
}

: > "$Q"
for fold in "${FOLDS[@]}"; do for task in "${TASKS[@]}"; do echo "$fold $task" >> "$Q"; done; done
say "start (GPUs ${GPUS[*]}; $(wc -l < "$Q") jobs)"
for gpu in "${GPUS[@]}"; do worker "$gpu" & done
wait
say "done"
