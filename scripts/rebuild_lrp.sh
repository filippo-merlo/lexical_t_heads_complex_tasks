#!/usr/bin/env bash
# T and LRP-T (Pham et al. 2026 mapped onto paper/head-sets-simple-design.tex Sec. 5.3): the task-word attention
# ranking and its AttnLRP version (scripts/rebuild_lrp.py, scripts/lrp_sets.py), patched exactly as Causal (zero-shot)
# in scripts/rebuild_causal_zs.sh, head by head only: T and lrpT, per task and shared, every size, with control means
# and random sets (task means only), alphas 1 and 2. Both folds; check 1; the choosing half's means of Yang et al.'s
# five templates (at least 10 correct prompts); replies of at most 16 tokens.
# One queue of (fold, task) jobs over the GPUs; a job whose output exists is skipped (restartable).
# Writes results/rebuild/additions/fold{1,2}/patching_simple.part-lrp-<task>.csv and answers/. Start detached:
#   setsid nohup scripts/rebuild_lrp.sh 0 3 5 > logs/rebuild/lrp_patch.log 2>&1 < /dev/null &
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
        --mean-styles rule --control-same-wording --min-mean-prompts 10)
HBH=('T[0-9]*' 'random_T[0-9]*' 'shared_T[0-9]*' 'random_shared_T[0-9]*'
     'lrpT[0-9]*' 'random_lrpT[0-9]*' 'shared_lrpT[0-9]*' 'random_shared_lrpT[0-9]*')
Q=logs/rebuild/lrp_queue
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
    if [ -f "$dir/patching_simple.part-lrp-$task.csv" ]; then
        say "skip fold $fold lrp-$task (done)"
    else
        LTH_RESULTS=$dir $PY scripts/patch.py --gpu "$gpu" "${COMMON[@]}" --alphas 1 2 --half "$half" \
            --names "${HBH[@]}" --tasks "$task" --part "lrp-$task" > "logs/rebuild/lrp_f${fold}_$task.log" 2>&1
        say "fold $fold head-by-head $task gpu $gpu exit $?"
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
