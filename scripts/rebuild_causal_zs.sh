#!/usr/bin/env bash
# Causal (zero-shot) (paper/head-sets-simple-design.tex, Sec. 4.2): the causal ranking scored in the zero-shot
# prompt (scripts/causal_context_cie.py, scripts/causal_context_sets.py), patched exactly as Causal in stage 1
# and Causal - generic in stage 2. Head by head: causalzs and causalzsnogen, per task and shared, every size,
# with control means and random sets (task means only), alphas 1 and 2; summed: Davidson et al.'s own form
# with the zero-shot score (shared_davidson_zs) and its random sets, alpha 1. Both folds; check 1; the choosing
# half's means of Yang et al.'s five templates (at least 10 correct prompts); replies of at most 16 tokens.
# One queue of (fold, task) jobs over the GPUs; a job whose output exists is skipped (restartable).
# Writes results/rebuild/additions/fold{1,2}/patching_simple.part-zs-<task>.csv, .part-zssum-<task>.csv
# and answers/. Start detached:
#   setsid nohup scripts/rebuild_causal_zs.sh 0 2 3 5 > logs/rebuild/causal_zs.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/rebuild
say() { echo "[$(date '+%F %T')] $*"; }
GPUS=("$@")
[ ${#GPUS[@]} -gt 0 ] || GPUS=(0 2 3 5)
TASKS=(${TASKS:-antonym synonym singular-plural present-past country-capital country-currency landmark-country
       park-country product-company next_item prev_item english-french english-german english-spanish})
FOLDS=(${FOLDS:-1 2})
COMMON=(--family simple --set-list --same-style --max-new-tokens 16 --task-only random
        --mean-styles rule --control-same-wording --min-mean-prompts 10)
HBH=('causalzs[0-9]*' 'random_causalzs[0-9]*' 'causalzsnogen*' 'random_causalzsnogen*'
     'shared_causalzs[0-9]*' 'random_shared_causalzs[0-9]*' 'shared_causalzsnogen*' 'random_shared_causalzsnogen*')
SUMMED=(shared_davidson_zs 'random_shared_davidson_zs_*')
Q=logs/rebuild/causal_zs_queue
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
    if [ -f "$dir/patching_simple.part-zs-$task.csv" ]; then
        say "skip fold $fold zs-$task (done)"
    else
        LTH_RESULTS=$dir $PY scripts/patch.py --gpu "$gpu" "${COMMON[@]}" --alphas 1 2 --half "$half" \
            --names "${HBH[@]}" --tasks "$task" --part "zs-$task" > "logs/rebuild/zs_f${fold}_$task.log" 2>&1
        say "fold $fold head-by-head $task gpu $gpu exit $?"
    fi
    if [ -f "$dir/patching_simple.part-zssum-$task.csv" ]; then
        say "skip fold $fold zssum-$task (done)"
    else
        LTH_RESULTS=$dir $PY scripts/e26_summed.py --gpu "$gpu" "${COMMON[@]}" --alphas 1 --half "$half" \
            --names "${SUMMED[@]}" --tasks "$task" --part "zssum-$task" > "logs/rebuild/zssum_f${fold}_$task.log" 2>&1
        say "fold $fold summed $task gpu $gpu exit $?"
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
