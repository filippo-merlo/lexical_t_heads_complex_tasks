#!/usr/bin/env bash
# Rebuilt design, stage 2 (paper/head-sets-simple-design.tex, Sec. 5): our additions.
# Phase A, one job per fold and task, from one queue over the GPUs:
#   - the ladder rules Yang terms (J-head), D and Causal - generic, per task and shared, every size, with
#     their control means and random sets (task means only);
#   - the screen: every combination of the nine features at 40 heads, per task and shared, task means
#     only (sets equal to an earlier one of the job are patched once, scripts/patch.py).
# Then scripts/rebuild_select.py picks Selected per scope and fold by cross-fitting and adds its sets.
# Phase B: Selected at every size, per task and shared, with control means and random sets.
# Check 1; the choosing half's means of Yang et al.'s five templates (at least 10 correct prompts),
# control means of the same templates; alphas 1 and 2; replies of at most 16 tokens. A job whose output
# exists is skipped (restartable). Writes results/rebuild/additions/fold{1,2}/patching_simple.part-s2-<task>.csv,
# .part-s2sel-<task>.csv and answers/. Start detached:
#   setsid nohup scripts/rebuild_stage2.sh 0 2 3 5 > logs/rebuild/stage2.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/rebuild
say() { echo "[$(date '+%F %T')] $*"; }
GPUS=("$@")
[ ${#GPUS[@]} -gt 0 ] || GPUS=(0 2 3 5)
TASKS=(antonym synonym singular-plural present-past country-capital country-currency landmark-country
       park-country product-company next_item prev_item english-french english-german english-spanish)
COMMON=(--family simple --set-list --same-style --max-new-tokens 16 --alphas 1 2
        --mean-styles rule --control-same-wording --min-mean-prompts 10)
LADDER=('yangJ*' 'random_yangJ*' 'D[0-9]*' 'random_D[0-9]*' 'causalnogen*' 'random_causalnogen*'
        'shared_yangJ*' 'random_shared_yangJ*' 'shared_D[0-9]*' 'random_shared_D[0-9]*'
        'shared_causalnogen*' 'random_shared_causalnogen*' 'screen_*' 'shared_screen_*')
SELECTED=('selected*' 'random_selected*' 'shared_selected*' 'random_shared_selected*')
Q=logs/rebuild/s2_queue

next_job() {  # pop the first line of the queue
    (
        flock 9
        line=$(head -n 1 "$Q")
        [ -n "$line" ] && sed -i 1d "$Q"
        echo "$line"
    ) 9> "$Q.lock"
}

job() {  # gpu phase fold task
    local gpu=$1 phase=$2 fold=$3 task=$4
    local half; half=$([ "$fold" = 1 ] && echo A || echo B)
    local dir=results/rebuild/additions/fold$fold
    local part names only
    if [ "$phase" = A ]; then
        part=s2-$task; names=("${LADDER[@]}"); only=(random screen shared_screen)
    else
        part=s2sel-$task; names=("${SELECTED[@]}"); only=(random)
    fi
    [ -f "$dir/patching_simple.part-$part.csv" ] && { say "skip fold $fold $part (done)"; return; }
    LTH_RESULTS=$dir $PY scripts/patch.py --gpu "$gpu" "${COMMON[@]}" --half "$half" --task-only "${only[@]}" \
        --names "${names[@]}" --tasks "$task" --part "$part" > "logs/rebuild/${part}_f$fold.log" 2>&1
    say "fold $fold $part gpu $gpu exit $?"
}

worker() {  # gpu phase
    local line
    while line=$(next_job) && [ -n "$line" ]; do
        job "$1" "$2" $line
    done
}

phase() {  # A or B
    : > "$Q"
    for task in "${TASKS[@]}"; do for fold in 1 2; do echo "$fold $task" >> "$Q"; done; done
    for gpu in "${GPUS[@]}"; do worker "$gpu" "$1" & done
    wait
}

say "stage 2 start (GPUs ${GPUS[*]})"
phase A
say "phase A done"
missing=0
for fold in 1 2; do for task in "${TASKS[@]}"; do
    [ -f "results/rebuild/additions/fold$fold/patching_simple.part-s2-$task.csv" ] || { say "missing fold $fold $task"; missing=1; }
done; done
[ "$missing" = 0 ] || { say "stage 2 stopped: phase A incomplete"; exit 1; }
$PY scripts/rebuild_select.py || { say "stage 2 stopped: selection failed"; exit 1; }
say "selection done"
phase B
say "stage 2 done"
