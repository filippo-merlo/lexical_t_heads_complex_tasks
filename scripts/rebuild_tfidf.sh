#!/usr/bin/env bash
# Rebuilt design, word-list variant (user 2026-11-21): the shared 40-head sets of the five best stage-2
# shared combinations of each fold, re-identified with the smooth TF-IDF list (task against WikiText;
# scripts/rebuild_tfidf_sets.py, run first). Patched exactly like the stage-2 screen
# (scripts/rebuild_stage2.sh, phase A): check 1, the choosing half's means of Yang et al.'s five
# templates (at least 10 correct prompts), task means only, alphas 1 and 2, replies of at most 16
# tokens. One job per fold and task, from one queue over the GPUs; a job whose output exists is skipped.
# Writes results/rebuild/tfidf/fold{1,2}/patching_simple.part-tfidf-<task>.csv and answers/. Start detached:
#   setsid nohup scripts/rebuild_tfidf.sh 0 2 3 5 > logs/rebuild/tfidf.log 2>&1 < /dev/null &
# Report: .venv/bin/python scripts/rebuild_tfidf_report.py
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
DIR=results/rebuild/tfidf SETS=shared_tfidf PART=tfidf
Q=logs/rebuild/${PART}_queue

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
    local dir=$DIR/fold$fold part=$PART-$task
    [ -f "$dir/patching_simple.part-$part.csv" ] && { say "skip fold $fold $part (done)"; return; }
    LTH_RESULTS=$dir $PY scripts/patch.py --gpu "$gpu" "${COMMON[@]}" --half "$half" --task-only "$SETS" \
        --names "${SETS}_*" --tasks "$task" --part "$part" > "logs/rebuild/${part}_f$fold.log" 2>&1
    say "fold $fold $part gpu $gpu exit $?"
}

worker() {  # gpu
    local line
    while line=$(next_job) && [ -n "$line" ]; do
        job "$1" $line
    done
}

for fold in 1 2; do [ -f "$DIR/fold$fold/heads.json" ] || { say "no heads.json for fold $fold"; exit 1; }; done
say "$PART start (GPUs ${GPUS[*]})"
: > "$Q"
for task in "${TASKS[@]}"; do for fold in 1 2; do echo "$fold $task" >> "$Q"; done; done
for gpu in "${GPUS[@]}"; do worker "$gpu" & done
wait
missing=0
for fold in 1 2; do for task in "${TASKS[@]}"; do
    [ -f "$DIR/fold$fold/patching_simple.part-$PART-$task.csv" ] || { say "missing fold $fold $task"; missing=1; }
done; done
[ "$missing" = 0 ] || { say "$PART incomplete"; exit 1; }
say "$PART done"
