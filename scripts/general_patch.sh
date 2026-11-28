#!/usr/bin/env bash
# The general shared combination (scripts/general_sets.py) at its sizes: shared_general<N> with control means and
# its random sets with task means only, as the pick's run of scripts/rebuild_screen_n.sh (part sel): alphas 1 and 2,
# all test items, replies of at most 16 tokens, Yang et al.'s five template means. Part gen-<task>, then
# scripts/rebuild_report.py --dir general_n --stage gen. One queue of (fold, task) jobs; a job whose output exists
# is skipped.
#   scripts/general_patch.sh 0 2 3 5
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
say() { echo "[$(date '+%F %T')] $*"; }
GPUS=("$@")
[ ${#GPUS[@]} -gt 0 ] || GPUS=(0 2 3 5)
DIR=results/rebuild/general_n
TASKS=(antonym synonym singular-plural present-past country-capital country-currency landmark-country
       park-country product-company next_item prev_item english-french english-german english-spanish)
COMMON=(--family simple --set-list --same-style --mean-styles rule --control-same-wording --min-mean-prompts 10
        --alphas 1 2 --max-new-tokens 16 --task-only random)
Q=logs/rebuild/general_queue
next_job() { ( flock 9; line=$(head -n 1 "$Q"); [ -n "$line" ] && sed -i 1d "$Q"; echo "$line" ) 9> "$Q.lock"; }
job() {  # gpu fold task
    local gpu=$1 fold=$2 task=$3 half t0=$SECONDS; half=$([ "$fold" = 1 ] && echo A || echo B)
    [ -f "$DIR/fold$fold/patching_simple.part-gen-$task.csv" ] && { say "skip fold $fold gen-$task (done)"; return; }
    LTH_RESULTS=$DIR/fold$fold $PY scripts/patch.py --gpu "$gpu" "${COMMON[@]}" --half "$half" \
        --names 'shared_general*' 'random_shared_general*' --tasks "$task" --part "gen-$task" > "logs/rebuild/gen-${task}_f$fold.log" 2>&1
    say "fold $fold gen-$task gpu $gpu exit $? ($((SECONDS - t0)) s)"
}
worker() { local line; while line=$(next_job) && [ -n "$line" ]; do job "$1" $line; done; }
say "general combination start (GPUs ${GPUS[*]})"
: > "$Q"
for task in "${TASKS[@]}"; do for fold in 1 2; do echo "$fold $task" >> "$Q"; done; done
for gpu in "${GPUS[@]}"; do worker "$gpu" & done
wait
missing=0
for task in "${TASKS[@]}"; do for fold in 1 2; do
    [ -f "$DIR/fold$fold/patching_simple.part-gen-$task.csv" ] || { say "missing fold $fold gen-$task"; missing=1; }
done; done
[ $missing = 0 ] || { say "STOPPED: runs incomplete"; exit 1; }
$PY scripts/rebuild_report.py --dir general_n --stage gen
say "general combination done"
