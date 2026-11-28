#!/usr/bin/env bash
# J-T or J-answer, shared, 20 and 40 heads (scripts/jdec_sets.py), with control means and their random sets with task
# means only, as scripts/pham_frame_patch.sh ran Pham et al.'s form: alphas 1 and 2, all test items, replies of at
# most 16 tokens, Yang et al.'s five template means. RULE jT -> part jdec-<task> (run 2026-11-28 with the pattern
# 'shared_j*' while heads.json held only the jT sets); RULE jA -> part jdecA-<task>. Then
# scripts/rebuild_report.py --dir jdec --stage jdec (both parts) and scripts/jdec_compare.py. One queue of
# (fold, task) jobs; a job whose output exists is skipped.
#   scripts/jdec_patch.sh jA 0 2 3 5
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
say() { echo "[$(date '+%F %T')] $*"; }
RULE=$1; shift
case $RULE in jT) PART=jdec ;; jA) PART=jdecA ;; *) echo "RULE must be jT or jA" >&2; exit 2 ;; esac
GPUS=("$@")
[ ${#GPUS[@]} -gt 0 ] || GPUS=(0 2 3 5)
DIR=results/rebuild/jdec
TASKS=(antonym synonym singular-plural present-past country-capital country-currency landmark-country
       park-country product-company next_item prev_item english-french english-german english-spanish)
COMMON=(--family simple --set-list --same-style --mean-styles rule --control-same-wording --min-mean-prompts 10
        --alphas 1 2 --max-new-tokens 16 --task-only random)
Q=logs/rebuild/${PART}_queue
next_job() { ( flock 9; line=$(head -n 1 "$Q"); [ -n "$line" ] && sed -i 1d "$Q"; echo "$line" ) 9> "$Q.lock"; }
job() {  # gpu fold task
    local gpu=$1 fold=$2 task=$3 half t0=$SECONDS; half=$([ "$fold" = 1 ] && echo A || echo B)
    [ -f "$DIR/fold$fold/patching_simple.part-$PART-$task.csv" ] && { say "skip fold $fold $PART-$task (done)"; return; }
    LTH_RESULTS=$DIR/fold$fold $PY scripts/patch.py --gpu "$gpu" "${COMMON[@]}" --half "$half" \
        --names "shared_$RULE*" "random_shared_$RULE*" --tasks "$task" --part "$PART-$task" > "logs/rebuild/$PART-${task}_f$fold.log" 2>&1
    say "fold $fold $PART-$task gpu $gpu exit $? ($((SECONDS - t0)) s)"
}
worker() { local line; while line=$(next_job) && [ -n "$line" ]; do job "$1" $line; done; }
say "$RULE start (GPUs ${GPUS[*]})"
: > "$Q"
for task in "${TASKS[@]}"; do for fold in 1 2; do echo "$fold $task" >> "$Q"; done; done
for gpu in "${GPUS[@]}"; do worker "$gpu" & done
wait
missing=0
for task in "${TASKS[@]}"; do for fold in 1 2; do
    [ -f "$DIR/fold$fold/patching_simple.part-$PART-$task.csv" ] || { say "missing fold $fold $PART-$task"; missing=1; }
done; done
[ $missing = 0 ] || { say "STOPPED: runs incomplete"; exit 1; }
$PY scripts/rebuild_report.py --dir jdec --stage jdec
$PY scripts/jdec_compare.py
say "$RULE done"
