#!/usr/bin/env bash
# Signed screen (user 2026-11-22; sets from scripts/rebuild_signed_sets.py): Selected with features subtracted,
# in two stages as scripts/rebuild_quick_screen.sh, task means only, means of Yang et al.'s five templates.
#   quick  every candidate at 40 heads (and Selected itself), alpha 2, the same seeded 10 test items per task as
#          the TF-IDF quick pass, replies of one token (first-token score)
#          -> results/rebuild/signed/fold<f>/patching_simple.part-quick-<task>.csv
#   then   scripts/rebuild_signed_report.py --shortlist: fold f's shortlist = up to 3 candidates per scope that
#          beat Selected on fold 3-f's quick pass; nothing to run in full if both folds' shortlists are empty
#   full   the union of both folds' shortlists and Selected (alphas 1 and 2, all test items, replies of at most
#          16 tokens) -> results/rebuild/signed/fold<f>/patching_simple.part-full-<task>.csv
#   then   scripts/rebuild_signed_report.py (fold f chooses between Selected and its shortlist on fold 3-f's full
#          pass; results/rebuild/signed/report.md)
# One queue of (fold, task) jobs over the GPUs; a job whose output exists is skipped.
#   setsid nohup scripts/rebuild_signed_screen.sh all 0 2 > logs/rebuild/signed.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/rebuild
say() { echo "[$(date '+%F %T')] $*"; }
PHASE=$1; shift
GPUS=("$@")
[ ${#GPUS[@]} -gt 0 ] || GPUS=(0 2 3 5)
DIR=results/rebuild/signed
TASKS=(antonym synonym singular-plural present-past country-capital country-currency landmark-country
       park-country product-company next_item prev_item english-french english-german english-spanish)
COMMON=(--family simple --set-list --same-style --mean-styles rule --control-same-wording --min-mean-prompts 10
        --mean-wordings rule --task-only signed shared_signed)
Q=logs/rebuild/signed_queue
next_job() { ( flock 9; line=$(head -n 1 "$Q"); [ -n "$line" ] && sed -i 1d "$Q"; echo "$line" ) 9> "$Q.lock"; }

job() {  # gpu pass fold task
    local gpu=$1 pass=$2 fold=$3 task=$4 half; half=$([ "$fold" = 1 ] && echo A || echo B)
    local part=$pass-$task opts names t0=$SECONDS
    if [ "$pass" = quick ]; then
        opts=(--alphas 2 --max-new-tokens 1 --n-items 10); names=('signed_*' 'shared_signed_*')
    else
        opts=(--alphas 1 2 --max-new-tokens 16); mapfile -t names < "$DIR/full_names.txt"
    fi
    [ -f "$DIR/fold$fold/patching_simple.part-$part.csv" ] && { say "skip fold $fold $part (done)"; return; }
    LTH_RESULTS=$DIR/fold$fold $PY scripts/patch.py --gpu "$gpu" "${COMMON[@]}" "${opts[@]}" --half "$half" \
        --names "${names[@]}" --tasks "$task" --part "$part" > "logs/rebuild/signed-${part}_f$fold.log" 2>&1
    say "fold $fold $part gpu $gpu exit $? ($((SECONDS - t0)) s)"
}
worker() { local line; while line=$(next_job) && [ -n "$line" ]; do job "$1" "$2" $line; done; }
phase() {  # quick | full
    : > "$Q"
    for task in "${TASKS[@]}"; do for fold in 1 2; do echo "$fold $task" >> "$Q"; done; done
    for gpu in "${GPUS[@]}"; do worker "$gpu" "$1" & done
    wait
    local missing=0
    for task in "${TASKS[@]}"; do for fold in 1 2; do
        [ -f "$DIR/fold$fold/patching_simple.part-$1-$task.csv" ] || { say "missing fold $fold $1-$task"; missing=1; }
    done; done
    return $missing
}

say "signed screen $PHASE start (GPUs ${GPUS[*]})"
case $PHASE in
    quick) phase quick || exit 1 ;;
    full)  phase full || exit 1 ;;
    all)   phase quick || exit 1
           $PY scripts/rebuild_signed_report.py --shortlist || exit 1
           if [ -s "$DIR/full_names.txt" ]; then
               phase full || exit 1
               $PY scripts/rebuild_signed_report.py || exit 1
           else
               say "no candidate beats Selected on either fold's quick pass: no full pass"
           fi ;;
    *) echo "phase: quick, full or all" >&2; exit 2 ;;
esac
say "signed screen $PHASE done"
