#!/usr/bin/env bash
# Shared combination screen at N heads (user 2026-11-22; scripts/rebuild_screen_n.py), two stages as
# scripts/rebuild_quick_screen.sh: sets; quick pass of all 511 shared sets (alpha 2, 10 seeded test items per task,
# one-token replies, task means only); shortlist (each fold's 10 best on the other fold's quick pass); full pass of
# the union (alphas 1 and 2, all test items, replies of at most 16 tokens, task means only); selection on the
# other fold; then the pick (shared_selected<N>) with control means and its 3 random sets (task means only), part
# sel, as stage 2's phase B, and scripts/rebuild_report.py --dir shared_n<N> --stage sel. One queue of (fold, task)
# jobs per pass; a job whose output exists is skipped (sets are rebuilt identically on a rerun).
#   setsid nohup scripts/rebuild_screen_n.sh 80 0 2 3 5 > logs/rebuild/screen_n80.log 2>&1 < /dev/null &
# LISTS=old: the word lists used until 2026-11-21 (rebuild_screen_n.py --lists old), results/rebuild/shared_n<N>_old.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/rebuild
say() { echo "[$(date '+%F %T')] $*"; }
N=$1; shift
GPUS=("$@")
[ ${#GPUS[@]} -gt 0 ] || GPUS=(0 2 3 5)
LISTS=${LISTS:-tfidf}
SUFFIX=$([ "$LISTS" = old ] && echo _old || echo "")
DIR=results/rebuild/shared_n$N$SUFFIX
TASKS=(antonym synonym singular-plural present-past country-capital country-currency landmark-country
       park-country product-company next_item prev_item english-french english-german english-spanish)
COMMON=(--family simple --set-list --same-style --mean-styles rule --control-same-wording --min-mean-prompts 10)
Q=logs/rebuild/screen_n${N}${SUFFIX}_queue
next_job() { ( flock 9; line=$(head -n 1 "$Q"); [ -n "$line" ] && sed -i 1d "$Q"; echo "$line" ) 9> "$Q.lock"; }
job() {  # gpu pass fold task
    local gpu=$1 pass=$2 fold=$3 task=$4 half; half=$([ "$fold" = 1 ] && echo A || echo B)
    local part=$pass-$task opts names t0=$SECONDS
    local only=(--task-only shared_screen)
    case $pass in
        quick) opts=(--alphas 2 --max-new-tokens 1 --n-items 10); names=('shared_screen_*') ;;
        full)  opts=(--alphas 1 2 --max-new-tokens 16); names=("${SHORT[@]}") ;;
        sel)   opts=(--alphas 1 2 --max-new-tokens 16); names=("shared_selected$N" "random_shared_selected${N}_*"); only=(--task-only random) ;;
    esac
    [ -f "$DIR/fold$fold/patching_simple.part-$part.csv" ] && { say "skip fold $fold $part (done)"; return; }
    LTH_RESULTS=$DIR/fold$fold $PY scripts/patch.py --gpu "$gpu" "${COMMON[@]}" "${only[@]}" "${opts[@]}" --half "$half" \
        --names "${names[@]}" --tasks "$task" --part "$part" > "logs/rebuild/n${N}${SUFFIX}-${part}_f$fold.log" 2>&1
    say "fold $fold $part gpu $gpu exit $? ($((SECONDS - t0)) s)"
}
worker() { local line; while line=$(next_job) && [ -n "$line" ]; do job "$1" $line; done; }
phase() {  # pass
    : > "$Q"
    for task in "${TASKS[@]}"; do for fold in 1 2; do echo "$1 $fold $task" >> "$Q"; done; done
    for gpu in "${GPUS[@]}"; do worker "$gpu" & done
    wait
    local missing=0
    for task in "${TASKS[@]}"; do for fold in 1 2; do
        [ -f "$DIR/fold$fold/patching_simple.part-$1-$task.csv" ] || { say "missing fold $fold $1-$task"; missing=1; }
    done; done
    return $missing
}
say "shared screen at $N heads start (GPUs ${GPUS[*]})"
$PY scripts/rebuild_screen_n.py sets --n "$N" --lists "$LISTS" || { say "STOPPED: sets"; exit 1; }
phase quick || { say "STOPPED: quick pass incomplete"; exit 1; }
mapfile -t SHORT < <($PY scripts/rebuild_screen_n.py shortlist --n "$N" --lists "$LISTS" | tee -a logs/rebuild/screen_n${N}${SUFFIX}_shortlist.log | grep '^shared_screen_')
[ ${#SHORT[@]} -gt 0 ] || { say "STOPPED: empty shortlist"; exit 1; }
say "shortlist: ${SHORT[*]}"
phase full || { say "STOPPED: full pass incomplete"; exit 1; }
$PY scripts/rebuild_screen_n.py select --n "$N" --lists "$LISTS" || { say "STOPPED: selection"; exit 1; }
phase sel || { say "STOPPED: Selected run incomplete"; exit 1; }
$PY scripts/rebuild_report.py --dir "shared_n$N$SUFFIX" --stage sel | grep "selected" || { say "STOPPED: report"; exit 1; }
say "shared screen at $N heads done"
