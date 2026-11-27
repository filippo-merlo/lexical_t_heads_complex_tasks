#!/usr/bin/env bash
# Shared combination screen at N heads (user 2026-11-22; scripts/rebuild_screen_n.py), two stages as
# scripts/rebuild_quick_screen.sh: sets; quick pass of all 511 shared sets (alpha 2, 10 seeded test items per task,
# one-token replies, task means only); shortlist (each fold's 10 best on the other fold's quick pass); full pass of
# the union (alphas 1 and 2, all test items, replies of at most 16 tokens, task means only); selection on the
# other fold; then the pick (shared_selected<N>) with control means and its 3 random sets (task means only), part
# sel, as stage 2's phase B, and scripts/rebuild_report.py --dir shared_n<N> --stage sel. One queue of (fold, task)
# jobs per pass; a job whose output exists is skipped (sets are rebuilt identically on a rerun).
#   setsid nohup scripts/rebuild_screen_n.sh 80 0 2 3 5 > logs/rebuild/screen_n80.log 2>&1 < /dev/null &
# POOL (user 2026-11-23, stage 3 with the shared Selected chosen at each size; options of scripts/rebuild_stage3.sh):
#   examples  quick and full pass with the P-format means -> select -> test (Q-format means, control means, random
#             sets) and sum (scripts/e26_summed.py, Q-format means) -> results/rebuild/examples_n<N>
#   wordings  candidates (stage 2's shortlist at N) in full with the pool means (Yang's five + P wordings) -> select
#             -> test (Q-wording means, control means, random sets) -> results/rebuild/wordings_n<N>
#   transfer  every size's stage-2 pick but 40 (N is ignored): wt (Q-wording means), et (Q-format means), tsum
#             (summed, Yang's five template means) -> results/rebuild/transfer_n
#   lrp       as additions with the AttnLRP features (scripts/lrp_screen_sets.py) -> results/rebuild/lrp_shared_n<N>
#   POOL=examples scripts/rebuild_screen_n.sh 80 0 2 3 5
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/rebuild
say() { echo "[$(date '+%F %T')] $*"; }
N=$1; shift
GPUS=("$@")
[ ${#GPUS[@]} -gt 0 ] || GPUS=(0 2 3 5)
POOL=${POOL:-additions}
case $POOL in
    additions) DIR=results/rebuild/shared_n$N; TAG=n$N; SCREEN_MEANS=rule ;;
    lrp)       DIR=results/rebuild/lrp_shared_n$N; TAG=lrp_n$N; SCREEN_MEANS=rule ;;  # D + AttnLRP features (lrp_screen_sets.py)
    examples)  DIR=results/rebuild/examples_n$N; TAG=examples_n$N; SCREEN_MEANS=fmtP; TEST_MEANS=fmtQ ;;
    wordings)  DIR=results/rebuild/wordings_n$N; TAG=wordings_n$N; SCREEN_MEANS=pool; TEST_MEANS=Q ;;
    transfer)  DIR=results/rebuild/transfer_n; TAG=transfer_n ;;
    *) echo "POOL: additions, lrp, examples, wordings or transfer" >&2; exit 2 ;;
esac
TASKS=(antonym synonym singular-plural present-past country-capital country-currency landmark-country
       park-country product-company next_item prev_item english-french english-german english-spanish)
COMMON=(--family simple --set-list --same-style --mean-styles rule --control-same-wording --min-mean-prompts 10)
Q=logs/rebuild/screen_${TAG}_queue
FULL_OPTS=(--alphas 1 2 --max-new-tokens 16)
PICK=("shared_selected$N" "random_shared_selected${N}_*")
next_job() { ( flock 9; line=$(head -n 1 "$Q"); [ -n "$line" ] && sed -i 1d "$Q"; echo "$line" ) 9> "$Q.lock"; }
job() {  # gpu pass fold task
    local gpu=$1 pass=$2 fold=$3 task=$4 half; half=$([ "$fold" = 1 ] && echo A || echo B)
    local part=$pass-$task opts names t0=$SECONDS script=scripts/patch.py means=()
    local only=(--task-only shared_screen)
    [ "$POOL" = additions ] || [ "$POOL" = lrp ] || [ "$POOL" = transfer ] || means=(--mean-wordings "$SCREEN_MEANS")  # transfer: set per pass
    case $pass in
        quick) opts=(--alphas 2 --max-new-tokens 1 --n-items 10); names=('shared_screen_*') ;;
        full)  opts=("${FULL_OPTS[@]}"); names=("${SHORT[@]}") ;;
        sel)   opts=("${FULL_OPTS[@]}"); names=("${PICK[@]}"); only=(--task-only random) ;;
        test)  opts=("${FULL_OPTS[@]}"); names=("${PICK[@]}"); only=(--task-only random); means=(--mean-wordings "$TEST_MEANS") ;;
        sum)   opts=("${FULL_OPTS[@]}"); names=("${PICK[@]}"); only=(--task-only random); means=(--mean-wordings "$TEST_MEANS")
               script=scripts/e26_summed.py ;;
        wt)    opts=("${FULL_OPTS[@]}"); names=('shared_selected*' 'random_shared_selected*'); only=(--task-only random); means=(--mean-wordings Q) ;;
        et)    opts=("${FULL_OPTS[@]}"); names=('shared_selected*' 'random_shared_selected*'); only=(--task-only random); means=(--mean-wordings fmtQ) ;;
        tsum)  opts=("${FULL_OPTS[@]}"); names=('shared_selected*' 'random_shared_selected*'); only=(--task-only random); means=(--mean-wordings rule)
               script=scripts/e26_summed.py ;;
    esac
    [ -f "$DIR/fold$fold/patching_simple.part-$part.csv" ] && { say "skip fold $fold $part (done)"; return; }
    LTH_RESULTS=$DIR/fold$fold $PY "$script" --gpu "$gpu" "${COMMON[@]}" "${means[@]}" "${only[@]}" "${opts[@]}" --half "$half" \
        --names "${names[@]}" --tasks "$task" --part "$part" > "logs/rebuild/${TAG}-${part}_f$fold.log" 2>&1
    say "fold $fold $part gpu $gpu exit $? ($((SECONDS - t0)) s)"
}
worker() { local line; while line=$(next_job) && [ -n "$line" ]; do job "$1" $line; done; }
phase() {  # pass...
    : > "$Q"
    local pass task fold
    for pass in "$@"; do for task in "${TASKS[@]}"; do for fold in 1 2; do echo "$pass $fold $task" >> "$Q"; done; done; done
    for gpu in "${GPUS[@]}"; do worker "$gpu" & done
    wait
    local missing=0
    for pass in "$@"; do for task in "${TASKS[@]}"; do for fold in 1 2; do
        [ -f "$DIR/fold$fold/patching_simple.part-$pass-$task.csv" ] || { say "missing fold $fold $pass-$task"; missing=1; }
    done; done; done
    return $missing
}
report() {  # stage
    $PY scripts/rebuild_report.py --dir "${DIR#results/rebuild/}" --stage "$1" | grep "selected" || { say "STOPPED: report $1"; exit 1; }
}
if [ "$POOL" = transfer ]; then
    say "stage-2 shared picks on other prompts start (GPUs ${GPUS[*]})"
    $PY scripts/rebuild_screen_n.py transfer-sets || { say "STOPPED: sets"; exit 1; }
    phase wt et tsum || { say "STOPPED: transfer runs incomplete"; exit 1; }
    for s in wt et tsum; do report "$s"; done
    say "transfer done"
    exit 0
fi
say "shared screen at $N heads, pool $POOL, start (GPUs ${GPUS[*]})"
$PY scripts/rebuild_screen_n.py sets --n "$N" --pool "$POOL" || { say "STOPPED: sets"; exit 1; }
if [ "$POOL" = wordings ]; then
    mapfile -t SHORT < <($PY scripts/rebuild_screen_n.py candidates --n "$N" --pool "$POOL" | tee -a logs/rebuild/screen_${TAG}_shortlist.log | grep '^shared_screen_')
else
    phase quick || { say "STOPPED: quick pass incomplete"; exit 1; }
    mapfile -t SHORT < <($PY scripts/rebuild_screen_n.py shortlist --n "$N" --pool "$POOL" | tee -a logs/rebuild/screen_${TAG}_shortlist.log | grep '^shared_screen_')
fi
[ ${#SHORT[@]} -gt 0 ] || { say "STOPPED: empty shortlist"; exit 1; }
say "shortlist: ${SHORT[*]}"
phase full || { say "STOPPED: full pass incomplete"; exit 1; }
$PY scripts/rebuild_screen_n.py select --n "$N" --pool "$POOL" || { say "STOPPED: selection"; exit 1; }
case $POOL in
    additions|lrp) phase sel || { say "STOPPED: Selected run incomplete"; exit 1; }; report sel ;;
    examples)  phase test sum || { say "STOPPED: Selected runs incomplete"; exit 1; }; report test; report sum ;;
    wordings)  phase test || { say "STOPPED: Selected run incomplete"; exit 1; }; report test ;;
esac
say "shared screen at $N heads done"
