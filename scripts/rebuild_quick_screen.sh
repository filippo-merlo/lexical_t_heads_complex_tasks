#!/usr/bin/env bash
# Two-stage screen of the feature combinations (in place of patching all of them in full): the same choice
# of Selected at a fraction of the patched runs. Replayed on the full screens of stage 2 (511 combinations)
# and stage 3.4 (127), it chose the same combinations in every draw (2026-11-21).
#   quick  every combination's set at 40 heads, per task and shared, task means only, alpha 2, a seeded
#          sample of 10 test items per task (patch.py --n-items), replies of one token (first-token score)
#          -> results/rebuild/<pool>/fold<f>/patching_simple.part-quick-<s2|s3e>-<task>.csv (named so that
#             no stage prefix of scripts/rebuild_report.py or scripts/rebuild_figures.py matches it)
#   then   scripts/rebuild_shortlist.py: fold f's shortlist = the 10 best on fold 3-f's quick pass, per scope
#   full   the union of both folds' shortlists in full (alphas 1 and 2, all test items, replies of at most
#          16 tokens), task means only, as the old screen
#          -> results/rebuild/<pool>/fold<f>/patching_simple.part-<s2-screen|s3e-screen>-<task>.csv
#   then   scripts/rebuild_select.py --pool <pool> --shortlist (fold f chooses among its shortlist)
# Pools: additions (stage 2: Yang et al.'s five templates, their means) and examples (stage 3.4: the P example
# formats, their means). One queue of (fold, task) jobs over the GPUs; a job whose output exists is skipped.
#   scripts/rebuild_quick_screen.sh POOL PHASE GPU...     PHASE: quick | full | all (quick, shortlist, full, select)
#   TASKS="antonym" scripts/rebuild_quick_screen.sh additions quick 0 2     (some tasks only, e.g. to time it)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/rebuild
say() { echo "[$(date '+%F %T')] $*"; }
POOL=$1 PHASE=$2; shift 2
GPUS=("$@")
[ ${#GPUS[@]} -gt 0 ] || GPUS=(0 2 3 5)
ALL_TASKS=(antonym synonym singular-plural present-past country-capital country-currency landmark-country
           park-country product-company next_item prev_item english-french english-german english-spanish)
read -r -a TASK_LIST <<< "${TASKS:-${ALL_TASKS[*]}}"
case $POOL in
    additions) MEANS=rule; QUICK=quick-s2; FULL=s2-screen ;;
    tfidf_screen) MEANS=rule; QUICK=quick-s2; FULL=s2-screen ;;  # stage 2 with TF-IDF lists (rebuild_tfidf_sets.py --screen)
    examples)  MEANS=fmtP; QUICK=quick-s3e; FULL=s3e-screen ;;
    *) echo "pool: additions, tfidf_screen or examples" >&2; exit 2 ;;
esac
COMMON=(--family simple --set-list --same-style --mean-styles rule --control-same-wording --min-mean-prompts 10
        --mean-wordings "$MEANS" --task-only screen shared_screen)
Q=logs/rebuild/quick_screen_${POOL}_queue

next_job() {
    (
        flock 9
        line=$(head -n 1 "$Q")
        [ -n "$line" ] && sed -i 1d "$Q"
        echo "$line"
    ) 9> "$Q.lock"
}

shortlist_names() {  # the union of both folds' shortlists, as set names
    $PY - "$POOL" <<'EOF'
import json, sys
d = json.load(open(f"results/rebuild/{sys.argv[1]}/shortlist.json"))
for scope, prefix in (("per task", "screen_"), ("shared", "shared_screen_")):
    for c in d["union"][scope]:
        print(prefix + c)
EOF
}

job() {  # gpu phase fold task
    local gpu=$1 phase=$2 fold=$3 task=$4
    local half; half=$([ "$fold" = 1 ] && echo A || echo B)
    local dir=results/rebuild/$POOL/fold$fold part names opts
    if [ "$phase" = quick ]; then
        part=$QUICK-$task; names=('screen_*' 'shared_screen_*'); opts=(--alphas 2 --max-new-tokens 1 --n-items 10)
    else
        part=$FULL-$task; mapfile -t names < <(shortlist_names); opts=(--alphas 1 2 --max-new-tokens 16)
    fi
    [ -f "$dir/patching_simple.part-$part.csv" ] && { say "skip fold $fold $part (done)"; return; }
    local t0=$SECONDS
    LTH_RESULTS=$dir $PY scripts/patch.py --gpu "$gpu" "${COMMON[@]}" "${opts[@]}" --half "$half" \
        --names "${names[@]}" --tasks "$task" --part "$part" > "logs/rebuild/${part}_f$fold.log" 2>&1
    say "fold $fold $part gpu $gpu exit $? ($((SECONDS - t0)) s)"
}

worker() {  # gpu phase
    local line
    while line=$(next_job) && [ -n "$line" ]; do
        job "$1" "$2" $line
    done
}

phase() {  # quick | full
    : > "$Q"
    for task in "${TASK_LIST[@]}"; do for fold in 1 2; do echo "$fold $task" >> "$Q"; done; done
    for gpu in "${GPUS[@]}"; do worker "$gpu" "$1" & done
    wait
    local missing=0 part; part=$([ "$1" = quick ] && echo "$QUICK" || echo "$FULL")
    for task in "${TASK_LIST[@]}"; do for fold in 1 2; do
        [ -f "results/rebuild/$POOL/fold$fold/patching_simple.part-$part-$task.csv" ] || { say "missing fold $fold $part-$task"; missing=1; }
    done; done
    return $missing
}

say "two-stage screen $POOL $PHASE start (GPUs ${GPUS[*]}; tasks ${TASK_LIST[*]})"
case $PHASE in
    quick) phase quick || exit 1 ;;
    full)  phase full || exit 1 ;;
    all)   phase quick || exit 1
           $PY scripts/rebuild_shortlist.py --pool "$POOL" || exit 1
           phase full || exit 1
           $PY scripts/rebuild_select.py --pool "$POOL" --shortlist || exit 1 ;;
    *) echo "phase: quick, full or all" >&2; exit 2 ;;
esac
say "two-stage screen $POOL $PHASE done"
