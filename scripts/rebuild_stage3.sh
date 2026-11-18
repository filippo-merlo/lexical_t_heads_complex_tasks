#!/usr/bin/env bash
# Rebuilt design, stage 3 (paper/head-sets-simple-design.tex, Sec. 6): the rules beyond the prompts they
# were chosen on. Rules: Yang terms (J), D, Selected, Causal, Causal - generic, per task and shared, every
# size, with control means of the same prompts and random sets (task means only); alphas 1 and 2; check 1;
# means of at least 10 correctly answered prompts of the choosing half; replies of at most 16 tokens.
# Phase A, one queue of (job, fold, task) over the GPUs:
#   wt   3.1 wordings, transfer: stage-2 sets with the Q-wording means (+ the 3.2 candidates at 40 heads,
#        task means only)                               -> results/rebuild/scenarios/wordings_transfer/fold<f>
#   et   3.3 examples, transfer: stage-2 sets with the Q-format means
#                                                       -> results/rebuild/scenarios/examples_transfer/fold<f>
#   sum  3.5 stage-2 sets summed after layer 11, Yang et al.'s five template means
#                                                       -> results/rebuild/scenarios/summed/fold<f>
#   ws   3.2 screen: the candidates (best ten stage-2 combinations of each fold, user 2026-11-18) rebuilt on
#        Yang's five + P wordings, at 40 heads, with that pool's means, task means only
#                                                       -> results/rebuild/wordings/fold<f> (part s3w-screen)
#   es   3.4 screen: all 127 example combinations at 40 heads, P-format means, task means only
#                                                       -> results/rebuild/examples/fold<f> (part s3e-screen)
# Then Selected of both pools (scripts/rebuild_select.py). Phase B:
#   wq   3.2 re-chosen sets with the Q-wording means (+ the candidates at 40 heads, task means only)
#   eq   3.4 re-chosen sets with the Q-format means
#   esum 3.5 the 3.4 sets summed after layer 11, Q-format means
# A job whose output exists is skipped (restartable). Start detached:
#   setsid nohup scripts/rebuild_stage3.sh 0 2 3 5 > logs/rebuild/stage3.log 2>&1 < /dev/null &
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
R5=()
for p in "" shared_; do
    R5+=("${p}yangJ*" "random_${p}yangJ*" "${p}D[0-9]*" "random_${p}D[0-9]*" "${p}selected*" "random_${p}selected*"
         "${p}causal[0-9]*" "random_${p}causal[0-9]*" "${p}causalnogen*" "random_${p}causalnogen*")
done
# The 3.2 candidates: union over folds of the best ten stage-2 combinations of each scope.
mapfile -t CAND < <($PY - <<'EOF'
import json
d = json.load(open("results/rebuild/additions/selected.json"))
for scope, prefix in (("per task", "screen_"), ("shared", "shared_screen_")):
    for c in sorted({c for f in "12" for c, _ in d[f][scope]["top10_other_fold"][:10]}):
        print(prefix + c)
EOF
)
Q=logs/rebuild/s3_queue

next_job() {
    (
        flock 9
        line=$(head -n 1 "$Q")
        [ -n "$line" ] && sed -i 1d "$Q"
        echo "$line"
    ) 9> "$Q.lock"
}

link_heads() {  # dir: patch the stage-2 sets from another results folder
    mkdir -p "$1"
    [ -e "$1/heads.json" ] || ln -s "$PWD/results/rebuild/additions/fold$2/heads.json" "$1/heads.json"
}

job() {  # gpu kind fold task
    local gpu=$1 kind=$2 fold=$3 task=$4
    local half; half=$([ "$fold" = 1 ] && echo A || echo B)
    local script=scripts/patch.py dir means part names only
    case $kind in
        wt) dir=results/rebuild/scenarios/wordings_transfer/fold$fold; link_heads "$dir" "$fold"
            means=Q; part=s3wt-$task; names=("${R5[@]}" "${CAND[@]}"); only=(random screen shared_screen) ;;
        et) dir=results/rebuild/scenarios/examples_transfer/fold$fold; link_heads "$dir" "$fold"
            means=fmtQ; part=s3et-$task; names=("${R5[@]}"); only=(random) ;;
        sum) dir=results/rebuild/scenarios/summed/fold$fold; link_heads "$dir" "$fold"; script=scripts/e26_summed.py
            means=rule; part=s3sum-$task; names=("${R5[@]}"); only=(random) ;;
        ws) dir=results/rebuild/wordings/fold$fold; means=pool; part=s3w-screen-$task; names=("${CAND[@]}")
            only=(screen shared_screen) ;;
        es) dir=results/rebuild/examples/fold$fold; means=fmtP; part=s3e-screen-$task; names=('screen_*' 'shared_screen_*')
            only=(screen shared_screen) ;;
        wq) dir=results/rebuild/wordings/fold$fold; means=Q; part=s3w-test-$task; names=("${R5[@]}" "${CAND[@]}")
            only=(random screen shared_screen) ;;
        eq) dir=results/rebuild/examples/fold$fold; means=fmtQ; part=s3e-test-$task; names=("${R5[@]}"); only=(random) ;;
        esum) dir=results/rebuild/examples/fold$fold; script=scripts/e26_summed.py; means=fmtQ; part=s3e-sum-$task
            names=("${R5[@]}"); only=(random) ;;
    esac
    [ -f "$dir/patching_simple.part-$part.csv" ] && { say "skip fold $fold $part (done)"; return; }
    LTH_RESULTS=$dir $PY "$script" --gpu "$gpu" "${COMMON[@]}" --half "$half" --mean-wordings "$means" \
        --task-only "${only[@]}" --names "${names[@]}" --tasks "$task" --part "$part" > "logs/rebuild/${part}_f$fold.log" 2>&1
    say "fold $fold $part gpu $gpu exit $?"
}

worker() {
    local line
    while line=$(next_job) && [ -n "$line" ]; do
        job "$1" $line
    done
}

phase() {  # kinds...
    : > "$Q"
    for kind in "$@"; do for task in "${TASKS[@]}"; do for fold in 1 2; do echo "$kind $fold $task" >> "$Q"; done; done; done
    for gpu in "${GPUS[@]}"; do worker "$gpu" & done
    wait
}

check() {  # kind part-prefix dir-pattern(with FOLD)
    local missing=0
    for fold in 1 2; do for task in "${TASKS[@]}"; do
        local d=${3//FOLD/$fold}
        [ -f "$d/patching_simple.part-$2-$task.csv" ] || { say "missing $1 fold $fold $task"; missing=1; }
    done; done
    return $missing
}

say "stage 3 start (GPUs ${GPUS[*]}; ${#CAND[@]} candidate sets)"
phase wt sum et ws es
say "phase A done"
ok=0
check ws s3w-screen "results/rebuild/wordings/foldFOLD" || ok=1
check es s3e-screen "results/rebuild/examples/foldFOLD" || ok=1
[ "$ok" = 0 ] || { say "stage 3 stopped: screens incomplete"; exit 1; }
$PY scripts/rebuild_select.py --pool wordings --candidates 10 || { say "stage 3 stopped: wordings selection failed"; exit 1; }
$PY scripts/rebuild_select.py --pool examples || { say "stage 3 stopped: examples selection failed"; exit 1; }
say "selection done"
phase wq eq esum
say "stage 3 done"
