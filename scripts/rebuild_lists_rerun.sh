#!/usr/bin/env bash
# Switch to the smooth TF-IDF word lists (user 2026-11-22; scripts/rebuild_lists_switch.py): re-runs every set
# that depends on the lists, with the options of the stage runners (check 1, alphas 1 and 2, replies of at most
# 16 tokens, means of at least 10 correct prompts, control means of the same wordings, random sets with task
# means only). Kept rows of Yang terms (J), Causal and Causal - generic stay in their parts; the re-run sets go
# into parts named <stage part>-L-<task>, which every report and figure glob of their stage matches.
#   check   one kept set of each kind (yangJ40, causalnogen40, shared_*) patched again on antonym, fold 1, into
#           logs/rebuild/lists_check; its rows must equal the kept ones, or the chain stops
#   s2L     D at every size, per task and shared, with controls and random sets      -> additions, part s2-L
#   stage2  scripts/rebuild_stage2.sh: ladder and screen passes exist (skipped), shortlist, Selected, phase B
#   prep3   rebuild_lists_switch.py stage3-archive, rebuild_sets_s3.py (both pools), stage3-split
#   s3A     wtL (3.1: D, Selected, the 3.2 candidates; Q means), sumL (3.5a), etL (3.3), ws (3.2 screen, in full)
#   select  rebuild_select.py --pool wordings --candidates 10; two-stage 3.4 screen (rebuild_quick_screen.sh examples all)
#   s3B     wqL (3.2 re-identified: D, Selected, candidates; Q means), eqL (3.4), esumL (3.5b)
#   nt      rebuild_lists_switch.py newtasks-archive, rebuild_newtask_sets.py, the unseen tasks in full (part nt)
#   report  rebuild_report.py for stage 2 and every stage-3 part, rebuild_newtask_report.py
# A job whose output exists is skipped. Run after rebuild_lists_switch.py stage2-archive, rebuild_sets.py and
# rebuild_lists_switch.py stage2-split. Start detached:
#   setsid nohup scripts/rebuild_lists_rerun.sh 0 2 3 5 > logs/rebuild/lists_rerun.log 2>&1 < /dev/null &
#   FROM=s3A scripts/rebuild_lists_rerun.sh ...     (start at a later step)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/rebuild
say() { echo "[$(date '+%F %T')] $*"; }
stop() { say "STOPPED: $*"; exit 1; }
GPUS=("$@")
[ ${#GPUS[@]} -gt 0 ] || GPUS=(0 2 3 5)
TASKS=(antonym synonym singular-plural present-past country-capital country-currency landmark-country
       park-country product-company next_item prev_item english-french english-german english-spanish)
NT_TASKS=(person-occupation person-sport person-instrument capitalize)
COMMON=(--family simple --set-list --same-style --max-new-tokens 16 --alphas 1 2
        --mean-styles rule --control-same-wording --min-mean-prompts 10)
RL=()  # the rules of R5 (scripts/rebuild_stage3.sh) that depend on the lists
for p in "" shared_; do RL+=("${p}D[0-9]*" "random_${p}D[0-9]*" "${p}selected*" "random_${p}selected*"); done
STEPS=(check s2L stage2 prep3 s3A select s3B nt report)
FROM=${FROM:-check}
Q=logs/rebuild/lists_queue

candidates() {  # the 3.2 candidates: union over folds of the best ten stage-2 combinations of each scope
    $PY - <<'EOF'
import json
d = json.load(open("results/rebuild/additions/selected.json"))
for scope, prefix in (("per task", "screen_"), ("shared", "shared_screen_")):
    for c in sorted({c for f in "12" for c, _ in d[f][scope]["top10_other_fold"][:10]}):
        print(prefix + c)
EOF
}

next_job() {
    (
        flock 9
        line=$(head -n 1 "$Q")
        [ -n "$line" ] && sed -i 1d "$Q"
        echo "$line"
    ) 9> "$Q.lock"
}

job() {  # gpu kind fold task
    local gpu=$1 kind=$2 fold=$3 task=$4
    local half; half=$([ "$fold" = 1 ] && echo A || echo B)
    local script=scripts/patch.py dir means=rule part names only
    case $kind in
        s2L)  dir=results/rebuild/additions/fold$fold; part=s2-L-$task; names=("${RL[@]:0:2}" "${RL[@]:4:2}"); only=(random) ;;
        wtL)  dir=results/rebuild/scenarios/wordings_transfer/fold$fold; means=Q; part=s3wt-L-$task
              names=("${RL[@]}" "${CAND[@]}"); only=(random screen shared_screen) ;;
        sumL) dir=results/rebuild/scenarios/summed/fold$fold; script=scripts/e26_summed.py; part=s3sum-L-$task
              names=("${RL[@]}"); only=(random) ;;
        etL)  dir=results/rebuild/scenarios/examples_transfer/fold$fold; means=fmtQ; part=s3et-L-$task
              names=("${RL[@]}"); only=(random) ;;
        ws)   dir=results/rebuild/wordings/fold$fold; means=pool; part=s3w-screen-$task; names=("${CAND[@]}")
              only=(screen shared_screen) ;;
        wqL)  dir=results/rebuild/wordings/fold$fold; means=Q; part=s3w-test-L-$task; names=("${RL[@]}" "${CAND[@]}")
              only=(random screen shared_screen) ;;
        eqL)  dir=results/rebuild/examples/fold$fold; means=fmtQ; part=s3e-test-L-$task; names=("${RL[@]}"); only=(random) ;;
        esumL) dir=results/rebuild/examples/fold$fold; script=scripts/e26_summed.py; means=fmtQ; part=s3e-sum-L-$task
              names=("${RL[@]}"); only=(random) ;;
        nt)   dir=results/rebuild/newtasks/fold$fold; part=nt-$task; names=(); only=(random) ;;
    esac
    [ -f "$dir/patching_simple.part-$part.csv" ] && { say "skip fold $fold $part (done)"; return; }
    local sel=(); [ ${#names[@]} -gt 0 ] && sel=(--names "${names[@]}")
    local t0=$SECONDS
    LTH_RESULTS=$dir $PY "$script" --gpu "$gpu" "${COMMON[@]}" --half "$half" --mean-wordings "$means" \
        --task-only "${only[@]}" "${sel[@]}" --tasks "$task" --part "$part" > "logs/rebuild/${part}_f$fold.log" 2>&1
    say "fold $fold $part gpu $gpu exit $? ($((SECONDS - t0)) s)"
}

worker() {
    local line
    while line=$(next_job) && [ -n "$line" ]; do job "$1" $line; done
}

phase() {  # kinds...
    : > "$Q"
    local kind task fold tasks
    for kind in "$@"; do
        if [ "$kind" = nt ]; then tasks=("${NT_TASKS[@]}"); else tasks=("${TASKS[@]}"); fi
        for task in "${tasks[@]}"; do for fold in 1 2; do echo "$kind $fold $task" >> "$Q"; done; done
    done
    for gpu in "${GPUS[@]}"; do worker "$gpu" & done
    wait
    local missing=0 dir part
    for kind in "$@"; do
        if [ "$kind" = nt ]; then tasks=("${NT_TASKS[@]}"); else tasks=("${TASKS[@]}"); fi
        for task in "${tasks[@]}"; do for fold in 1 2; do
            case $kind in
                s2L) dir=additions; part=s2-L ;; wtL) dir=scenarios/wordings_transfer; part=s3wt-L ;;
                sumL) dir=scenarios/summed; part=s3sum-L ;; etL) dir=scenarios/examples_transfer; part=s3et-L ;;
                ws) dir=wordings; part=s3w-screen ;; wqL) dir=wordings; part=s3w-test-L ;;
                eqL) dir=examples; part=s3e-test-L ;; esumL) dir=examples; part=s3e-sum-L ;; nt) dir=newtasks; part=nt ;;
            esac
            [ -f "results/rebuild/$dir/fold$fold/patching_simple.part-$part-$task.csv" ] || { say "missing $kind fold $fold $task"; missing=1; }
        done; done
    done
    return $missing
}

check() {  # kept rows reproduced by a new run of the same sets
    local d=logs/rebuild/lists_check
    rm -rf "$d"; mkdir -p "$d"
    cp results/rebuild/additions/fold1/heads.json "$d/heads.json"
    LTH_RESULTS=$d $PY scripts/patch.py --gpu "${GPUS[0]}" "${COMMON[@]}" --half A --mean-wordings rule \
        --names yangJ40 causalnogen40 shared_yangJ40 shared_causalnogen40 --tasks antonym --part check \
        > logs/rebuild/lists_check.log 2>&1 || return 1
    $PY - "$d/patching_simple.part-check.csv" results/rebuild/additions/fold1/patching_simple.part-s2-antonym.csv <<'EOF'
import csv, sys
key = lambda r: (r["heads"], r["source"], r["mean_wording"], r["alpha"])
new = {key(r): r for r in csv.DictReader(open(sys.argv[1])) if r["source"] != "none"}
old = {key(r): r for r in csv.DictReader(open(sys.argv[2])) if key(r) in new}
assert len(new) >= 40 and set(new) == set(old), (len(new), len(old))
bad = [k for k in new if any(new[k][c] != old[k][c] for c in ("accuracy", "first_token", "n_prompts", "n_mean"))]
assert not bad, bad[:5]
print(f"check: {len(new)} rows (yangJ40, causalnogen40, shared_*; task and control) equal to the kept ones")
EOF
}

started=0
for step in "${STEPS[@]}"; do
    [ "$step" = "$FROM" ] && started=1
    [ "$started" = 1 ] || continue
    say "step $step"
    case $step in
        check)  check || stop "kept rows not reproduced" ;;
        s2L)    phase s2L || stop "D ladder incomplete" ;;
        stage2) scripts/rebuild_stage2.sh "${GPUS[@]}" > logs/rebuild/lists_stage2.log 2>&1 || stop "stage 2 runner"
                grep -q "stage 2 done" logs/rebuild/lists_stage2.log || stop "stage 2 runner did not finish" ;;
        prep3)  $PY scripts/rebuild_lists_switch.py stage3-archive || stop "stage3-archive"
                $PY scripts/rebuild_sets_s3.py --pool wordings > logs/rebuild/lists_sets_wordings.log 2>&1 || stop "wordings sets"
                $PY scripts/rebuild_sets_s3.py --pool examples > logs/rebuild/lists_sets_examples.log 2>&1 || stop "examples sets"
                $PY scripts/rebuild_lists_switch.py stage3-split || stop "stage3-split" ;;
        s3A)    mapfile -t CAND < <(candidates); say "${#CAND[@]} candidate sets"
                phase wtL sumL etL ws || stop "stage 3 phase A incomplete" ;;
        select) $PY scripts/rebuild_select.py --pool wordings --candidates 10 || stop "wordings selection"
                scripts/rebuild_quick_screen.sh examples all "${GPUS[@]}" > logs/rebuild/lists_examples_screen.log 2>&1 \
                    || stop "two-stage example screen" ;;
        s3B)    mapfile -t CAND < <(candidates)
                phase wqL eqL esumL || stop "stage 3 phase B incomplete" ;;
        nt)     $PY scripts/rebuild_lists_switch.py newtasks-archive || stop "newtasks-archive"
                $PY scripts/rebuild_newtask_sets.py > logs/rebuild/lists_newtask_sets.log 2>&1 || stop "unseen-task sets"
                phase nt || stop "unseen tasks incomplete" ;;
        report) $PY scripts/rebuild_report.py --stage s2 > logs/rebuild/lists_report.log 2>&1 || stop "report s2"
                for x in "s3wt scenarios/wordings_transfer" "s3sum scenarios/summed" "s3et scenarios/examples_transfer" \
                         "s3w-test wordings" "s3e-test examples" "s3e-sum examples"; do
                    set -- $x
                    $PY scripts/rebuild_report.py --stage "$1" --dir "$2" >> logs/rebuild/lists_report.log 2>&1 || stop "report $1"
                done
                $PY scripts/rebuild_newtask_report.py >> logs/rebuild/lists_report.log 2>&1 || stop "report newtasks" ;;
    esac
    say "step $step done"
done
say "lists re-run done"
