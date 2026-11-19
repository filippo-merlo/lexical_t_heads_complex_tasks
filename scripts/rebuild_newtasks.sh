#!/usr/bin/env bash
# Rebuilt design (paper/head-sets-simple-design.tex), unseen tasks (user 2026-11-19): the shared 40-head sets
# identified on the 14 tasks, patched into tasks that took no part in identifying them
# (lth.simple.UNSEEN_TASKS), with each unseen task's own means.
#   1. unpatched outputs of the unseen tasks (scripts/run_outputs.py; wordings on disk are skipped)
#   2. their scores (scripts/score_outputs.py; capitalize keeps case, lth.evaluation.CASE_SENSITIVE_TASKS)
#   3. records and readings (scripts/build_readings.py)
#   4. item split and sets (scripts/rebuild_newtask_sets.py)
#   5. patching, one job per fold and task, with the options of stage 2 (scripts/rebuild_stage2.sh): check 1,
#      the identification half's means of Yang et al.'s five templates (at least 10 correct prompts), control
#      means of the same templates from the control task, alphas 1 and 2, replies of at most 16 tokens;
#      random sets with task means only. A job whose output exists is skipped (restartable).
# Writes results/rebuild/newtasks/fold{1,2}/patching_simple.part-nt-<task>.csv and answers/.
#   setsid nohup scripts/rebuild_newtasks.sh 0 [PID to wait for] > logs/newtasks/run.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/newtasks
say() { echo "[$(date '+%F %T')] $*"; }
GPU=${1:?gpu}
WAIT=${2:-}
TASKS=(person-occupation person-sport person-instrument capitalize)
COMMON=(--family simple --set-list --same-style --max-new-tokens 16 --alphas 1 2
        --mean-styles rule --control-same-wording --min-mean-prompts 10)

if [ -n "$WAIT" ]; then
    say "waiting for process $WAIT"
    while kill -0 "$WAIT" 2>/dev/null; do sleep 30; done
fi
say "start (GPU $GPU)"
$PY scripts/run_outputs.py --gpu "$GPU" --family simple --tasks "${TASKS[@]}" > logs/newtasks/run_outputs2.log 2>&1 \
    || { say "stopped: run_outputs"; exit 1; }
$PY scripts/score_outputs.py --family simple --tasks "${TASKS[@]}" > logs/newtasks/score_outputs.log 2>&1 \
    || { say "stopped: score_outputs"; exit 1; }
$PY scripts/build_readings.py --gpu "$GPU" --family simple --tasks "${TASKS[@]}" > logs/newtasks/build_readings.log 2>&1 \
    || { say "stopped: build_readings"; exit 1; }
$PY scripts/rebuild_newtask_sets.py > logs/newtasks/sets.log 2>&1 || { say "stopped: sets"; exit 1; }
say "outputs, scores, readings and sets done"
for fold in 1 2; do
    half=$([ "$fold" = 1 ] && echo A || echo B)
    dir=results/rebuild/newtasks/fold$fold
    for task in "${TASKS[@]}"; do
        part=nt-$task
        [ -f "$dir/patching_simple.part-$part.csv" ] && { say "skip fold $fold $part (done)"; continue; }
        LTH_RESULTS=$dir $PY scripts/patch.py --gpu "$GPU" "${COMMON[@]}" --half "$half" --task-only random \
            --tasks "$task" --part "$part" > "logs/newtasks/${part}_f$fold.log" 2>&1
        code=$?
        say "fold $fold $part exit $code"
        [ "$code" = 0 ] || { say "stopped: patch fold $fold $task"; exit 1; }
    done
done
say "unseen tasks done"
