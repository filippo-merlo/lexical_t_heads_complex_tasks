#!/usr/bin/env bash
# Complex tasks rerun with the simple-task fixes (2026-10-28): rule heads from the most
# accurate rule wording, causal checks on the rule style with one mean per wording, the
# second check (prompts answered wrongly), and reasoning-mode E0 at 300 items. Start it
# detached, after moving the 100-item reasoning runs to $EARLIER:
#   setsid nohup scripts/pipeline_rerun.sh > logs/pipeline_rerun.log 2>&1 < /dev/null &
#
#  1. Head identification (the direct-answer tasks are final), then one job queue on GPUs
#     0, 2, 3, 5, longest jobs first: reasoning-mode E0 of every task and variant (the
#     earlier 100 items' replies reused) and the direct-answer causal checks.
#  2. E0 table and head identification again, with the 300-item reasoning runs.
#  3. Reasoning-mode causal checks, one job per family and task, on the same queue.
# Model loads are spaced 90 s apart (the node's commit limit fails four at once). Logs in
# logs/rerun/; writes logs/pipeline_rerun.done at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
EARLIER=data/runs/reasoning_100_20261028
GPUS=(0 2 3 5)
mkdir -p logs/rerun

say() { echo "[$(date '+%F %T')] $*"; }

next_job() {  # pop the first line of queue file $1
    flock "$1.lock" bash -c 'head -n1 "$0"; sed -i 1d "$0"' "$1"
}

run_job() {  # gpu kind args...
    local gpu=$1 kind=$2 log
    shift 2
    case $kind in
        e0)  # level variant
            log=logs/rerun/e0_${1//\//-}_$2.log
            $PY scripts/run_reasoning.py --gpu "$gpu" --levels "$1" --variants "$2" --reuse "$EARLIER" > "$log" 2>&1 ;;
        patch)  # family [level]
            log=logs/rerun/patch_$1${2:+_${2//\//-}}.log
            $PY scripts/patch.py --gpu "$gpu" --family "$1" ${2:+--levels "$2"} > "$log" 2>&1 ;;
    esac
    say "GPU $gpu: $kind $* exit $?"
}

worker() {  # gpu queue
    local job
    while job=$(next_job "$2") && [ -n "$job" ]; do
        flock logs/rerun/load.lock sleep 90
        run_job "$1" $job
    done
}

run_queue() {  # queue file: one job per line
    local lanes=()
    for gpu in "${GPUS[@]}"; do
        worker "$gpu" "$1" &
        lanes+=($!)
    done
    wait "${lanes[@]}"
}

[ -d "$EARLIER" ] || { say "missing $EARLIER"; exit 1; }

say "stage 1: identification, reasoning-mode E0, direct-answer causal checks"
$PY scripts/identify_heads.py > logs/rerun/identify_1.log 2>&1 || { say "identify failed"; exit 1; }
# Ordered by the run times of the 100-item runs (two thirds of the items are new).
cat > logs/rerun/queue_1 <<'EOF'
e0 set_game/easy combined
e0 set_game/easy examples
e0 set_game/easy w3
e0 set_game/easy w4
e0 set_game/easy w6
e0 set_game/easy w2
e0 tapatan/easy w3
e0 operator_func/easy examples
e0 set_game/easy orig
e0 tapatan/easy orig
e0 tapatan/easy w2
e0 tapatan/easy w4
e0 tapatan/easy w5
e0 tapatan/easy w6
e0 set_game/easy bare
e0 set_game/easy w5
e0 tapatan/easy w1
e0 tapatan/easy bare
e0 set_game/easy w1
patch complex
patch complex-wrong
e0 tapatan/easy combined
e0 tapatan/easy examples
e0 operator_func/easy orig
e0 operator_func/easy w1
e0 operator_func/easy w2
e0 operator_func/easy w3
e0 operator_func/easy w4
e0 operator_func/easy w5
e0 operator_func/easy w6
e0 operator_func/easy combined
e0 operator_func/easy bare
EOF
run_queue logs/rerun/queue_1

say "stage 2: E0 table and identification"
$PY scripts/e0_table.py > logs/rerun/e0_table.log 2>&1 || { say "e0_table failed"; exit 1; }
$PY scripts/identify_heads.py > logs/rerun/identify_2.log 2>&1 || { say "identify failed"; exit 1; }

say "stage 3: reasoning-mode causal checks"
: > logs/rerun/queue_3
for task in set_game tapatan operator_func; do
    for family in reasoning reasoning-pre reasoning-wrong reasoning-pre-wrong; do
        echo "patch $family $task/easy" >> logs/rerun/queue_3
    done
done
run_queue logs/rerun/queue_3

say "done"
touch logs/pipeline_rerun.done
