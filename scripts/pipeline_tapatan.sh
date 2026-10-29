#!/usr/bin/env bash
# The rest of scripts/pipeline_rerun.sh after five Tapatan E0 jobs failed on repeated
# boards (fixed in run_reasoning.py: one reply per distinct prompt). Start it detached:
#   setsid nohup scripts/pipeline_tapatan.sh > logs/pipeline_tapatan.log 2>&1 < /dev/null &
#
#  1. Each GPU joins once its Set Game causal check of pipeline_rerun.sh is over: the
#     missing Tapatan variants, then the Operator Function causal checks.
#  2. E0 table and head identification.
#  3. Tapatan causal checks.
# Logs in logs/rerun/; writes logs/pipeline_tapatan.done at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
EARLIER=data/runs/reasoning_100_20261028
GPUS=(0 2 3 5)

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
    while pgrep -f "scripts/(patch|run_reasoning).py --gpu $1 " > /dev/null; do sleep 60; done
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

say "stage 1: Tapatan E0, Operator Function causal checks"
cat > logs/rerun/queue_t1 <<'EOF'
e0 tapatan/easy w3
e0 tapatan/easy w4
e0 tapatan/easy w5
e0 tapatan/easy w6
e0 tapatan/easy bare
patch reasoning operator_func/easy
patch reasoning-pre operator_func/easy
patch reasoning-wrong operator_func/easy
patch reasoning-pre-wrong operator_func/easy
EOF
run_queue logs/rerun/queue_t1

say "stage 2: E0 table and identification"
$PY scripts/e0_table.py > logs/rerun/e0_table_2.log 2>&1 || { say "e0_table failed"; exit 1; }
$PY scripts/identify_heads.py > logs/rerun/identify_3.log 2>&1 || { say "identify failed"; exit 1; }

say "stage 3: Tapatan causal checks"
cat > logs/rerun/queue_t3 <<'EOF'
patch reasoning tapatan/easy
patch reasoning-pre tapatan/easy
patch reasoning-wrong tapatan/easy
patch reasoning-pre-wrong tapatan/easy
EOF
run_queue logs/rerun/queue_t3

say "done"
touch logs/pipeline_tapatan.done
