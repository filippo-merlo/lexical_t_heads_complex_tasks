#!/usr/bin/env bash
# The reasoning-mode causal check, after scripts/pipeline.sh has finished (it waits for
# logs/pipeline.done). One task per GPU, each patched with the heads found at the last
# prompt token and then at the pre-answer position. Start it detached:
#   setsid nohup scripts/pipeline_reasoning_patch.sh > logs/pipeline_reasoning_patch.log 2>&1 < /dev/null &
# Writes results/patching_reasoning{,-pre}_<level>.csv and logs/pipeline_reasoning_patch.done.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

say() { echo "[$(date '+%F %T')] $*"; }

until [ -f logs/pipeline.done ]; do sleep 60; done
say "pipeline done; reasoning-mode causal check"

lane() {  # gpu level
    local tag=${2//\//-}
    for family in reasoning reasoning-pre; do
        $PY scripts/patch.py --gpu "$1" --family "$family" --levels "$2" > "logs/patch_${family}_${tag}.log" 2>&1
        say "$family $2 exit $?"
    done
}

lane 0 operator_func/easy &
A=$!
sleep 90
lane 2 tapatan/easy &
B=$!
sleep 90
lane 3 set_game/easy &
C=$!
wait "$A" "$B" "$C"
say "done"
touch logs/pipeline_reasoning_patch.done
