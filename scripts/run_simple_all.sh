#!/usr/bin/env bash
# The simple tasks end to end, as Yang et al.'s code runs them (2026-10-28 rerun): E0 and
# readings on GPUs 0 and 2, then head identification, then the causal check on GPU 0, then
# the E0 table and the Results tables and figures. Start it detached:
#   setsid nohup scripts/run_simple_all.sh > logs/run_simple_all.log 2>&1 < /dev/null &
# With WAIT_PIDS set (E0 lanes already running), it waits for those processes instead of
# starting the lanes. Writes logs/run_simple_all.done at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID

say() { echo "[$(date '+%F %T')] $*"; }

if [ -z "${WAIT_PIDS:-}" ]; then
    $PY scripts/run_simple.py --gpu 0 --tasks antonym synonym singular-plural present-past country-capital \
        country-currency landmark-country > logs/simple_g0.log 2>&1 &
    WAIT_PIDS=$!
    sleep 60
    $PY scripts/run_simple.py --gpu 2 --tasks park-country product-company next_item prev_item english-french \
        english-german english-spanish > logs/simple_g2.log 2>&1 &
    WAIT_PIDS="$WAIT_PIDS $!"
fi
for pid in $WAIT_PIDS; do
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
done
say "E0 lanes finished"
$PY scripts/identify_heads.py > logs/identify_heads.log 2>&1; say "identify exit $?"
$PY scripts/patch.py --gpu 0 --family simple > logs/patch_simple.log 2>&1; say "patch simple exit $?"
$PY scripts/e0_table.py && $PY scripts/results_tex.py; say "tables exit $?"
touch logs/run_simple_all.done
