#!/usr/bin/env bash
# Run the stages that need no further decision, unattended. Start it detached:
#   setsid nohup scripts/pipeline.sh > logs/pipeline.log 2>&1 < /dev/null &
# Every stage skips what is already on disk, so the script can be rerun after a failure.
#
#  1. E0 table and head identification on what is done (simple, complex), then the
#     complex causal check (GPU 5).
#  2. Reasoning-mode E0 and readings: Operator Function (GPU 0), Tapatan (GPU 2),
#     Set Game easy and medium (GPU 3), Set Game hard (GPU 5, after step 1).
#  3. E0 table and head identification on everything.
# The reasoning-mode causal check is not run here: its sample size and alpha grid are
# still to be chosen (proposal Sec. 5.3). Writes logs/pipeline.done at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

say() { echo "[$(date '+%F %T')] $*"; }

say "stage 1: E0 table, head identification, complex causal check"
$PY scripts/e0_table.py || say "e0_table failed"
$PY scripts/identify_heads.py > logs/identify_heads.log 2>&1 || say "identify_heads failed"
(
    $PY scripts/patch.py --gpu 5 --family complex > logs/patch_complex.log 2>&1
    say "complex patching exit $?"
    $PY scripts/run_reasoning.py --gpu 5 --levels set_game/hard >> logs/reasoning_g5.log 2>&1
    say "reasoning set_game/hard exit $?"
) &
LANE5=$!

say "stage 2: reasoning-mode E0"
sleep 90
$PY scripts/run_reasoning.py --gpu 0 --levels operator_func/easy operator_func/medium operator_func/hard \
    >> logs/reasoning_g0.log 2>&1 &
LANE0=$!
sleep 90
$PY scripts/run_reasoning.py --gpu 2 --levels tapatan/easy tapatan/medium tapatan/hard \
    >> logs/reasoning_g2.log 2>&1 &
LANE2=$!
sleep 90
$PY scripts/run_reasoning.py --gpu 3 --levels set_game/easy set_game/medium >> logs/reasoning_g3.log 2>&1 &
LANE3=$!
wait "$LANE0"; say "reasoning operator_func exit $?"
wait "$LANE2"; say "reasoning tapatan exit $?"
wait "$LANE3"; say "reasoning set_game easy/medium exit $?"
wait "$LANE5"

say "stage 3: E0 table and head identification"
$PY scripts/e0_table.py || say "e0_table failed"
$PY scripts/identify_heads.py > logs/identify_heads.log 2>&1 || say "identify_heads failed"
say "done"
touch logs/pipeline.done
