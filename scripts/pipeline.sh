#!/usr/bin/env bash
# Run every remaining stage unattended, in order. Start it detached:
#   setsid nohup scripts/pipeline.sh WAIT_PIDS... > logs/pipeline.log 2>&1 < /dev/null &
# WAIT_PIDS are E0 workers already running; the E0 stages wait for them. Every stage
# skips what is already on disk, so the script can be rerun after a failure.
#
#  1. Set Game in reasoning mode (GPU 3), beside the running workers.
#  2. When every E0 worker is done: a completion pass over all complex and reasoning
#     levels, which adds whatever is missing (the six rule wordings, once
#     materials/complex/wordings.json exists).
#  3. E0 table and head identification.
#  4. Causal checks: complex (GPU 3), reasoning at the last prompt token (GPU 2), and at
#     the pre-answer position (GPU 5); the simple one is already running on GPU 0.
# Writes logs/pipeline.done at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
WAIT_PIDS=("$@")
REASONING_LEVELS=(set_game/easy set_game/medium set_game/hard operator_func/easy operator_func/medium
                  operator_func/hard tapatan/easy tapatan/medium tapatan/hard)
COMPLEX_LEVELS=(noun_class_agreement/easy noun_class_agreement/medium noun_class_agreement/hard
                lexical_category_inference/easy lexical_category_inference/medium-or
                lexical_category_inference/medium-and lexical_category_inference/hard-or
                lexical_category_inference/hard-and)

say() { echo "[$(date '+%F %T')] $*"; }
wait_for() { for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done; }

say "stage 1: Set Game in reasoning mode"
$PY scripts/run_reasoning.py --gpu 3 --levels set_game/easy set_game/medium set_game/hard \
    > logs/reasoning_g3.log 2>&1 &
SETGAME=$!
wait_for "${WAIT_PIDS[@]}" "$SETGAME"

say "stage 2: completion pass (adds wordings if materials/complex/wordings.json exists)"
$PY scripts/run_complex.py --gpu 3 --levels "${COMPLEX_LEVELS[@]}" >> logs/complex_g3b.log 2>&1 &
A=$!
sleep 90
$PY scripts/run_reasoning.py --gpu 2 --levels "${REASONING_LEVELS[@]}" >> logs/reasoning_g2.log 2>&1 &
B=$!
wait "$A" "$B"

say "stage 3: E0 table and head identification"
$PY scripts/e0_table.py || say "e0_table failed"
$PY scripts/identify_heads.py > logs/identify_heads.log 2>&1 || say "identify_heads failed"

say "stage 4: causal checks"
$PY scripts/patch.py --gpu 3 --family complex > logs/patch_complex.log 2>&1 &
C=$!
sleep 90
$PY scripts/patch.py --gpu 2 --family reasoning > logs/patch_reasoning.log 2>&1 &
D=$!
sleep 90
$PY scripts/patch.py --gpu 5 --family reasoning-pre > logs/patch_reasoning_pre.log 2>&1 &
E=$!
wait "$C"; say "complex patching exit $?"
wait "$D"; say "reasoning patching exit $?"
wait "$E"; say "reasoning-pre patching exit $?"
say "done"
touch logs/pipeline.done
