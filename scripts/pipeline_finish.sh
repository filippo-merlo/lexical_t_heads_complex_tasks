#!/usr/bin/env bash
# Everything after scripts/pipeline.sh (it waits for logs/pipeline.done): the six rule
# wordings of every complex task, head identification, and the causal checks. Start it
# detached:
#   setsid nohup scripts/pipeline_finish.sh > logs/pipeline_finish.log 2>&1 < /dev/null &
# Every stage skips what is already on disk, so the script can be rerun after a failure.
#
#  1. Rule wordings w1-w6 (materials/complex/wordings.json): Noun Class Agreement, Lexical
#     Category Inference and Operator Function on GPU 5; Set Game and Tapatan split two
#     wordings per GPU over GPUs 0, 2, 3 (their replies are the longest).
#  2. E0 table and head identification.
#  3. Causal checks: complex (GPU 5), and in reasoning mode one task per GPU, with the
#     heads found at the last prompt token and then at the pre-answer position.
# Writes logs/pipeline_finish.done at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

say() { echo "[$(date '+%F %T')] $*"; }

until [ -f logs/pipeline.done ]; do sleep 60; done
say "stage 1: rule wordings"
(
    $PY scripts/run_complex.py --gpu 5 --levels noun_class_agreement/easy lexical_category_inference/easy \
        >> logs/complex_g5b.log 2>&1
    say "wordings complex exit $?"
    $PY scripts/run_reasoning.py --gpu 5 --levels operator_func/easy >> logs/reasoning_g5.log 2>&1
    say "wordings operator_func exit $?"
) &
LANES=($!)
for spec in "0 w1 w2" "2 w3 w4" "3 w5 w6"; do
    set -- $spec
    sleep 90
    (
        $PY scripts/run_reasoning.py --gpu "$1" --levels set_game/easy --variants "$2" "$3" \
            >> "logs/reasoning_g$1.log" 2>&1
        say "wordings set_game $2 $3 exit $?"
        $PY scripts/run_reasoning.py --gpu "$1" --levels tapatan/easy --variants "$2" "$3" \
            >> "logs/reasoning_g$1.log" 2>&1
        say "wordings tapatan $2 $3 exit $?"
    ) &
    LANES+=($!)
done
wait "${LANES[@]}"

say "stage 2: E0 table and head identification"
$PY scripts/e0_table.py || say "e0_table failed"
$PY scripts/identify_heads.py > logs/identify_heads.log 2>&1 || say "identify_heads failed"

say "stage 3: causal checks"
lane() {  # gpu level
    local tag=${2//\//-}
    for family in reasoning reasoning-pre; do
        $PY scripts/patch.py --gpu "$1" --family "$family" --levels "$2" > "logs/patch_${family}_${tag}.log" 2>&1
        say "$family $2 exit $?"
    done
}
( $PY scripts/patch.py --gpu 5 --family complex > logs/patch_complex.log 2>&1; say "complex exit $?" ) &
A=$!
sleep 90
lane 0 operator_func/easy &
B=$!
sleep 90
lane 2 tapatan/easy &
C=$!
sleep 90
lane 3 set_game/easy &
D=$!
wait "$A" "$B" "$C" "$D"
say "done"
touch logs/pipeline_finish.done
