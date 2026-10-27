#!/usr/bin/env bash
# Launch the GPU stages on the pool 0 2 3 5, one worker per GPU, model loads staggered by
# 90 s (the node's memory commit limit refuses several simultaneous loads). Workers are
# detached (setsid) and log to logs/. Each worker skips what is already on disk.
#   scripts/run_all.sh runs      # E0 + readings: simple on GPUs 0 2, complex on 3 5
#   scripts/run_all.sh patching  # causal check (after scripts/identify_heads.py)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
PY=.venv/bin/python

launch() {  # name, command...
    local name=$1; shift
    setsid nohup "$@" > "logs/$name.log" 2>&1 < /dev/null &
    echo "$name: pid $!"
}

case "${1:-}" in
runs)
    launch simple_g0 $PY scripts/run_simple.py --gpu 0 --tasks english-spanish english-french antonym
    sleep 90
    launch simple_g2 $PY scripts/run_simple.py --gpu 2 --tasks english-german synonym landmark-country \
        park-country product-company present-past next_item prev_item singular-plural country-capital country-currency
    sleep 90
    launch complex_g3 $PY scripts/run_complex.py --gpu 3 --levels set_game/easy set_game/medium set_game/hard \
        tapatan/easy tapatan/medium tapatan/hard noun_class_agreement/easy noun_class_agreement/medium noun_class_agreement/hard
    sleep 90
    launch complex_g5 $PY scripts/run_complex.py --gpu 5 --levels operator_func/easy operator_func/medium operator_func/hard \
        lexical_category_inference/easy lexical_category_inference/medium-or lexical_category_inference/medium-and \
        lexical_category_inference/hard-or lexical_category_inference/hard-and
    ;;
patching)
    launch patch_simple $PY scripts/patch.py --gpu 0 --family simple
    sleep 90
    launch patch_complex $PY scripts/patch.py --gpu 2 --family complex
    ;;
*)
    echo "usage: $0 runs|patching" >&2; exit 2 ;;
esac
