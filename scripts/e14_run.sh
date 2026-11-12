#!/usr/bin/env bash
# E14 (all wordings): everything after E0 (scripts/e14_outputs.py, started by hand on GPUs 0/2/3/5).
#   1. wait until the 30 new templates have outputs and records for all 14 tasks;
#   2. GPUs 0 and 2: Davidson et al.'s causal score per fold on the 14 question-answer templates
#      (davidson_fv.py --instructions all --cie-only); GPUs 3 and 5: half readings of the new templates,
#      then (CPU) the word lists and D scores, then the attention pass per fold on all 35 templates;
#   3. (CPU) the head sets (e14_sets.py);
#   4. check 1 patching, both folds, alphas 1 and 2, every set with the means of all 35 rule templates
#      (random draws: task means only), replies of at most 16 tokens; the 28 (fold, task) jobs dealt
#      round-robin to the four GPUs. Writes results/e14_wordings/fold{1,2}/patching_simple.part-<lane>.csv
#      and answers/.
# Start detached:
#   setsid nohup scripts/e14_run.sh > logs/e14/run.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/e14
say() { echo "[$(date '+%F %T')] $*"; }
TASKS=(antonym synonym singular-plural present-past country-capital country-currency landmark-country park-country
       product-company next_item prev_item english-french english-german english-spanish)
GPUS=(0 2 3 5)

e0_done() {
    $PY - <<'EOF'
import sys
sys.path.insert(0, ".")
from lth import simple
from lth.store import done, outputs_path, readings_dir
sys.exit(0 if all(outputs_path("simple", t, w).exists() and done(readings_dir("simple", t), w)
                  for t in simple.TASKS for w in simple.E4B_WORDINGS) else 1)
EOF
}
until e0_done && ! pgrep -f "scripts/e14_outputs.py" > /dev/null; do sleep 120; done
say "E0 done"

if [ ! -e results/e14_wordings/sets.done ]; then
    for fold in 1 2; do
        half=$([ "$fold" = 1 ] && echo A || echo B)
        gpu=$([ "$fold" = 1 ] && echo 0 || echo 2)
        $PY scripts/davidson_fv.py --gpu "$gpu" --half "$half" --tasks all --instructions all --cie-only \
            --out "e14_wordings/fold$fold/fv" > "logs/e14/fv_f$fold.log" 2>&1 &
        FV[$fold]=$!
    done
    $PY scripts/heldout_readings.py --gpu 3 --wordings e4b --tasks "${TASKS[@]:0:7}" > logs/e14/readings_g3.log 2>&1 &
    R1=$!
    $PY scripts/heldout_readings.py --gpu 5 --wordings e4b --tasks "${TASKS[@]:7:7}" > logs/e14/readings_g5.log 2>&1 &
    R2=$!
    wait "$R1" "$R2"
    say "half readings exit $?"
    $PY scripts/e14_lists.py > logs/e14/lists.log 2>&1
    say "word lists exit $?"
    LTH_READINGS=data/readings/heldout/A $PY scripts/e13_attention.py --gpu 3 --fold 1 --e14 > logs/e14/attention_f1.log 2>&1 &
    A1=$!
    LTH_READINGS=data/readings/heldout/B $PY scripts/e13_attention.py --gpu 5 --fold 2 --e14 > logs/e14/attention_f2.log 2>&1 &
    A2=$!
    wait "$A1" "$A2"
    say "attention exit $?"
    wait "${FV[1]}" "${FV[2]}"
    say "causal scores exit $?"
    $PY scripts/e14_sets.py > logs/e14/sets.log 2>&1 || { say "sets failed"; exit 1; }
    touch results/e14_wordings/sets.done
    say "sets done"
fi

lane() {  # lane index: the (fold, task) jobs i, i+4, ... of the 28
    local i=$1 gpu=${GPUS[$1]}
    for fold in 1 2; do
        local half; half=$([ "$fold" = 1 ] && echo A || echo B)
        local mine=()
        for j in "${!TASKS[@]}"; do
            local k=$(( (fold - 1) * ${#TASKS[@]} + j ))
            [ $(( k % 4 )) = "$i" ] && mine+=("${TASKS[$j]}")
        done
        [ ${#mine[@]} = 0 ] && continue
        LTH_RESULTS=results/e14_wordings/fold$fold $PY scripts/patch.py --gpu "$gpu" --family simple --set-list \
            --same-style --alphas 1 2 --half "$half" --max-new-tokens 16 --task-only random --mean-wordings all \
            --tasks "${mine[@]}" --part "g$gpu" > "logs/e14/patch_f${fold}_g$gpu.log" 2>&1
        say "fold $fold patch lane g$gpu (${mine[*]}) exit $?"
    done
}
PIDS=()
for i in 0 1 2 3; do
    lane "$i" &
    PIDS+=($!)
    sleep 30
done
wait "${PIDS[@]}"
say "all patching done"
touch logs/e14/patch.done
