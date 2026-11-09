#!/usr/bin/env bash
# E6 patching as E4 (scripts/heldout_patch.sh, frozen): ten alphas, rule means of the choosing
# half, control in both checks, replies of at most 16 tokens. Stages (STAGES, default all):
#   main   dcc, dcc20, dcc40, dccA, and E4's cc20 again as a check; both checks;
#   mainT  dccT, dccT20, dccT40, dccTA (scripts/diff_sets.py --task-words-only); both checks;
#   random the random sets, check 1, task means only.
# Four GPUS: GPUS[0] / [1] fold 1 / fold 2 check 1 (main, mainT, random), GPUS[2] / [3] fold 1 /
# fold 2 check 2 (main, mainT). Every part is run with --tasks (all 14), so that patch.py writes
# results/diff_heads/fold{f}/patching_{family}.part-{stage}.csv and answers/ under its own name.
#   GPUS="2 0 3 5" setsid nohup scripts/diff_patch.sh > logs/diff_heads/patch.log 2>&1 < /dev/null &
#   GPUS="2 0 3 5" STAGES="main mainT" ...   (only those stages)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ALPHAS="-10 -4 1 2 3 4 5 6 8 10"
TASKS=$($PY -c "from lth import simple; print(' '.join(simple.TASKS))")
read -ra G <<< "${GPUS:?set GPUS, four nvidia-smi indices}"
[ "${#G[@]}" = 4 ] || { echo "GPUS needs four indices"; exit 1; }
STAGES=" ${STAGES:-main mainT random} "
mkdir -p logs/diff_heads
say() { echo "[$(date '+%F %T')] $*"; }
set -f  # the name patterns go to patch.py unexpanded
declare -A NAMES=([main]="dcc dcc20 dcc40 dccA cc20" [mainT]="dccT dccT20 dccT40 dccTA" [random]="random*")

run() {  # gpu stage fold family
    local gpu=$1 stage=$2 fold=$3 family=$4
    [[ "$STAGES" == *" $stage "* ]] || return 0
    local half; half=$([ "$fold" = 1 ] && echo A || echo B)
    LTH_RESULTS=results/diff_heads/fold$fold $PY scripts/patch.py --gpu "$gpu" --family "$family" --set-list \
        --same-style --alphas $ALPHAS --half "$half" --wrong-control --max-new-tokens 16 --task-only random \
        --tasks $TASKS --part "$stage" --names ${NAMES[$stage]} \
        > "logs/diff_heads/patch_f${fold}_${family}_${stage}.log" 2>&1
    say "fold $fold $family $stage exit $?"
}

(run "${G[0]}" main 1 simple; run "${G[0]}" mainT 1 simple; run "${G[0]}" random 1 simple) &
(run "${G[1]}" main 2 simple; run "${G[1]}" mainT 2 simple; run "${G[1]}" random 2 simple) &
(run "${G[2]}" main 1 simple-wrong; run "${G[2]}" mainT 1 simple-wrong) &
(run "${G[3]}" main 2 simple-wrong; run "${G[3]}" mainT 2 simple-wrong) &
wait
say "patching done"
