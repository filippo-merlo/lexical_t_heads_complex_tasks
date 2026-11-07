#!/usr/bin/env bash
# E5 (paper/proposal-llm-heads.tex), patching, as E4 (scripts/heldout_patch.sh): per fold, the
# LLM sets (after removal and raw), cc30, fv30 and E4's fv20 (check) in both checks with the task
# and control means; the random sets in check 1 with the task means only. Ten alphas, rule means
# of the choosing half, control in both checks, replies of at most 16 tokens. Sets from
# results/llm_heads/fold{1,2}/heads.json (scripts/llm_heads_sets.py). Writes
# results/llm_heads/fold{1,2}/patching_{simple,simple-wrong}.part-{main,random}.csv and answers/.
# MAIN_NAMES / RANDOM_NAMES (name patterns) and STAGE (suffix of the part names) select a subset,
# e.g. the larger sets added after the first patching (STAGE=big). Start detached:
#   TASKS="english-german" setsid nohup scripts/llm_heads_patch.sh > logs/llm_heads/patch.log 2>&1 < /dev/null &
#   STAGE=big MAIN_NAMES="allJ60* allJ80* allJNt* batJ60* batJ80* batJNt* batL60* batL80* batLNt* cc60 cc80 fv60 fv80" \
#     RANDOM_NAMES="random_*60_* random_*80_* random_*Nt_*" setsid nohup scripts/llm_heads_patch.sh > logs/llm_heads/patch_big.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ALPHAS="-10 -4 1 2 3 4 5 6 8 10"
TASKS=${TASKS:-english-german}
STAGE=${STAGE:-}
read -ra MAIN <<< "${MAIN_NAMES:-all* bat* cc30 fv30 fv20}"
read -ra RANDOM_SETS <<< "${RANDOM_NAMES:-random*}"
set -f  # the name patterns go to patch.py unexpanded
mkdir -p logs/llm_heads
rm -f logs/llm_heads/patch$STAGE.done
say() { echo "[$(date '+%F %T')] $*"; }

run() {  # gpu stage fold family names...
    local gpu=$1 stage=$2 fold=$3 family=$4; shift 4
    local half; half=$([ "$fold" = 1 ] && echo A || echo B)
    LTH_RESULTS=results/llm_heads/fold$fold $PY scripts/patch.py --gpu "$gpu" --family "$family" --set-list \
        --same-style --alphas $ALPHAS --half "$half" --wrong-control --max-new-tokens 16 --task-only random \
        --tasks $TASKS --part "$stage" --names "$@" > "logs/llm_heads/patch_f${fold}_${family}_${stage}.log" 2>&1
    say "fold $fold $family $stage exit $?"
}

(run 0 "main$STAGE" 1 simple "${MAIN[@]}"; run 0 "random$STAGE" 1 simple "${RANDOM_SETS[@]}") &
(run 2 "main$STAGE" 2 simple "${MAIN[@]}"; run 2 "random$STAGE" 2 simple "${RANDOM_SETS[@]}") &
run 3 "main$STAGE" 1 simple-wrong "${MAIN[@]}" &
run 5 "main$STAGE" 2 simple-wrong "${MAIN[@]}" &
wait
say "all parts done"
touch logs/llm_heads/patch$STAGE.done
