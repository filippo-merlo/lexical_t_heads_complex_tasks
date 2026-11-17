#!/usr/bin/env bash
# Rebuilt design, stage 1 (paper/head-sets-simple-design.tex, Sec. 4): the two published rules.
# Head by head: Yang et al.'s own form and Yang terms (Logit-head), Causal, per task and shared, with
# their random sets; Davidson et al.'s own form as one summed vector (alpha 1) with its random sets.
# Both folds; check 1; the choosing half's means of Yang et al.'s five templates (at least 10 correct
# prompts), control means of the same templates (random sets: task means only); alphas 1 and 2;
# replies of at most 16 tokens. Sets: results/rebuild/additions/fold{1,2}/heads.json
# (scripts/rebuild_sets.py). Writes results/rebuild/additions/fold{1,2}/patching_simple.part-s1*-<group>.csv
# and answers/. Start detached:
#   setsid nohup scripts/rebuild_stage1.sh > logs/rebuild/stage1.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/rebuild
say() { echo "[$(date '+%F %T')] $*"; }
COMMON=(--family simple --set-list --same-style --max-new-tokens 16 --task-only random
        --mean-styles rule --control-same-wording --min-mean-prompts 10)
HBH=(yang_own 'random_yang_own_*' 'yangL*' 'random_yangL*' 'causal[0-9]*' 'random_causal[0-9]*'
     'shared_yangL*' 'random_shared_yangL*' 'shared_causal[0-9]*' 'random_shared_causal[0-9]*')
SUMMED=(shared_davidson_own 'random_shared_davidson_own_*')

part() {  # gpu group tasks...
    local gpu=$1 group=$2; shift 2
    for fold in 1 2; do
        local half; half=$([ "$fold" = 1 ] && echo A || echo B)
        local dir=results/rebuild/additions/fold$fold
        LTH_RESULTS=$dir $PY scripts/patch.py --gpu "$gpu" "${COMMON[@]}" --alphas 1 2 --half "$half" \
            --names "${HBH[@]}" --tasks "$@" --part "s1-$group" > "logs/rebuild/s1_f${fold}_$group.log" 2>&1
        say "fold $fold head-by-head $group exit $?"
        LTH_RESULTS=$dir $PY scripts/e26_summed.py --gpu "$gpu" "${COMMON[@]}" --alphas 1 --half "$half" \
            --names "${SUMMED[@]}" --tasks "$@" --part "s1sum-$group" > "logs/rebuild/s1sum_f${fold}_$group.log" 2>&1
        say "fold $fold summed $group exit $?"
    done
}

say "stage 1 start"
part 0 a antonym synonym singular-plural present-past &
part 2 b country-capital country-currency landmark-country park-country &
part 3 c product-company next_item prev_item &
part 5 d english-french english-german english-spanish &
wait
say "stage 1 done"
