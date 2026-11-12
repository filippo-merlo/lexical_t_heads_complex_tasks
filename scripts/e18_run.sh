#!/usr/bin/env bash
# E18 (list length): E13's filter sets on 20- and 40-word lists. Runs alongside the N 100/120 run
# (needs only E13's 20-80 results), on GPUs 0/2/3/5 (model loads staggered by 45 s, host RAM):
#   1. attention pass for T (scripts/e13_attention.py --e18 N), 2 lengths x 2 folds;
#   2. sets (scripts/e18_sets.py, CPU; only sets that differ from E13's are listed);
#   3. patching, E13's protocol (scripts/e13_patch.sh): alphas 1 and 2, rule means of the choosing half,
#      control mean, replies of at most 16 tokens, check 1 only; per GPU E13's task group, only the
#      tasks that have a set to patch.
# Writes results/e18_list_length/w{20,40}/fold{1,2}/patching_simple.part-<group>.csv and answers/. Detached:
#   setsid nohup scripts/e18_run.sh > logs/e18/run.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/e18
say() { echo "[$(date '+%F %T')] $*"; }

say "start"

# 1. Attention passes.
GPUS=(0 2 3 5)
i=0
PIDS=()
for n in 20 40; do
    for fold in 1 2; do
        half=$([ "$fold" = 1 ] && echo A || echo B)
        LTH_READINGS=data/readings/heldout/$half $PY scripts/e13_attention.py --gpu "${GPUS[$i]}" --fold "$fold" \
            --e18 "$n" > "logs/e18/attention_w${n}_f$fold.log" 2>&1 &
        PIDS+=($!)
        i=$((i + 1))
        sleep 45
    done
done
fail=0
for p in "${PIDS[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { say "attention pass failed"; exit 1; }
say "attention passes done"

# 2. Sets.
$PY scripts/e18_sets.py > logs/e18/sets.log 2>&1 || { say "sets failed"; exit 1; }
cat logs/e18/sets.log

# 3. Patching.
pending() {  # n fold tasks... -> the tasks with sets in heads.json
    local n=$1 fold=$2; shift 2
    $PY -c 'import json,sys; h=json.load(open(sys.argv[1]))["simple"]; print(" ".join(t for t in sys.argv[2:] if t in h))' \
        "results/e18_list_length/w$n/fold$fold/heads.json" "$@"
}

part() {  # gpu group tasks...
    local gpu=$1 group=$2; shift 2
    for n in 20 40; do
        for fold in 1 2; do
            local half tasks; half=$([ "$fold" = 1 ] && echo A || echo B)
            tasks=$(pending "$n" "$fold" "$@")
            if [ -z "$tasks" ]; then say "w$n fold $fold $group: nothing to patch"; continue; fi
            LTH_RESULTS=results/e18_list_length/w$n/fold$fold $PY scripts/patch.py --gpu "$gpu" --family simple \
                --set-list --same-style --alphas 1 2 --half "$half" --max-new-tokens 16 \
                --tasks $tasks --part "$group" > "logs/e18/patch_w${n}_f${fold}_$group.log" 2>&1
            say "w$n fold $fold patch $group ($tasks) exit $?"
        done
    done
}

part 0 a antonym synonym singular-plural present-past &
PIDS=($!)
sleep 60
part 2 b country-capital country-currency landmark-country park-country &
PIDS+=($!)
sleep 45
part 3 c product-company next_item prev_item &
PIDS+=($!)
sleep 45
part 5 d english-french english-german english-spanish &
PIDS+=($!)
wait "${PIDS[@]}"
say "all parts done"
touch logs/e18/run.done
