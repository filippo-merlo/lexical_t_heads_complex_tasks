#!/usr/bin/env bash
# N = 100 and 120 added to E13 (capped grid) and E14 (all wordings), 2026-11-12. The sets come from
# scripts/e13_sets.py and scripts/e14_sets.py with the larger SIZES; every smaller set is unchanged.
# Same protocol as scripts/e13_patch.sh and scripts/e14_run.sh, restricted to the new sets with
# patch.py --names; new part files (suffix _n100-120), so the earlier results stay as they are.
# Per GPU: E13 patching of its task group (both folds), Davidson et al.'s summed vector at N 100/120
# for the same tasks, then its E14 lane (the (fold, task) jobs i, i+4, ... of the 28).
# Start detached:
#   setsid nohup scripts/e13_e14_n100_120.sh > logs/n100_120/run.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/n100_120
say() { echo "[$(date '+%F %T')] $*"; }
TASKS=(antonym synonym singular-plural present-past country-capital country-currency landmark-country park-country
       product-company next_item prev_item english-french english-german english-spanish)
GPUS=(0 2 3 5)
NEW=('*100*' '*120*')

e13_part() {  # gpu group tasks...
    local gpu=$1 group=$2; shift 2
    for fold in 1 2; do
        local half; half=$([ "$fold" = 1 ] && echo A || echo B)
        LTH_RESULTS=results/e13_capped/fold$fold $PY scripts/patch.py --gpu "$gpu" --family simple --set-list \
            --same-style --alphas 1 2 --half "$half" --max-new-tokens 16 --task-only random \
            --names "${NEW[@]}" --tasks "$@" --part "${group}_n100-120" > "logs/n100_120/e13_patch_f${fold}_$group.log" 2>&1
        say "E13 fold $fold patch $group exit $?"
    done
    for fold in 1 2; do
        $PY scripts/e13_fv_residual.py --gpu "$gpu" --fold "$fold" --tasks "$@" --sizes 100 120 \
            > "logs/n100_120/e13_fv_f${fold}_$group.log" 2>&1
        say "E13 fold $fold fv_residual $group exit $?"
    done
}

e14_lane() {  # lane index
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
            --names "${NEW[@]}" --tasks "${mine[@]}" --part "g${gpu}_n100-120" > "logs/n100_120/e14_patch_f${fold}_g$gpu.log" 2>&1
        say "E14 fold $fold patch lane g$gpu (${mine[*]}) exit $?"
    done
}

gpu_job() {  # lane index, E13 group, its tasks
    local i=$1 group=$2; shift 2
    e13_part "${GPUS[$i]}" "$group" "$@"
    e14_lane "$i"
}

gpu_job 0 a antonym synonym singular-plural present-past &
PIDS=($!)
sleep 60
gpu_job 1 b country-capital country-currency landmark-country park-country &
PIDS+=($!)
sleep 30
gpu_job 2 c product-company next_item prev_item &
PIDS+=($!)
sleep 30
gpu_job 3 d english-french english-german english-spanish &
PIDS+=($!)
wait "${PIDS[@]}"
say "all done"
touch logs/n100_120/run.done
