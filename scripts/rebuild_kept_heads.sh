#!/usr/bin/env bash
# Step 1 diagnostic: patch the kept sets (stage-2 options, task means only). One job per GPU.
cd /project/fmerlo/lexical_t_heads_complex_tasks
export CUDA_DEVICE_ORDER=PCI_BUS_ID
COMMON=(--family simple --set-list --same-style --max-new-tokens 16 --alphas 1 2 --mean-styles rule --control-same-wording --min-mean-prompts 10)
run() {  # gpu fold task
    local half; half=$([ "$2" = 1 ] && echo A || echo B)
    LTH_RESULTS=results/rebuild/diagnostics/kept_heads/fold$2 .venv/bin/python scripts/patch.py --gpu "$1" "${COMMON[@]}" --half "$half" \
        --task-only kept --names 'kept*' --tasks "$3" --part "kept-$3" > "logs/rebuild/kept-$3_f$2.log" 2>&1
    echo "[$(date '+%F %T')] fold $2 kept-$3 gpu $1 exit $?"
}
( run 0 1 country-capital; run 0 2 singular-plural ) &
( run 2 2 country-capital ) &
( run 3 1 present-past; run 3 1 singular-plural ) &
( run 5 2 present-past ) &
wait
echo "kept runs done"
