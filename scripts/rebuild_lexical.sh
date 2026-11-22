#!/usr/bin/env bash
# Decoding features beyond the task-word list, step A (user 2026-11-22; sets: scripts/rebuild_lexical.py): the
# 40-head sets of every combination of T, Y, A, C, alone and with Selected's attention part, per task and shared,
# patched like the stage-2 screen (check 1, the choosing half's means of Yang et al.'s five templates with at
# least 10 correct prompts, task means only, alphas 1 and 2, replies of at most 16 tokens). One job per fold and
# task over the GPUs; a job whose output exists is skipped. Writes results/rebuild/lexical/fold{1,2}/
# patching_simple.part-lex-<task>.csv and answers/. Report: scripts/rebuild_lexical_report.py
#   setsid nohup scripts/rebuild_lexical.sh 0 2 3 5 > logs/rebuild/lexical.log 2>&1 < /dev/null &
# KIND=mix: the mixed sets of T + attention and C + attention (mix_*, shared_mix_*), parts lex-mix-<task>.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/rebuild
say() { echo "[$(date '+%F %T')] $*"; }
GPUS=("$@")
[ ${#GPUS[@]} -gt 0 ] || GPUS=(0 2 3 5)
TASKS=(antonym synonym singular-plural present-past country-capital country-currency landmark-country
       park-country product-company next_item prev_item english-french english-german english-spanish)
COMMON=(--family simple --set-list --same-style --max-new-tokens 16 --alphas 1 2
        --mean-styles rule --control-same-wording --min-mean-prompts 10)
KIND=${KIND:-lex}
case $KIND in
    lex) PART=lex; SETS=(lex shared_lex) ;;
    mix) PART=lex-mix; SETS=(mix shared_mix) ;;
    *) echo "KIND: lex or mix" >&2; exit 2 ;;
esac
Q=logs/rebuild/lexical_${KIND}_queue
next_job() { ( flock 9; line=$(head -n 1 "$Q"); [ -n "$line" ] && sed -i 1d "$Q"; echo "$line" ) 9> "$Q.lock"; }
job() {  # gpu fold task
    local gpu=$1 fold=$2 task=$3 half; half=$([ "$2" = 1 ] && echo A || echo B)
    local dir=results/rebuild/lexical/fold$fold part=$PART-$task t0=$SECONDS
    [ -f "$dir/patching_simple.part-$part.csv" ] && { say "skip fold $fold $part (done)"; return; }
    LTH_RESULTS=$dir $PY scripts/patch.py --gpu "$gpu" "${COMMON[@]}" --half "$half" --task-only "${SETS[@]}" \
        --names "${SETS[0]}_*" "${SETS[1]}_*" --tasks "$task" --part "$part" > "logs/rebuild/${part}_f$fold.log" 2>&1
    say "fold $fold $part gpu $gpu exit $? ($((SECONDS - t0)) s)"
}
worker() { local line; while line=$(next_job) && [ -n "$line" ]; do job "$1" $line; done; }
say "lexical $KIND start (GPUs ${GPUS[*]})"
: > "$Q"
for task in "${TASKS[@]}"; do for fold in 1 2; do echo "$fold $task" >> "$Q"; done; done
for gpu in "${GPUS[@]}"; do worker "$gpu" & done
wait
missing=0
for fold in 1 2; do for task in "${TASKS[@]}"; do
    [ -f "results/rebuild/lexical/fold$fold/patching_simple.part-$PART-$task.csv" ] || { say "missing fold $fold $task"; missing=1; }
done; done
[ "$missing" = 0 ] || { say "lexical $KIND incomplete"; exit 1; }
say "lexical $KIND done"
