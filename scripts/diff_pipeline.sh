#!/usr/bin/env bash
# E6, end to end: readings (scripts/diff_readings.py) and sets of both folds (scripts/diff_sets.py)
# on the first GPU of GPUS (the dcc sets, E4's cc20 appended as a check, then the task-words-only
# dccT sets), then the patching on all four (scripts/diff_patch.sh, E4's settings). Writes
# results/diff_heads/fold{1,2}/ and logs/diff_heads/.
#   mkdir -p logs/diff_heads; GPUS="2 0 3 5" setsid nohup scripts/diff_pipeline.sh > logs/diff_heads/pipeline.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ALPHAS="-10 -4 1 2 3 4 5 6 8 10"
mkdir -p logs/diff_heads
say() { echo "[$(date '+%F %T')] $*"; }

read -ra G <<< "${GPUS:?set GPUS, four nvidia-smi indices}"
[ "${#G[@]}" = 4 ] || { say "GPUS needs four indices"; exit 1; }
GPU=${G[0]}
say "GPUs ${G[*]}"

$PY scripts/diff_readings.py --gpu "$GPU" > logs/diff_heads/readings.log 2>&1 || { say "readings failed"; exit 1; }
say "readings done"
for fold in 1 2; do
    half=$([ "$fold" = 1 ] && echo A || echo B)
    $PY scripts/diff_sets.py --choose "$half" --out "results/diff_heads/fold$fold" > "logs/diff_heads/sets_f$fold.log" 2>&1 \
        || { say "sets fold $fold failed"; exit 1; }
    # E4's cc20 of this fold, patched again as a check.
    $PY - "$fold" <<'EOF'
import json, sys
fold = sys.argv[1]
mine = json.load(open(f"results/diff_heads/fold{fold}/heads.json"))
e4 = json.load(open(f"results/heldout/fold{fold}/heads.json"))
for t, v in mine["simple"].items():
    v["sets"].append(next(s for s in e4["simple"][t]["sets"] if s[1] == "cc20"))
json.dump(mine, open(f"results/diff_heads/fold{fold}/heads.json", "w"), indent=1)
EOF
done
say "sets done"

for fold in 1 2; do  # the task-words-only sets (the common words do not cancel in the difference)
    half=$([ "$fold" = 1 ] && echo A || echo B)
    $PY scripts/diff_sets.py --choose "$half" --out "results/diff_heads/fold$fold" --task-words-only \
        > "logs/diff_heads/sets_T_f$fold.log" 2>&1 || { say "dccT sets fold $fold failed"; exit 1; }
done
say "dccT sets done"
GPUS="${G[*]}" scripts/diff_patch.sh
say "all done"
touch logs/diff_heads/pipeline.done
