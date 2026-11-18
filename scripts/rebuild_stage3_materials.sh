#!/usr/bin/env bash
# Rebuilt design, stage 3 (paper/head-sets-simple-design.tex, Sec. 6): what the re-chosen rules are built
# from, on the choosing half of each fold. Four GPU lanes:
#   - readings (J-head and Logit-head top 10) of the P example formats, then their attention rows;
#   - attention rows of the P wordings, then the example-prompt causal score of fold 1;
#   - the instruction causal score on inst1-4 + P question-answer wordings, fold 1;
#   - the same, fold 2, then the example-prompt causal score of fold 2.
# Then builds the sets of both pools (scripts/rebuild_sets_s3.py, CPU). Start detached:
#   setsid nohup scripts/rebuild_stage3_materials.sh 0 2 3 5 > logs/rebuild/stage3_materials.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs/rebuild
say() { echo "[$(date '+%F %T')] $*"; }
G=("$@")
[ ${#G[@]} -eq 4 ] || G=(0 2 3 5)
run() {  # log command...
    local log=$1; shift
    "$@" > "logs/rebuild/$log.log" 2>&1
    say "$log exit $?"
}
say "stage 3 materials start (GPUs ${G[*]})"
( run s3m_readings_pformats $PY scripts/heldout_readings.py --gpu "${G[0]}" --wordings pformats
  run s3m_attention_pformats $PY scripts/rebuild_attention.py --gpu "${G[0]}" --pool pformats ) &
( run s3m_attention_p $PY scripts/rebuild_attention.py --gpu "${G[1]}" --pool p
  run s3m_demo_cie_f1 $PY scripts/rebuild_demo_cie.py --gpu "${G[1]}" --fold 1 ) &
( run s3m_fv_f1 $PY scripts/davidson_fv.py --gpu "${G[2]}" --half A --tasks all --instructions p --cie-only \
      --out rebuild/wordings/fold1/fv ) &
( run s3m_fv_f2 $PY scripts/davidson_fv.py --gpu "${G[3]}" --half B --tasks all --instructions p --cie-only \
      --out rebuild/wordings/fold2/fv
  run s3m_demo_cie_f2 $PY scripts/rebuild_demo_cie.py --gpu "${G[3]}" --fold 2 ) &
wait
say "GPU materials done"
run s3m_sets_wordings $PY scripts/rebuild_sets_s3.py --pool wordings
run s3m_sets_examples $PY scripts/rebuild_sets_s3.py --pool examples
say "stage 3 materials done"
