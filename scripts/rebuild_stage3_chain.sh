#!/usr/bin/env bash
# One-off chain (2026-11-18): wait for the stage-3 materials and both instruction causal scores, rebuild
# the wordings pool's sets if their first build ran before the causal scores existed, then start stage 3.
set -uo pipefail
cd "$(dirname "$0")/.."
say() { echo "[$(date '+%F %T')] $*"; }
until [ -f results/rebuild/wordings/fold1/fv/cie.npz ] && [ -f results/rebuild/wordings/fold2/fv/cie.npz ] \
      && grep -q "stage 3 materials done" logs/rebuild/stage3_materials.log \
      && ! pgrep -f "davidson_fv.py --gpu [0-9] --half [AB] --tasks all --instructions p" > /dev/null; do sleep 30; done
say "materials and causal scores ready"
grep -q "s3m_sets_examples exit 0" logs/rebuild/stage3_materials.log || { say "chain stopped: example sets failed"; exit 1; }
if ! grep -q "s3m_sets_wordings exit 0" logs/rebuild/stage3_materials.log; then
    .venv/bin/python scripts/rebuild_sets_s3.py --pool wordings > logs/rebuild/s3m_sets_wordings.log 2>&1 \
        || { say "chain stopped: wordings sets failed"; exit 1; }
    say "wordings sets rebuilt"
fi
scripts/rebuild_stage3.sh 0 2 3 5 > logs/rebuild/stage3.log 2>&1
say "stage 3 runner exit $?"
