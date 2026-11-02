#!/usr/bin/env bash
# E1b (2026-11-02): both simple-task causal checks with the consensus rule heads of
# scripts/identify_consensus.py, in results/consensus (patch.py reads heads.json and writes
# its tables there through LTH_RESULTS). Means, control, alpha and patched items as in E1.
# Start it detached:
#   setsid nohup scripts/pipeline_consensus.sh > logs/consensus/pipeline.log 2>&1 < /dev/null &
# LTH_RESULTS=results/consensus_k3 runs it on another set of identify_consensus.py, with its
# logs in logs/consensus_k3. LTH_SETS="logithead jhead" patches only those head sets.
# One job queue on GPUs 0, 2, 3, 5 (model loads 90 s apart), then the shares merged into
# results/consensus/patching_<family>.csv and its answers. Writes logs/consensus/pipeline.done.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
GPUS=(0 2 3 5)
export LTH_RESULTS=${LTH_RESULTS:-results/consensus}
LOGS=logs/$(basename "$LTH_RESULTS")
SETS=${LTH_SETS:-logithead jhead jhead_only}
mkdir -p $LOGS

say() { echo "[$(date '+%F %T')] $*"; }

next_job() {  # pop the first line of queue file $1
    flock "$1.lock" bash -c 'head -n1 "$0"; sed -i 1d "$0"' "$1"
}

worker() {  # gpu queue
    local job family share
    while job=$(next_job "$2") && [ -n "$job" ]; do
        flock $LOGS/load.lock sleep 90
        read -r family share tasks <<< "$job"
        $PY scripts/patch.py --gpu "$1" --family "$family" --tasks $tasks --part "$share" --sets $SETS \
            > "$LOGS/patch_$family.part-$share.log" 2>&1
        say "GPU $1: $family $share exit $?"
    done
}

[ -f $LTH_RESULTS/heads.json ] || { say "no $LTH_RESULTS/heads.json"; exit 1; }
cat > $LOGS/queue <<'EOF'
simple a antonym synonym singular-plural present-past
simple b country-capital country-currency landmark-country park-country
simple c product-company next_item prev_item
simple d english-french english-german english-spanish
simple-wrong a antonym synonym singular-plural present-past
simple-wrong b country-capital country-currency landmark-country park-country
simple-wrong c product-company next_item prev_item
simple-wrong d english-french english-german english-spanish
EOF
lanes=()
for gpu in "${GPUS[@]}"; do
    worker "$gpu" $LOGS/queue &
    lanes+=($!)
done
wait "${lanes[@]}"

$PY - <<'PY'
import gzip, os, shutil
from pathlib import Path

import pandas as pd

results = Path(os.environ["LTH_RESULTS"])
for family in ("simple", "simple-wrong"):
    parts = sorted(results.glob(f"patching_{family}.part-*.csv"))
    answers = sorted((results / "answers").glob(f"patching_{family}.part-*.jsonl.gz"))
    assert len(parts) == 4, parts
    pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(results / f"patching_{family}.csv", index=False)
    with gzip.open(results / "answers" / f"patching_{family}.jsonl.gz", "wb") as out:
        for p in answers:
            with gzip.open(p, "rb") as f:
                shutil.copyfileobj(f, out)
    for p in parts + answers:
        p.unlink()
    print("merged", family)
PY
say "merge exit $?"
touch $LOGS/pipeline.done
