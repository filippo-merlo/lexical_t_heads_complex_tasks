#!/usr/bin/env bash
# Fetch the code and data of the two source studies, pinned.
#   third_party/Lexical_Task_Representations  Yang et al. (2026): Todd datasets, instruction
#                                             templates, term lists, control pairs
#   third_party/rules-vs-examples             Fu et al. (2026): prompt builders and parsers
#   data/fu/                                  Fu et al.'s released test and demonstration data
# Usage: scripts/fetch_sources.sh   (from the repository root)
set -euo pipefail
cd "$(dirname "$0")/.."

YANG_URL=https://github.com/jojozyang/Lexical_Task_Representations.git
YANG_COMMIT=f81ef39
FU_URL=https://github.com/tinlaboratory/rules-vs-examples.git
FU_COMMIT=e276b4a

clone() {  # url dir commit
    if [ ! -d "$2" ]; then git clone -q "$1" "$2"; fi
    if [ -n "$3" ]; then git -C "$2" checkout -q "$3"; fi
    echo "$2 at $(git -C "$2" rev-parse HEAD)"
}

mkdir -p third_party data/fu
clone "$YANG_URL" third_party/Lexical_Task_Representations "$YANG_COMMIT"
clone "$FU_URL" third_party/rules-vs-examples "$FU_COMMIT"

if [ ! -f data/fu/dataset.zip ]; then
    .venv/bin/python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('tin-lab/rules_vs_examples', 'dataset.zip', repo_type='dataset', local_dir='data/fu')"
fi
(cd data/fu && unzip -q -n dataset.zip)
echo "data/fu: $(find data/fu -type f | wc -l) files"
