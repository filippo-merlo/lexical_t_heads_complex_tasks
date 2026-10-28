#!/usr/bin/env bash
# All causal checks (scripts/patch.py), with their answers, once heads.json is final. Start
# it detached:
#   setsid nohup scripts/patch_all.sh > logs/patch_all.log 2>&1 < /dev/null &
# Lanes are balanced by the run times of 2026-10-28: Set Game's two reasoning-mode checks
# (about 5 h each) get a GPU each.
#   GPU 5: complex, then Set Game at the pre-answer position
#   GPU 0: Operator Function (both positions), then the simple tasks
#   GPU 2: Tapatan (both positions)
#   GPU 3: Set Game at the last prompt token
# Writes logs/patch_all.done at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

say() { echo "[$(date '+%F %T')] $*"; }

patch() {  # gpu family [level]
    local name=$2${3:+_${3//\//-}}
    $PY scripts/patch.py --gpu "$1" --family "$2" ${3:+--levels "$3"} > "logs/patch_$name.log" 2>&1
    say "$name exit $?"
}

( patch 5 complex; patch 5 reasoning-pre set_game/easy ) &
LANES=($!)
sleep 90
( patch 0 reasoning operator_func/easy; patch 0 reasoning-pre operator_func/easy; patch 0 simple ) &
LANES+=($!)
sleep 90
( patch 2 reasoning tapatan/easy; patch 2 reasoning-pre tapatan/easy ) &
LANES+=($!)
sleep 90
( patch 3 reasoning set_game/easy ) &
LANES+=($!)
wait "${LANES[@]}"
say "done"
touch logs/patch_all.done
