#!/bin/sh
# v1-frame stability battery on M0rk under the GPU lock, judged against the sibling's Q8_0 reference run
# (../inkling-2-mini-quant/results/pilot/stability-ref.json, same battery and thresholds).
set -eu
. "$(dirname "$0")/lib.sh"
trap 'gpu_done' EXIT
gpu_wait; step "stability m0rk"
[ -f "$ROOT/results/stability/stability-m0rk.json" ] || $PY "$ROOT/scripts/stability.py" m0rk "$M/tmp/m0rk.gguf"
gpu_done
$PY - <<PYEOF | tee "$ROOT/results/stability/verdict-m0rk.md"
import json, sys; sys.path.insert(0, "$ROOT/scripts")
from stability import judge
v = judge(json.load(open("$SIB/results/pilot/stability-ref.json")), json.load(open("$ROOT/results/stability/stability-m0rk.json")))
print("| check | Q8_0 reference | M0rk | rule | result |\n|---|---|---|---|---|")
for c in v["checks"]: print(f"| {c['name']} | {c['ref']} | {c['cand']} | {c['rule']} | {'pass' if c['pass'] else 'FAIL'} |")
print(f"\nverdict: {'PASS' if v['pass'] else 'FAIL'}")
PYEOF
step "stability done"
