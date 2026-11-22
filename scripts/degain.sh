#!/bin/sh
# Attenuation test (CHANGELOG 07:15): de-attenuated q2opt donors for the 17 Q2_0 down slots, then M0u and X5 on S.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
Q2="0 1 2 3 4 6 7 10 12 13 14 15 16 20 21 22 23"
for l in $Q2; do
    f="$M/down/q2u-L$l.gguf"; [ -f "$f" ] && continue
    cpu_take; $PY "$ROOT/scripts/degain.py" "$SIB/models/q2opt/L$l.gguf" "$f" "$l" >> "$L/degain.log"; cpu_give
done
grep gain "$L/degain.log" | tail -17
$PY -c "
import json; Q2=[0,1,2,3,4,6,7,10,12,13,14,15,16,20,21,22,23]
m = json.load(open('$ROOT/configs/m0.json')); d = {}
for k, v in m['down'].items():
    lo, _, hi = k.partition('-')
    for l in range(int(lo), int(hi or lo) + 1): d[str(l)] = 'q2u' if v == 'q2opt' else v
m['down'] = d; json.dump(m, open('$ROOT/configs/m0u.json', 'w'))
json.dump({'down': {str(l): 'q2u' for l in Q2}}, open('$ROOT/configs/x5.json', 'w'))"
out="$M/tmp/x5.gguf"; [ -f "$out" ] || { cpu_take; $PY "$ROOT/scripts/udmix.py" "$ROOT/configs/x5.json" "$out"; cpu_give; }
score x5-S "$out" S; rm -f "$out"
out="$M/tmp/m0u.gguf"; [ -f "$out" ] || { cpu_take; $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/m0u.json" "$out"; cpu_give; }
score m0u-S "$out" S; rm -f "$out"
$PY "$ROOT/analysis/kltable.py" > /dev/null; step "degain done"
