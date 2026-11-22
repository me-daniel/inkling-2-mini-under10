#!/bin/sh
# Own path (CHANGELOG 12:00): our q2opt Q2_0 down at fixed per-row gain 0.93 and 1.07 inside M0rk (M0rk itself has 1.00).
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
for g in 93 107; do
    for l in 0 1 2 3 4 6 7 10 12 13 14 15 16 20 21 22 23; do
        f="$M/down/q2t$g-L$l.gguf"
        [ -f "$f" ] || { mem_wait; $PY "$ROOT/scripts/degain2.py" "$SIB/models/q2opt/L$l.gguf" "$f" "$l" --kind ffn_down_exps --target "$(echo "scale=2; $g/100" | bc)" >> "$L/degain2-gsweep.log"; }
    done
    $PY -c "
import json; m = json.load(open('$ROOT/configs/m0rk.json'))
m['down'] = {k: ('q2t$g' if v == 'q2u' else v) for k, v in m['down'].items()}
json.dump(m, open('$ROOT/configs/m0rk-g$g.json', 'w'))"
done
until ! pgrep -f "scripts/m0rk-udne[.]sh" >/dev/null && [ -f "$L/ppl-m0rk-udne-S.log" ]; do sleep 30; done  # GPU order
for g in 93 107; do
    out="$M/tmp/m0rk-g$g.gguf"; [ -f "$out" ] || { cpu_take; $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/m0rk-g$g.json" "$out"; cpu_give; }  # splice measured 256 s, 6.7 GB RSS: not a small job
    score "m0rk-g$g-S" "$out" S; rm -f "$out"
done
$PY "$ROOT/analysis/kltable.py" > /dev/null; step "gsweep done"
