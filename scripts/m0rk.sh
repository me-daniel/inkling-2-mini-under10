#!/bin/sh
# Own path (CHANGELOG 10:10): M0rk = M0u with the mean-aware refit gate_up rescaled per row back to llama-quantize's
# own per-row gain (the gain constraint), not to 1. Read against M0u. GPU score only after the sibling's two runs.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
for spec in "iq1_s:0 1 2 3 4 6:$M/pools/gu-iq1_s.gguf" "iq2_xxs:5 7 8 12 13 14 18:$SIB/models/pool-2bit.gguf" \
            "iq2_xs:9 10 11 15 17 22:$M/pools/gu-iq2_xs.gguf" "iq2_s:16 19 20 21 23:$M/pools/gu-iq2_s.gguf"; do
    ty=${spec%%:*}; r=${spec#*:}; layers=${r%%:*}; ref=${r#*:}
    for l in $layers; do f="$M/down/rfk-L$l.gguf"; [ -f "$f" ] || { mem_wait; $PY "$ROOT/scripts/degain2.py" "$M/meanfit/rf-$ty.gguf" "$f" "$l" --kind ffn_gate_up_exps --target-from "$ref" >> "$L/degain2-rfk.log"; }; done
done
cat "$L/degain2-rfk.log"
$PY -c "
import json; m = json.load(open('$ROOT/configs/m0u.json')); m['gate_up'] = {str(l): 'rfk' for l in range(24)}
json.dump(m, open('$ROOT/configs/m0rk.json', 'w'))"
out="$M/tmp/m0rk.gguf"; [ -f "$out" ] || { cpu_take; $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/m0rk.json" "$out"; cpu_give; }
until [ -f "$ROOT/.go-gpu" ]; do sleep 30; done   # created once the sibling's cand-u and rp-refined-9-11 are scored
score m0rk-S "$out" S
$PY "$ROOT/analysis/kltable.py" > /dev/null; step "m0rk done"
