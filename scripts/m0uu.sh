#!/bin/sh
# Own-encoding path (no Unsloth tensors): M0uu = M0u with every other expert tensor of ours rescaled per row to gain 1:
# the 24 llama-quantize gate_up tensors (M0's types) and the 7 IQ4_NL down slots. Codes unchanged. Scored on S vs M0u.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
for spec in "iq1_s:0 1 2 3 4 6:$M/pools/gu-iq1_s.gguf" "iq2_xxs:5 7 8 12 13 14 18:$SIB/models/pool-2bit.gguf" \
            "iq2_xs:9 10 11 15 17 22:$M/pools/gu-iq2_xs.gguf" "iq2_s:16 19 20 21 23:$M/pools/gu-iq2_s.gguf"; do
    ls_=${spec#*:}; layers=${ls_%%:*}; src=${spec##*:}
    for l in $layers; do f="$M/down/ogu-L$l.gguf"; [ -f "$f" ] || { cpu_take; $PY "$ROOT/scripts/degain2.py" "$src" "$f" "$l" --kind ffn_gate_up_exps >> "$L/degain2-own.log"; cpu_give; }; done
done
for l in 5 8 9 11 17 18 19; do f="$M/down/o4u-L$l.gguf"; [ -f "$f" ] || { cpu_take; $PY "$ROOT/scripts/degain2.py" "$SIB/models/pool-3bit.gguf" "$f" "$l" --kind ffn_down_exps >> "$L/degain2-own.log"; cpu_give; }; done
cat "$L/degain2-own.log"
$PY -c "
import json; m = json.load(open('$ROOT/configs/m0u.json'))
m['gate_up'] = {str(l): 'ogu' for l in range(24)}
for l in (5, 8, 9, 11, 17, 18, 19): m['down'][str(l)] = 'o4u'
json.dump(m, open('$ROOT/configs/m0uu.json', 'w'))"
out="$M/tmp/m0uu.gguf"; [ -f "$out" ] || { cpu_take; $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/m0uu.json" "$out"; cpu_give; }
score m0uu-S "$out" S
$PY "$ROOT/analysis/kltable.py" > /dev/null; step "m0uu done"
