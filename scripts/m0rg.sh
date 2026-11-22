#!/bin/sh
# Own path (CHANGELOG 09:15): M0rg = M0uu with the mean-aware refit gate_up (M0r's donors) rescaled per row to gain 1,
# i.e. the refit with its attenuation removed. Read against M0uu (plain llama-quantize gate_up at gain 1).
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
for spec in "iq1_s:0 1 2 3 4 6" "iq2_xxs:5 7 8 12 13 14 18" "iq2_xs:9 10 11 15 17 22" "iq2_s:16 19 20 21 23"; do
    ty=${spec%%:*}; layers=${spec#*:}
    for l in $layers; do f="$M/down/rfg-L$l.gguf"; [ -f "$f" ] || { mem_wait; $PY "$ROOT/scripts/degain2.py" "$M/meanfit/rf-$ty.gguf" "$f" "$l" --kind ffn_gate_up_exps >> "$L/degain2-rfg.log"; }; done
done
cat "$L/degain2-rfg.log"
$PY -c "
import json; m = json.load(open('$ROOT/configs/m0uu.json')); m['gate_up'] = {str(l): 'rfg' for l in range(24)}
json.dump(m, open('$ROOT/configs/m0rg.json', 'w'))"
out="$M/tmp/m0rg.gguf"; [ -f "$out" ] || { cpu_take; $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/m0rg.json" "$out"; cpu_give; }
score m0rg-S "$out" S
$PY "$ROOT/analysis/kltable.py" > /dev/null; step "m0rg done"
