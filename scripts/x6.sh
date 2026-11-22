#!/bin/sh
# small-job rule (agreed 22 Nov 09:05): per-tensor rescales (< 1 GB, < 1 min) run beside a held CPU lock, behind mem_wait.
# X6 (CHANGELOG 08:20): UD's own tensors in X2 rescaled per row to gain 1; X6d (17 Q2_0 downs), X6dg (+ 24 gate_up).
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
UD="$M/ext/mainline/UD-IQ2_XXS.gguf"; Q2="0 1 2 3 4 6 7 10 12 13 14 15 16 20 21 22 23"
for l in $Q2; do f="$M/down/udg-L$l.gguf"; [ -f "$f" ] || { mem_wait; $PY "$ROOT/scripts/degain2.py" "$UD" "$f" "$l" --kind ffn_down_exps >> "$L/degain2.log"; }; done
for l in $(seq 0 23); do f="$M/down/udgu-L$l.gguf"; [ -f "$f" ] || { mem_wait; $PY "$ROOT/scripts/degain2.py" "$UD" "$f" "$l" --kind ffn_gate_up_exps >> "$L/degain2.log"; }; done
cat "$L/degain2.log"
$PY -c "
import json; Q2=[0,1,2,3,4,6,7,10,12,13,14,15,16,20,21,22,23]
json.dump({'down': {str(l): 'udg' for l in Q2}}, open('$ROOT/configs/x6d.json', 'w'))
json.dump({'down': {str(l): 'udg' for l in Q2}, 'gate_up': {str(l): 'udgu' for l in range(24)}}, open('$ROOT/configs/x6dg.json', 'w'))"
for n in x6d x6dg; do
    out="$M/tmp/$n.gguf"; [ -f "$out" ] || { cpu_take; $PY "$ROOT/scripts/udmix.py" "$ROOT/configs/$n.json" "$out"; cpu_give; }
    score "$n-S" "$out" S
done
$PY "$ROOT/analysis/kltable.py" > /dev/null; step "x6 done"
