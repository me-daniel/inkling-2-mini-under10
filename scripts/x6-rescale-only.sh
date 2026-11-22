#!/bin/sh
# X6 donors only (small-job rule); builds and GPU scores deferred while the sibling's two runs have priority (09:20).
set -eu
. "$(dirname "$0")/lib.sh"
UD="$M/ext/mainline/UD-IQ2_XXS.gguf"
for l in 0 1 2 3 4 6 7 10 12 13 14 15 16 20 21 22 23; do f="$M/down/udg-L$l.gguf"; [ -f "$f" ] || { mem_wait; $PY "$ROOT/scripts/degain2.py" "$UD" "$f" "$l" --kind ffn_down_exps >> "$L/degain2.log"; }; done
for l in $(seq 0 23); do f="$M/down/udgu-L$l.gguf"; [ -f "$f" ] || { mem_wait; $PY "$ROOT/scripts/degain2.py" "$UD" "$f" "$l" --kind ffn_gate_up_exps >> "$L/degain2.log"; }; done
step "x6 donors done"
