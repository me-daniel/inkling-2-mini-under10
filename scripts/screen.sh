#!/bin/sh
# Screen udmix bundles on the first 8 chunks of S (paired with the same chunks of X2's full-S run).
# QUEUE="name ..." reads configs/<name>.json. Builds are deleted after scoring unless KEEP=1.
set -eu
. "$(dirname "$0")/lib.sh"
for n in $QUEUE; do
    out="$M/tmp/$n.gguf"
    [ -f "$out" ] || { mem_wait; $PY "$ROOT/scripts/udmix.py" "$ROOT/configs/$n.json" "$out"; }
    score "$n-S8" "$out" S --chunks 8
    [ "${KEEP:-0}" = 1 ] || rm -f "$out"
done
$PY "$ROOT/analysis/kltable.py" > /dev/null
step "screen done: $QUEUE"
