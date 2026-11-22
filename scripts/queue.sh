#!/bin/sh
# Build and score allocations on S, one at a time: QUEUE="name1 name2 ..." reads configs/<name>.json, writes
# models/tmp/<name>.gguf, scores it as <name>-S (16 chunks), keeps the file only if KEEP=1. Resumes per name.
set -eu
. "$(dirname "$0")/lib.sh"
mkdir -p "$M/tmp"
for n in $QUEUE; do
    out="$M/tmp/$n.gguf"
    if [ ! -f "$out" ]; then mem_wait; $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/$n.json" "$out"; fi
    score "$n-S" "$out" S
    [ "${KEEP:-0}" = 1 ] || rm -f "$out"
done
$PY "$ROOT/analysis/kltable.py" > /dev/null
step "queue done: $QUEUE"
