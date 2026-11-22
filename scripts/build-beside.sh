#!/bin/sh
# Build beside the sibling's long CPU lock (rule agreed 22 Nov ~12:30): one build at a time, only while no
# sibling gram capture (llama-imatrix) runs and memory_pressure shows >= 40% free at start.
set -eu
. "$(dirname "$0")/lib.sh"
for n in "$@"; do
    out="$M/tmp/$n.gguf"; [ -f "$out" ] && continue
    while pgrep -f "inkling-2-mini-quant.*llama-imatrix" >/dev/null || \
          [ "$(memory_pressure | awk '/free percentage/ {sub("%","",$NF); print $NF}')" -lt 40 ]; do sleep 30; done
    step "build $n"; $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/$n.json" "$out"
done
