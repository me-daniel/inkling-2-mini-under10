#!/bin/sh
# Rebuilds c1 with the sibling's splice.py at ca5b1e4 (code-review fixes) and checks it reproduces c1's sha256.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give' EXIT
out="$M/tmp/c1-recheck.gguf"; rm -f "$out" "$out.manifest.json"
cpu_take; step "rebuild c1 with splice.py $(git -C "$SIB" rev-parse --short HEAD)"
$PY "$ROOT/scripts/assemble.py" "$ROOT/configs/c1.json" "$out" --over
cpu_give
h=$(shasum -a 256 "$out" | cut -c1-64); want=$(cut -c1-64 "$ROOT/results/step3/c1.sha256")
echo "splice $(git -C "$SIB" rev-parse --short HEAD): $h $([ "$h" = "$want" ] && echo MATCH || echo "DIFFERS from $want")" | tee "$ROOT/results/c1/splice-recheck.txt"
rm -f "$out" "$out.manifest.json"
