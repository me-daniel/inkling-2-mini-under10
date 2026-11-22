#!/bin/sh
# Diagnostic (CHANGELOG 11:45): M0rk with UD's own non-expert tensors. Remaining non-expert share in our best stack.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
out="$M/tmp/m0rk-udne.gguf"; [ -f "$out" ] || { cpu_take; $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/m0rk-udne.json" "$out"; cpu_give; }
score m0rk-udne-S "$out" S
rm -f "$out"; $PY "$ROOT/analysis/kltable.py" > /dev/null; step "m0rk-udne done"
