#!/bin/sh
# Diagnostic (CHANGELOG 11:45): M0rk with UD's own non-expert tensors. Remaining non-expert share in our best stack.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
out="$M/tmp/m0rk-udne.gguf"; [ -f "$out" ] || { mem_wait; /usr/bin/time -l $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/m0rk-udne.json" "$out" 2> "$L/time-assemble.txt" || { cat "$L/time-assemble.txt"; exit 1; }; grep -E "real|maximum resident" "$L/time-assemble.txt"; }  # small-job rule: I/O-bound splice
score m0rk-udne-S "$out" S
rm -f "$out"; $PY "$ROOT/analysis/kltable.py" > /dev/null; step "m0rk-udne done"
