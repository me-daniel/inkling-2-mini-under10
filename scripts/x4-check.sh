#!/bin/sh
set -eu
. "$(dirname "$0")/lib.sh"
trap cpu_give EXIT
cpu_take; $PY "$ROOT/scripts/down_err.py" 0 12 > "$ROOT/results/down-err-x4.tsv"; cpu_give
cat "$ROOT/results/down-err-x4.tsv"
