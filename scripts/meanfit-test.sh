#!/bin/sh
set -eu
. "$(dirname "$0")/lib.sh"
trap cpu_give EXIT
IM="$SIB/models/ggml-org/imatrix.gguf"; mkdir -p "$M/meanfit"
for spec in "0:$M/pools/gu-iq1_s.gguf" "5:$SIB/models/pool-2bit.gguf" "9:$M/pools/gu-iq2_xs.gguf" "16:$M/pools/gu-iq2_s.gguf"; do
    l=${spec%%:*}; src=${spec#*:}
    cpu_take; $PY "$ROOT/scripts/meanfit.py" "$src" "$IM" "$M/meanfit/test-L$l.gguf" --layers "$l-$l"; cpu_give
done
