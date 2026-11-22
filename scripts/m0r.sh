#!/bin/sh
# M0r: M0 with every gate_up tensor refit (mean-aware scales, codes unchanged), scored on S against M0 (0.0979).
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
IM="$SIB/models/ggml-org/imatrix.gguf"
for spec in "iq1_s:0,1,2,3,4,6:$M/pools/gu-iq1_s.gguf" "iq2_xxs:5,7,8,12,13,14,18:$SIB/models/pool-2bit.gguf" \
            "iq2_xs:9,10,11,15,17,22:$M/pools/gu-iq2_xs.gguf" "iq2_s:16,19,20,21,23:$M/pools/gu-iq2_s.gguf"; do
    ty=${spec%%:*}; rest=${spec#*:}; ls=${rest%%:*}; src=${rest#*:}; out="$M/meanfit/rf-$ty.gguf"
    [ -f "$out" ] || { cpu_take; step "refit $ty layers $ls"; $PY "$ROOT/scripts/meanfit.py" "$src" "$IM" "$out" --layers "$ls"; cpu_give; }
done
out="$M/tmp/m0r.gguf"; [ -f "$out" ] || { cpu_take; $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/m0r.json" "$out"; cpu_give; }
score m0r-S "$out" S
$PY "$ROOT/analysis/kltable.py" > /dev/null; step "m0r done"
