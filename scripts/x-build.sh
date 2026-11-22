#!/bin/sh
# Builds on Unsloth's own UD-IQ2_XXS (mainline layout, sha256-checked) as the base, replacing only down projections.
#   X1: every Q2_0 down -> our GPTQ-Q2_0 (alpha 0.25), every NVFP4 down -> IQ4_NL (llama-quantize, ggml-org imatrix)
#   X2: only the NVFP4 downs -> IQ4_NL; UD's own Q2_0 downs stay
# Replaced down tensors are unscaled, so their per-expert .scale tensors are dropped. Both all-Metal. Then S scores.
set -eu
. "$(dirname "$0")/lib.sh"
UD="$M/ext/mainline/UD-IQ2_XXS.gguf"; P3="$SIB/models/pool-3bit.gguf"; LIMIT=8486236352
NV="5 8 9 11 17 18 19"; Q2="0 1 2 3 4 6 7 10 12 13 14 15 16 20 21 22 23"
build() { # name with-gptq(0/1)
    out="$M/tmp/$1.gguf"; [ -f "$out" ] && return 0
    set -- "$1" "$2"; args=""
    for l in $NV; do args="$args --take $P3 ^blk\\.$l\\.ffn_down_exps\\.weight\$ --drop ^blk\\.$l\\.ffn_down_exps\\.scale\$"; done
    if [ "$2" = 1 ]; then
        for l in $Q2; do
            args="$args --take $M/down/gptq-L$l.gguf ^blk\\.$l\\.ffn_down_exps\\.weight\$"
            [ "$l" -le 19 ] && args="$args --drop ^blk\\.$l\\.ffn_down_exps\\.scale\$"
        done
    fi
    mem_wait; step "build $1"
    # shellcheck disable=SC2086
    $PY "$SIB/scripts/splice.py" "$UD" "$out" $args --verify > "$out.json"
    sz=$(stat -f %z "$out"); echo "$1: $sz bytes (limit $LIMIT)"
    [ "$sz" -le "$LIMIT" ] || { rm -f "$out"; echo "$1 over the limit, deleted"; exit 1; }
}
build x1 1; score x1-S "$M/tmp/x1.gguf" S
build x2 0; score x2-S "$M/tmp/x2.gguf" S
$PY "$ROOT/analysis/kltable.py" > /dev/null; step "x done"
