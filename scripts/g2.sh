#!/bin/sh
# G2 (CHANGELOG 2026-11-21, item 4): two builds that differ only in the down tensors of B3 (layers 12-17):
# GPTQ-Q2_0 against the matched diag control from the same Grams. Base allocation: configs/m0.json (UD-IQ2_XXS's
# per-layer types) with all six B3 down tensors as Q2_0 from models/down/<mode>-L12..17.gguf. Both scored on S (16 chunks). Needs the layer 12-17 Grams.
set -eu
. "$(dirname "$0")/lib.sh"
BF16="$SIB/models/Inkling-2-Mini-BF16.gguf"; ALPHA=${ALPHA:-1.0}
mkdir -p "$M/down" "$M/tmp" "$ROOT/configs"
for l in 12 13 14 15 16 17; do
    for mode in gptq diag; do
        f="$M/down/$mode-L$l.gguf"; [ -f "$f" ] && continue
        step "encode $mode L$l"
        $PY "$ROOT/scripts/gptq_q2.py" "$BF16" "$f" --layer "$l" --mode "$mode" --alpha "$ALPHA" --grams "$M"/grams/fit-* >> "$L/encode.log"
    done
done
for mode in gptq diag; do
    spec="$ROOT/configs/g2-$mode.json"
    $PY -c "import json,sys; s=json.load(open(sys.argv[1])); s['down']={'0-4':'q2opt','5':'iq4_nl','6-7':'q2opt','8-9':'iq4_nl','10':'q2opt','11':'iq4_nl','12-17':sys.argv[2],'18-19':'iq4_nl','20-23':'q2opt'}; json.dump(s,open(sys.argv[3],'w'))" \
        "$ROOT/configs/m0.json" "$mode" "$spec"
    out="$M/tmp/g2-$mode.gguf"
    [ -f "$out" ] || { mem_wait; $PY "$ROOT/scripts/assemble.py" "$spec" "$out"; }
    score "g2-$mode-S" "$out" S
    rm -f "$out"
done
$PY "$ROOT/analysis/kltable.py" > /dev/null
step "g2 done"
