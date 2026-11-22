#!/bin/sh
# In-distribution GPTQ test (after G2 failed): capture down Grams on X2 itself for its 17 Q2_0 down layers
# (240 chunks of calib-fit), encode layer 12 as GPTQ and as the matched diag control, splice each into X2, score S.
set -eu
. "$(dirname "$0")/lib.sh"
X2="$M/tmp/x2.gguf"; D="$M/grams/x2-fit"; BF16="$SIB/models/Inkling-2-Mini-BF16.gguf"
if [ ! -f "$D/done" ]; then
    mkdir -p "$D"; mem_wait; gpu_wait; step "capture x2-fit"
    GRAM_DIR="$D" GRAM_MATCH='^blk\.(0|1|2|3|4|6|7|10|12|13|14|15|16|20|21|22|23)\.ffn_down_exps\.weight$' "$BIN/llama-imatrix" \
        -m "$X2" -f "$ROOT/data/calib-fit.txt" --parse-special --chunks 240 -o "$D/imatrix.gguf" -c 512 -ngl 99 > "$L/gram-x2-fit.log" 2>&1
    gpu_done; touch "$D/done"
fi
for mode in gptq diag; do
    f="$M/down/x2$mode-L12.gguf"; [ -f "$f" ] && continue
    mem_wait; step "encode x2$mode L12"
    $PY "$ROOT/scripts/gptq_q2.py" "$BF16" "$f" --layer 12 --mode "$mode" --alpha 0.25 --grams "$D" >> "$L/encode.log"
done
for mode in gptq diag; do
    echo "{\"down\": {\"12\": \"x2$mode\"}}" > "$ROOT/configs/x2-L12$mode.json"
    out="$M/tmp/x2-L12$mode.gguf"
    [ -f "$out" ] || { mem_wait; $PY "$ROOT/scripts/udmix.py" "$ROOT/configs/x2-L12$mode.json" "$out"; }
    score "x2-L12$mode-S" "$out" S
    rm -f "$out"
done
$PY "$ROOT/analysis/kltable.py" > /dev/null; step "x2gram done"
