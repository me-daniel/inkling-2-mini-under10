#!/bin/sh
# Pilot capture for the GPTQ bound: down-projection Grams of layers 0 and 12 on two disjoint parts of the
# bartowski v6 prose text (fit: chunks 0-47, eval: chunks 48-71, 512 tokens each), forward pass on the IQ4_NL build.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); SIB=$(cd "$ROOT/../inkling-2-mini-quant" && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"; M="$SIB/models/Inkling-2-Mini-IQ4_NL.gguf"
for part in fit:0:48 eval:48:24; do
    name=${part%%:*}; rest=${part#*:}; from=${rest%%:*}; n=${rest#*:}
    D="$ROOT/models/gram-pilot/$name"; [ -f "$D/done" ] && continue
    mkdir -p "$D"
    GRAM_DIR="$D" GRAM_MATCH='^blk\.(0|12)\.ffn_down_exps\.weight$' "$BIN/llama-imatrix" -m "$M" \
        -f "$ROOT/data/raw/bartowski-v6_prose.txt" -o "$D/imatrix.gguf" -c 512 --from-chunk "$from" --chunks "$n" -ngl 99 \
        > "$ROOT/logs/gram-pilot-$name.log" 2>&1
    touch "$D/done"
done
echo "gram pilot done"
