#!/bin/sh
# One gate_up pool at M0's per-layer types (UD-IQ2_XXS's map), quantized with a given importance matrix:
#   sh scripts/pool-map.sh NAME IMATRIX.gguf  ->  models/pools/map-NAME.gguf (down Q2_0, never used)
set -eu
. "$(dirname "$0")/lib.sh"
name=$1 imat=$2; out="$M/pools/map-$name.gguf"; [ -f "$out" ] && exit 0
mem_wait; step "quantize map-$name"
"$BIN/llama-quantize" --imatrix "$imat" \
    --tensor-type 'blk\.(0|1|2|3|4|6)\.ffn_gate_up_exps=iq1_s' \
    --tensor-type 'blk\.(5|7|8|12|13|14|18)\.ffn_gate_up_exps=iq2_xxs' \
    --tensor-type 'blk\.(9|10|11|15|17|22)\.ffn_gate_up_exps=iq2_xs' \
    --tensor-type 'blk\.(16|19|20|21|23)\.ffn_gate_up_exps=iq2_s' \
    --tensor-type 'ffn_down_exps=q2_0' "$SIB/models/Inkling-2-Mini-BF16.gguf" "$out.part" Q8_0 > "$L/quant-map-$name.log" 2>&1
mv "$out.part" "$out"; step "map-$name done"
