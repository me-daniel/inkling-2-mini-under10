#!/bin/sh
# Non-expert donor with UD-IQ2_XXS's exact per-tensor types (configs/ne-ud.tensor-types, read from its header) but
# our own llama-quantize encoding. Experts Q2_0 (never taken from this pool).
set -eu
. "$(dirname "$0")/lib.sh"
out="$M/pools/ne-ud.gguf"; [ -f "$out" ] && exit 0
set -- --tensor-type "ffn_gate_up_exps=q2_0" --tensor-type "ffn_down_exps=q2_0"
while IFS= read -r o; do [ -n "$o" ] && set -- "$@" --tensor-type "$o"; done < "$ROOT/configs/ne-ud.tensor-types"
mem_wait; step "quantize ne-ud ($# args)"
"$BIN/llama-quantize" --imatrix "$SIB/models/ggml-org/imatrix.gguf" "$@" "$SIB/models/Inkling-2-Mini-BF16.gguf" "$out.part" Q8_0 > "$L/quant-ne-ud.log" 2>&1
mv "$out.part" "$out"; step "ne-ud done"
