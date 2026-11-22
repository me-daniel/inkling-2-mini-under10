#!/bin/sh
# Donor pools for the allocation (CPU only, about 4 min each). Experts not under study are Q2_0 to keep files small.
#   gu-<type>: ffn_gate_up_exps at <type>; everything else as noted
#   ne-q5 / ne-q4: the non-expert matrices that are Q6_K in the sibling's cand.gguf, one step and two steps lower
set -eu
. "$(dirname "$0")/lib.sh"
P="$M/pools"; mkdir -p "$P"
IMAT="$SIB/models/ggml-org/imatrix.gguf"; BF16="$SIB/models/Inkling-2-Mini-BF16.gguf"
ATT='blk\.[0-9]+\.attn_(q|k|v|output|gate|rel|sink_proj|k_conv_proj|v_conv_proj)\.weight'
quant() { # name overrides...
    out="$P/$1.gguf"; shift; [ -f "$out" ] && return 0
    mem_wait
    step "quantize $(basename "$out")"
    tt=""; for o in "$@"; do tt="$tt --tensor-type $o"; done
    # shellcheck disable=SC2086
    "$BIN/llama-quantize" --imatrix "$IMAT" $tt "$BF16" "$out.part" Q8_0 > "$L/quant-$(basename "$out" .gguf).log" 2>&1
    mv "$out.part" "$out"
}
for t in iq2_xs iq2_s iq1_s iq1_m; do quant "gu-$t" "ffn_gate_up_exps=$t" "ffn_down_exps=q2_0"; done
quant ne-q5 "ffn_gate_up_exps=q2_0" "ffn_down_exps=q2_0" "$ATT=q5_k" "token_embd\.weight=q4_k" "output\.weight=q5_k"
quant ne-q4 "ffn_gate_up_exps=q2_0" "ffn_down_exps=q2_0" "$ATT=q4_k" "token_embd\.weight=q4_k" "output\.weight=q4_k"
step "pools done"
