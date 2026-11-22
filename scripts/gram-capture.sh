#!/bin/sh
# Down-projection Grams for GPTQ (patch 0008). Forward pass at P = 3 on the sibling's all-Metal cand.gguf (11.5 GB),
# chosen to fit beside the sibling's own GPU jobs; every capture uses the same forward model.
#   eval   : layers 0 and 12 on data/calib-eval.txt (36 held-out conversations), for G1 only
#   fit-g1 : layers 0 and 12 on data/calib-fit.txt (prose + 137 base conversations), for G1
#   fit-<a>-<b>: the other layers in groups of six, only after G1 passes (run with STAGE=all)
set -eu
. "$(dirname "$0")/lib.sh"
MODEL="$SIB/models/cand.gguf"
run() { # name text regex [extra args]
    D="$M/grams/$1"; [ -f "$D/done" ] && return 0
    n=$1 txt=$2 rx=$3; shift 3
    mkdir -p "$D"; gpu_wait; step "capture $n"
    GRAM_DIR="$D" GRAM_MATCH="$rx" "$BIN/llama-imatrix" -m "$MODEL" -f "$ROOT/data/$txt" --parse-special "$@" \
        -o "$D/imatrix.gguf" -c 512 -ngl 99 > "$L/gram-$n.log" 2>&1
    gpu_done; touch "$D/done"
}
run eval calib-eval.txt '^blk\.(0|12)\.ffn_down_exps\.weight$'
# capped at 240 chunks (123K tokens) to fit the evening's shared GPU; fit-0-12 decides G1 (layers 0 and 12)
run fit-0-12 calib-fit.txt '^blk\.([0-9]|1[0-2])\.ffn_down_exps\.weight$' --chunks 240
[ "${STAGE:-g1}" = all ] && run fit-13-23 calib-fit.txt '^blk\.(1[3-9]|2[0-3])\.ffn_down_exps\.weight$' --chunks 240
step "captures done (${STAGE:-g1})"
