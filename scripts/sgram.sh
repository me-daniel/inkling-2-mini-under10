#!/bin/sh
# Diagnostic after G2: down Grams for layers 0 and 12 on S text itself (selection set, never T1/T2), forward on X2,
# then the bias split of diag / GPTQ / UD encodings on that Gram (results/bias-split-S.tsv).
set -eu
. "$(dirname "$0")/lib.sh"
D="$M/grams/s-x2"
if [ ! -f "$D/done" ]; then
    mkdir -p "$D"; mem_wait; gpu_wait; step "capture s-x2"
    GRAM_DIR="$D" GRAM_MATCH='^blk\.(0|12)\.ffn_down_exps\.weight$' "$BIN/llama-imatrix" -m "$M/tmp/x2.gguf" \
        -f "$ROOT/data/S.txt" -o "$D/imatrix.gguf" -c 512 -ngl 99 > "$L/gram-s-x2.log" 2>&1
    gpu_done; touch "$D/done"
fi
for l in 12 0; do EVAL_DIR="$D" $PY "$ROOT/scripts/bias_split.py" "$l"; done > "$ROOT/results/bias-split-S.tsv"
cat "$ROOT/results/bias-split-S.tsv"; step "sgram done"
