#!/bin/sh
# Q2_0 down donors for every layer whose Gram exists: GPTQ at ALPHA (chosen in G1 on the held-out calibration Gram,
# never on S/T) and the matched diag control. models/down/{gptq,diag}-L<l>.gguf. Resumes. LAYERS overrides the order.
set -eu
. "$(dirname "$0")/lib.sh"
BF16="$SIB/models/Inkling-2-Mini-BF16.gguf"; mkdir -p "$M/down"
for l in ${LAYERS:-12 13 14 15 16 17 0 1 2 3 4 5 6 7 8 9 10 11 18 19 20 21 22 23}; do
    for mode in gptq diag; do
        f="$M/down/$mode-L$l.gguf"; [ -f "$f" ] && continue
        ls "$M"/grams/fit-*/"blk.$l.ffn_down_exps.weight.gram.f32" > /dev/null 2>&1 || continue
        mem_wait; step "encode $mode L$l"
        $PY "$ROOT/scripts/gptq_q2.py" "$BF16" "$f" --layer "$l" --mode "$mode" --alpha "$ALPHA" --grams "$M"/grams/fit-* >> "$L/encode.log"
    done
done
step "encode done"
