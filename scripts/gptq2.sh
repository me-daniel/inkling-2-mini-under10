#!/bin/sh
# GPTQ v2 after G2 failed (CHANGELOG 2026-11-22): fit Grams captured on X2 itself (the deployment forward), shrinkage
# alpha chosen automatically on a third text (WikiText-2 train, data/calib-wt.txt, 0 13-gram overlap with S/T1/T2/T3),
# then X3 = X2 with GPTQ-Q2_0 at all 17 of its Q2_0 down slots, scored on S.
set -eu
. "$(dirname "$0")/lib.sh"
X2="$M/tmp/x2.gguf"; BF16="$SIB/models/Inkling-2-Mini-BF16.gguf"
Q2="0 1 2 3 4 6 7 10 12 13 14 15 16 20 21 22 23"
cap() { # dir text regex [extra]
    d="$M/grams/$1"; txt=$2; rx=$3; shift 3; [ -f "$d/done" ] && return 0
    mkdir -p "$d"; mem_wait; gpu_wait; step "capture $(basename "$d")"
    GRAM_DIR="$d" GRAM_MATCH="$rx" "$BIN/llama-imatrix" -m "$X2" -f "$ROOT/data/$txt" --parse-special "$@" \
        -o "$d/imatrix.gguf" -c 512 -ngl 99 > "$L/gram-$(basename "$d").log" 2>&1
    gpu_done; touch "$d/done"
}
cap wt-x2 calib-wt.txt '^blk\.(0|12)\.ffn_down_exps\.weight$' --chunks 120
cap x2-fit calib-fit.txt '^blk\.(0|1|2|3|4|6|7|10|12|13|14|15|16|20|21|22|23)\.ffn_down_exps\.weight$' --chunks 240
# alpha sweep on layers 0 and 12: fit x2-fit, judge on wt-x2; the alpha with the lowest mean error wins (no hand pick)
A="$ROOT/results/alpha-wt.tsv"
if [ ! -f "$A" ]; then
    for l in 0 12; do mem_wait; $PY "$ROOT/scripts/gptq_shrink.py" "$BF16" "$M/grams/x2-fit" "$M/grams/wt-x2" "$l" mps 0.25 1.0 4.0 16.0; done > "$A.part"
    mv "$A.part" "$A"
fi
cat "$A"
ALPHA=$($PY -c "
import csv, collections
rows = [r for r in csv.reader(open('$A'), delimiter='\t') if r and r[0] != 'layer']
m = collections.defaultdict(list)
for r in rows: m[r[1]].append(float(r[3]))
best = min((sum(v) / len(v), k) for k, v in m.items() if k != 'diag')
diag = sum(m['diag']) / len(m['diag'])
print(best[1] if best[0] < diag else 'none')")
step "alpha chosen on WikiText-2 train: $ALPHA"
[ "$ALPHA" = none ] && { step "GPTQ does not beat diag on the third text; stop"; exit 0; }
spec="$ROOT/configs/x3.json"; echo '{"down": {' > "$spec"; sep=""
for l in $Q2; do
    f="$M/down/g2v-L$l.gguf"
    [ -f "$f" ] || { mem_wait; step "encode L$l"; $PY "$ROOT/scripts/gptq_q2.py" "$BF16" "$f" --layer "$l" --mode gptq --alpha "$ALPHA" --grams "$M/grams/x2-fit" >> "$L/encode.log"; }
    printf '%s"%s": "g2v"' "$sep" "$l" >> "$spec"; sep=", "
done
echo '}}' >> "$spec"
out="$M/tmp/x3.gguf"; [ -f "$out" ] || { mem_wait; $PY "$ROOT/scripts/udmix.py" "$spec" "$out"; }
score x3-S "$out" S
$PY "$ROOT/analysis/kltable.py" > /dev/null; step "gptq2 done"
