#!/bin/sh
# GPTQ v2, after the captures of scripts/gptq2.sh, with the pooled alpha rule (CHANGELOG 05:10). Strictly sequential:
# conversation sweep, pooled choice, encode the 17 Q2_0 down layers, build X3, score S.
set -eu
. "$(dirname "$0")/lib.sh"
BF16="$SIB/models/Inkling-2-Mini-BF16.gguf"; R="$ROOT/results"; Q2="0 1 2 3 4 6 7 10 12 13 14 15 16 20 21 22 23"
trap 'cpu_give; gpu_done' EXIT INT TERM
[ -f "$M/grams/x2-fit/done" ] && [ -f "$R/alpha-wt.tsv" ] || { echo "needs the x2-fit Grams and results/alpha-wt.tsv"; exit 1; }
if [ ! -f "$R/alpha-conv.tsv" ]; then
    for l in 0 12; do cpu_take; $PY "$ROOT/scripts/gptq_shrink.py" "$BF16" "$M/grams/x2-fit" "$M/grams/eval" "$l" mps 0.25 1.0 4.0 16.0 >> "$R/alpha-conv.tsv.part"; cpu_give; done
    mv "$R/alpha-conv.tsv.part" "$R/alpha-conv.tsv"
fi
ALPHA=$($PY -c "
import csv, collections
m = collections.defaultdict(list)
for f in ('$R/alpha-wt.tsv', '$R/alpha-conv.tsv'):
    for r in csv.reader(open(f), delimiter='\t'):
        if r and r[0] != 'layer': m[r[1]].append(float(r[3]))
means = {k: sum(v) / len(v) for k, v in m.items()}
best = min((v, k) for k, v in means.items() if k != 'diag')
print(best[1] if best[0] < means['diag'] else 'none')")
{ echo "pooled mean held-out error by alpha (layers 0, 12; WikiText train + conversations):"; $PY -c "
import csv, collections
m = collections.defaultdict(list)
for f in ('$R/alpha-wt.tsv', '$R/alpha-conv.tsv'):
    for r in csv.reader(open(f), delimiter='\t'):
        if r and r[0] != 'layer': m[r[1]].append(float(r[3]))
for k, v in m.items(): print(f'  {k}: {sum(v)/len(v):.5f}')"; echo "choice: $ALPHA"; } | tee "$R/alpha-pooled.txt"
[ "$ALPHA" = none ] && { step "no alpha beats diag on the pooled held-out texts; stop"; exit 0; }
spec="$ROOT/configs/x3.json"; printf '{"down": {' > "$spec"; sep=""
for l in $Q2; do
    f="$M/down/g2p-L$l.gguf"
    [ -f "$f" ] || { cpu_take; step "encode L$l (alpha $ALPHA)"; $PY "$ROOT/scripts/gptq_q2.py" "$BF16" "$f" --layer "$l" --mode gptq --alpha "$ALPHA" --grams "$M/grams/x2-fit" >> "$L/encode.log"; cpu_give; }
    printf '%s"%s": "g2p"' "$sep" "$l" >> "$spec"; sep=", "
done
echo '}}' >> "$spec"
out="$M/tmp/x3.gguf"; [ -f "$out" ] || { cpu_take; $PY "$ROOT/scripts/udmix.py" "$spec" "$out"; cpu_give; }
score x3-S "$out" S
$PY "$ROOT/analysis/kltable.py" > /dev/null; step "gptq2b done"
