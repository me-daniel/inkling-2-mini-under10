#!/bin/sh
# Pooled alpha choice (CHANGELOG 05:10): sweep on the held-out conversations Gram, then the mean with the WikiText sweep.
set -eu
. "$(dirname "$0")/lib.sh"
BF16="$SIB/models/Inkling-2-Mini-BF16.gguf"; A="$ROOT/results/alpha-conv.tsv"
until [ -f "$M/grams/x2-fit/done" ]; do sleep 30; done
if [ ! -f "$A" ]; then
    for l in 0 12; do mem_wait; $PY "$ROOT/scripts/gptq_shrink.py" "$BF16" "$M/grams/x2-fit" "$M/grams/eval" "$l" mps 0.25 1.0 4.0 16.0; done > "$A.part"
    mv "$A.part" "$A"
fi
until [ -f "$ROOT/results/alpha-wt.tsv" ]; do sleep 30; done
$PY -c "
import csv, collections
m = collections.defaultdict(list)
for f in ('$ROOT/results/alpha-wt.tsv', '$A'):
    for r in csv.reader(open(f), delimiter='\t'):
        if r and r[0] != 'layer': m[r[1]].append(float(r[3]))
means = {k: sum(v) / len(v) for k, v in m.items()}
for k, v in sorted(means.items()): print(k, round(v, 5))
best = min((v, k) for k, v in means.items() if k != 'diag')
print('pooled choice:', best[1] if best[0] < means['diag'] else 'none')" | tee "$ROOT/results/alpha-pooled.txt"
