#!/bin/sh
# G1 (CHANGELOG 2026-11-21, item 3): held-out relative output error of GPTQ-Q2_0 against the matched diag control,
# layers 0 and 12, fit Gram = fit-0-12 (123K tokens), eval Gram = 36 held-out conversations. -> results/g1.tsv
set -eu
. "$(dirname "$0")/lib.sh"
out="$ROOT/results/g1.tsv"; : > "$out.part"
for l in 0 12; do
    $PY "$ROOT/scripts/gptq_shrink.py" "$SIB/models/Inkling-2-Mini-BF16.gguf" "$M/grams/fit-0-12" "$M/grams/eval" "$l" mps 0.25 1.0 4.0 >> "$out.part"
done
mv "$out.part" "$out"; cat "$out"
