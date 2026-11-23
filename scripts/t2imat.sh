#!/bin/sh
# Step 3c (CHANGELOG 23 Nov 03:45): M0 and its two imatrix variants on T2. Builds take the CPU lock, scores the GPU
# lock. Each file is deleted after its score. Resumes.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
R="$ROOT/results/step3"; mkdir -p "$R"
for name in m0 m0-imat-mine m0-imat-bart; do
    out="$M/tmp/$name.gguf"
    grep -q 'Mean *KLD' "$L/ppl-$name-T2.log" 2>/dev/null && continue
    [ -f "$out" ] || { cpu_take; step "assemble $name"; $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/$name.json" "$out"; cpu_give; }
    score "$name-T2" "$out" T2
    rm -f "$out"
done
$PY "$ROOT/analysis/kltable.py" > /dev/null
$PY - <<PYEOF | tee "$R/t2-imatrix.txt"
import sys; sys.path.insert(0, "$ROOT/analysis")
from compare import load, paired
k = load(); b = k["m0-T2"]["chunks"]
for n in ("m0-imat-mine", "m0-imat-bart"):
    c = k[n + "-T2"]["chunks"]; d, lo, hi = paired(c, b)
    print(f"T2 {n} vs m0: {c.mean():.5f} vs {b.mean():.5f}, diff {d:+.5f} [{lo:+.5f}, {hi:+.5f}]")
PYEOF
step "t2imat done"
