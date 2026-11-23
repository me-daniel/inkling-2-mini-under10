#!/bin/sh
# v1 release check for M0rk: the same S score on unpatched upstream llama.cpp (the sibling's clean c661033 build, which
# contains the 690b87d loop-scale fix). GPU lock only. Reading: |upstream - ours| should be at backend-noise level
# (the sibling measured up to 0.0004).
set -eu
. "$(dirname "$0")/lib.sh"
trap 'gpu_done' EXIT
BIN="$SIB/vendor/llama.cpp-upstream/build/bin"
score m0rk-upstream-S "$M/tmp/m0rk.gguf" S
$PY "$ROOT/analysis/kltable.py" > /dev/null
$PY - <<PYEOF | tee "$ROOT/results/step2/upstream-m0rk.txt"
import sys; sys.path.insert(0, "$ROOT/analysis")
from compare import load, paired
k = load(); a, b = k["m0rk-upstream-S"], k["m0rk-S"]
d, lo, hi = paired(a["chunks"], b["chunks"])
print(f"S m0rk on upstream c661033 vs our build fcaddf97a: {a['chunks'].mean():.5f} vs {b['chunks'].mean():.5f}, diff {d:+.5f} [{lo:+.5f}, {hi:+.5f}], top-1 {a['top']:.2f}% vs {b['top']:.2f}%")
PYEOF
