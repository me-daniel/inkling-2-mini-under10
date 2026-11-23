#!/bin/sh
# Off-domain check (CHANGELOG 23 Nov 11:15) on the sibling's battery A, one domain at a time as its reference .kld
# appears (it regenerates, we only read). Writes results/batA/done-<domain> when a domain is scored, so the sibling
# can delete that .kld. Resumes.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'gpu_done' EXIT
R="$ROOT/results/batA"; mkdir -p "$R"; E="$M/ext"
for d in python rust german math; do
    [ -f "$R/done-$d" ] && continue
    until [ -f "$SIB/models/logits/batA-$d-P3.kld" ]; do sleep 60; done
    step "battery A $d: reference present"
    for spec in "c1:$M/tmp/c1.gguf" "m0rk:$M/tmp/m0rk.gguf" "c1r:$M/tmp/c1r.gguf" "GSQ-RCO-IQ2_XS:$E/GSQ-RCO-IQ2_XS.gguf" \
                "AD-IQ2_M:$E/AD-IQ2_M.gguf" "UD-IQ2_XXS:$E/UD-IQ2_XXS.gguf" "UD-Q2_K_XL:$E/UD-Q2_K_XL.gguf"; do
        name=${spec%%:*}; f=${spec#*:}
        [ -f "$f" ] || { step "skip $name (no file yet)"; continue; }
        score "batA-$d-$name" "$f" "batA-$d"
    done
    date '+%H:%M:%S' > "$R/done-$d"; step "battery A $d: done, the sibling may delete its .kld"
done
$PY "$ROOT/analysis/kltable.py" > /dev/null
$PY - <<PYEOF | tee "$R/compare.txt"
import sys; sys.path.insert(0, "$ROOT/analysis")
from compare import load, paired
k = load()
for d in ("python", "rust", "german", "math"):
    for b in ("GSQ-RCO-IQ2_XS", "AD-IQ2_M", "UD-IQ2_XXS", "UD-Q2_K_XL", "m0rk", "c1r"):
        a, c = k.get(f"batA-{d}-c1"), k.get(f"batA-{d}-{b}")
        if a is None or c is None: continue
        x, lo, hi = paired((c["chunks"] if b == "c1r" else a["chunks"]), (a["chunks"] if b == "c1r" else c["chunks"]))
        lab = f"c1r - c1" if b == "c1r" else f"c1 - {b}"
        print(f"{d:7s} {lab:22s} {x:+.5f} [{lo:+.5f}, {hi:+.5f}]  KL c1 {a['chunks'].mean():.5f} vs {c['chunks'].mean():.5f}  ratio c1/other {a['chunks'].mean() / c['chunks'].mean():.3f}")
PYEOF
step "battery A done"
