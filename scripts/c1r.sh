#!/bin/sh
# Step 5 (CHANGELOG 23 Nov 08:35): c1r = c1 with full-Gram refined codes in its IQ2_S / IQ3_XXS gate_up layers. Resumes.
# Grams: our llama-imatrix on the sibling's cand.gguf over its calib-mix.txt. Refiner: the sibling's refine-codes.py,
# read-only. Each group's Gram .f32 files are deleted after its refinement (about 2.7 GB per layer).
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
C1="$M/tmp/c1.gguf"; R="$ROOT/results/c1r"; mkdir -p "$R"
capture() { # tag "layers"
    D="$M/grams/gu-$1-mix"; [ -f "$D/done" ] && return 0; mkdir -p "$D"
    RX=$(echo "$2" | tr ' ' '|')
    gpu_wait; step "capture Grams $2"
    GRAM_DIR="$D" GRAM_MATCH="^blk\\.($RX)\\.ffn_gate_up_exps\\.weight\$" "$BIN/llama-imatrix" -m "$SIB/models/cand.gguf" \
        -f "$SIB/data/calib-mix.txt" --chunks 300 -c 512 -ngl 99 --parse-special \
        --override-kv "inkling2_loop.loop.count=int:3" -o "$D/imatrix.gguf" > "$L/c1r-gram-$1.log" 2>&1
    gpu_done; touch "$D/done"
}
refine() { # tag "layers"
    out="$M/refined/c1r-$1.gguf"; [ -f "$out" ] && return 0
    cpu_take; step "refine $2"
    $PY "$SIB/scripts/refine-codes.py" --quant "$C1" --bf16 "$SIB/models/Inkling-2-Mini-BF16.gguf" --grams "$M/grams/gu-$1-mix" \
        --layers $2 --sweeps 4 --tol 1e-4 --shrink 0.1 --min-tokens 2304 --workers 4 --out "$out" > "$L/c1r-refine-$1.log" 2>&1
    cpu_give
    rm -f "$M/grams/gu-$1-mix"/*.gram.f32
}
capture A "13 16 18 19"
refine A "13 16 18 19" &
RA=$!
capture B "20 21 23"
wait $RA
refine B "20 21 23"
TAKES=""
for l in 13 16 18 19; do TAKES="$TAKES --take $M/refined/c1r-A.gguf ^blk\\.$l\\.ffn_gate_up_exps\\.weight\$"; done
for l in 20 21 23; do TAKES="$TAKES --take $M/refined/c1r-B.gguf ^blk\\.$l\\.ffn_gate_up_exps\\.weight\$"; done
out="$M/tmp/c1r.gguf"
if [ ! -f "$out" ]; then
    cpu_take; step "splice c1r"
    $PY "$SIB/scripts/splice.py" "$C1" "$out.part" $TAKES --verify > "$L/splice-c1r.log"
    mv "$out.part" "$out"; cpu_give
fi
[ "$(stat -f %z "$out")" = "$(stat -f %z "$C1")" ] || { echo "c1r size differs from c1"; exit 1; }
shasum -a 256 "$out" | tee "$R/c1r.sha256"
score c1r-S "$out" S
score final-c1r-T1-P3 "$out" T1
score final-c1r-T2-P3 "$out" T2
$PY "$ROOT/analysis/kltable.py" > /dev/null
$PY - <<PYEOF | tee "$R/c1r.txt"
import sys; sys.path.insert(0, "$ROOT/analysis")
from compare import load, paired
k = load()
for s, a, b in (("S", "c1r-S", "c1-S"), ("T1", "final-c1r-T1-P3", "final-c1-T1-P3"), ("T2", "final-c1r-T2-P3", "final-c1-T2-P3")):
    d, lo, hi = paired(k[a]["chunks"], k[b]["chunks"])
    print(f"{s} c1r vs c1: {k[a]['chunks'].mean():.5f} vs {k[b]['chunks'].mean():.5f}, diff {d:+.5f} [{lo:+.5f}, {hi:+.5f}], top-1 {k[a]['top']:.2f}% vs {k[b]['top']:.2f}%")
PYEOF
$PY "$ROOT/analysis/compare.py" c1r GSQ-RCO-IQ2_XS UD-IQ2_XXS UD-IQ1_M UD-IQ1_S | tee "$R/c1r-compare.txt"
step "c1r done"
