#!/bin/sh
# Held-out T3 (pre-registered in CHANGELOG 23 Nov 21:00, Daniel's go): the sibling's data/T3.txt and its Q8_0 reference
# ref-T3-P3.kld, read-only, T1 protocol (P = 3, 16 chunks of 4,096 tokens). Hashes every file before scoring, writes
# results/t3/done at the end so the sibling may delete its .kld. Resumes.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'gpu_done' EXIT
R="$ROOT/results/t3"; E="$M/ext"; mkdir -p "$R"
[ "$(shasum -a 256 "$SIB/data/T3.txt" | cut -c1-64)" = 5a26eea1323fbade49482767b9ea4de14f539262b82135ab0e0e0a917be689a7 ] || { echo "T3 hash mismatch"; exit 1; }
FILES="c1:$M/tmp/c1.gguf m0rk:$M/tmp/m0rk.gguf GSQ-RCO-IQ2_XS:$E/GSQ-RCO-IQ2_XS.gguf AD-IQ2_M:$E/AD-IQ2_M.gguf AD-IQ2_XXS-Q2:$E/AD-IQ2_XXS-Q2.gguf AD-IQ1_M:$E/AD-IQ1_M.gguf AD-IQ1_S:$E/AD-IQ1_S.gguf UD-IQ2_XXS:$E/UD-IQ2_XXS.gguf"
[ -f "$R/sha.txt" ] || for s in $FILES; do echo "${s%%:*} $(shasum -a 256 "${s#*:}" | cut -c1-64) $(stat -L -f %z "${s#*:}")"; done > "$R/sha.txt"
grep -q "^c1 9975a56dbf2e2bcd91b7c56ff98ef59f0840cdc51d29a8980c0f67b4707531b0 " "$R/sha.txt" || { echo "c1 is not the frozen file"; exit 1; }
grep -q "^m0rk a62fd9cc49e9" "$R/sha.txt" || { echo "m0rk is not the frozen file"; exit 1; }
for s in $FILES; do score "final-${s%%:*}-T3-P3" "${s#*:}" T3; done
date '+%H:%M:%S' > "$R/done"; step "T3 scored, the sibling may delete ref-T3-P3.kld"
$PY "$ROOT/analysis/kltable.py" > /dev/null
$PY - <<PYEOF | tee "$R/compare.txt"
import sys; sys.path.insert(0, "$ROOT/analysis")
from compare import load, paired
k = load(); r = lambda n: k[f"final-{n}-T3-P3"]
def test(cand, bases, label):
    ok = True
    for b in bases:
        c, o = r(cand), r(b); d, lo, hi = paired(c["chunks"], o["chunks"]); win = hi < 0 and c["top"] >= o["top"]; ok &= win
        print(f"T3 {cand} vs {b}: KL {c['chunks'].mean():.5f} vs {o['chunks'].mean():.5f}, diff {d:+.5f} [{lo:+.5f}, {hi:+.5f}], top-1 {c['top']:.2f}% vs {o['top']:.2f}% -> {'better' if win else 'not better'}")
    print(f"{label}: {'PASS' if ok else 'FAIL'}")
test("c1", ["GSQ-RCO-IQ2_XS", "AD-IQ2_M", "AD-IQ2_XXS-Q2", "AD-IQ1_M", "AD-IQ1_S"], "c1 Mac track (9 GB tier)")
test("m0rk", ["AD-IQ2_XXS-Q2", "AD-IQ1_M", "AD-IQ1_S"], "M0rk Mac track (8.49 GB tier)")
for cand in ("c1", "m0rk"):
    c, o = r(cand), r("UD-IQ2_XXS"); d, lo, hi = paired(c["chunks"], o["chunks"])
    print(f"gap, descriptive: T3 {cand} vs UD-IQ2_XXS: KL {c['chunks'].mean():.5f} vs {o['chunks'].mean():.5f}, diff {d:+.5f} [{lo:+.5f}, {hi:+.5f}], ratio {c['chunks'].mean() / o['chunks'].mean():.3f}, top-1 {c['top']:.2f}% vs {o['top']:.2f}%")
PYEOF
step "T3 done"
