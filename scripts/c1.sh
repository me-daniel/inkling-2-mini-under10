#!/bin/sh
# The step-3 combination c1 (CHANGELOG 23 Nov 08:00) on the M0rk-mix base. Keeps the file. Resumes.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
IM="$M/imatrix/mix-cand-300.gguf"; CAP=8976127992; R="$ROOT/results/step3"
for spec in "4:$SIB/models/pool-2bit.gguf" "6:$SIB/models/pool-2bit.gguf" "13:$M/pools/gu-iq2_s.gguf" "18:$M/pools/gu-iq2_s.gguf"; do
    l=${spec%%:*}; pool=${spec#*:}; f="$M/down/rfmx-L$l.gguf"
    [ -f "$f" ] && continue
    cpu_take; step "mix refit gate_up $l from $(basename "$pool")"
    $PY "$ROOT/scripts/meanfit.py" "$pool" "$IM" "$M/meanfit/rfmx-L$l.gguf" --layers "$l" >> "$L/c1-meanfit.log"
    $PY "$ROOT/scripts/degain2.py" "$M/meanfit/rfmx-L$l.gguf" "$f" "$l" --kind ffn_gate_up_exps --target-from "$pool" >> "$L/c1-degain.log"
    cpu_give
done
$PY -c "
import json; m = json.load(open('$ROOT/configs/m0rk-mix.json'))
for l in (4, 6, 13, 18): m['gate_up'][str(l)] = 'rfmx'
for l in (20, 23): m['gate_up'][str(l)] = 'iq3_xxs'
for l in (13, 21): m['down'][str(l)] = 'iq4_nl'
json.dump(m, open('$ROOT/configs/c1.json', 'w'))"
out="$M/tmp/c1.gguf"
[ -f "$out" ] || { cpu_take; step "assemble c1"; $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/c1.json" "$out" --over; cpu_give; }
sz=$(stat -f %z "$out"); echo "c1 $sz bytes" | tee "$R/c1-size.txt"
[ "$sz" -le "$CAP" ] || { echo "c1 over the cap"; exit 1; }
shasum -a 256 "$out" | tee "$R/c1.sha256"
score c1-S "$out" S
score final-c1-T1-P3 "$out" T1
score final-c1-T2-P3 "$out" T2
$PY "$ROOT/analysis/kltable.py" > /dev/null
$PY - <<PYEOF | tee "$R/c1.txt"
import sys; sys.path.insert(0, "$ROOT/analysis")
from compare import load, paired
k = load()
for s, a, b in (("S", "c1-S", "m0rk-mix-S"), ("T1", "final-c1-T1-P3", "final-m0rk-mix-T1-P3"), ("T2", "final-c1-T2-P3", "final-m0rk-mix-T2-P3")):
    d, lo, hi = paired(k[a]["chunks"], k[b]["chunks"])
    print(f"{s} c1 vs m0rk-mix: {k[a]['chunks'].mean():.5f} vs {k[b]['chunks'].mean():.5f}, diff {d:+.5f} [{lo:+.5f}, {hi:+.5f}], top-1 {k[a]['top']:.2f}% vs {k[b]['top']:.2f}%")
PYEOF
$PY "$ROOT/analysis/compare.py" c1 GSQ-RCO-IQ2_XS AD-IQ2_M AD-IQ2_XXS-Q2 AD-IQ1_M AD-IQ1_S UD-IQ1_S UD-IQ2_XXS UD-IQ1_M UD-Q2_K_XL | tee "$R/c1-compare.txt"
step "c1 done"
