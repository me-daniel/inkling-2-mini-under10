#!/bin/sh
# Step 3d (CHANGELOG 23 Nov 06:15): M0rk's recipe with refit statistics from a mixed-domain imatrix. Resumes.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
IM="$M/imatrix/mix-cand-300.gguf"
if [ ! -f "$IM" ]; then
    gpu_wait; step "imatrix on calib-mix (300 x 512, --parse-special)"
    "$BIN/llama-imatrix" -m "$SIB/models/cand.gguf" -f "$SIB/data/calib-mix.txt" --chunks 300 -c 512 -ngl 99 --parse-special \
        --override-kv "inkling2_loop.loop.count=int:3" -o "$IM.part" > "$L/imatrix-mix.log" 2>&1
    mv "$IM.part" "$IM"; gpu_done
fi
for spec in "iq1_s:0,1,2,3,4,6:$M/pools/gu-iq1_s.gguf" "iq2_xxs:5,7,8,12,13,14,18:$SIB/models/pool-2bit.gguf" \
            "iq2_xs:9,10,11,15,17,22:$M/pools/gu-iq2_xs.gguf" "iq2_s:16,19,20,21,23:$M/pools/gu-iq2_s.gguf"; do
    ty=${spec%%:*}; rest=${spec#*:}; ls=${rest%%:*}; pool=${rest#*:}; out="$M/meanfit/rfm-$ty.gguf"
    [ -f "$out" ] || { cpu_take; step "refit $ty layers $ls (mix statistics)"; $PY "$ROOT/scripts/meanfit.py" "$pool" "$IM" "$out" --layers "$ls" | tee -a "$L/meanfit-mix.log"; cpu_give; }
    for l in $(echo "$ls" | tr ',' ' '); do
        f="$M/down/rfm-L$l.gguf"
        [ -f "$f" ] || { cpu_take; $PY "$ROOT/scripts/degain2.py" "$out" "$f" "$l" --kind ffn_gate_up_exps --target-from "$pool" >> "$L/degain2-rfm.log"; cpu_give; }
    done
done
$PY -c "
import json; m = json.load(open('$ROOT/configs/m0rk.json')); m['gate_up'] = {str(l): 'rfm' for l in range(24)}
json.dump(m, open('$ROOT/configs/m0rk-mix.json', 'w'))"
out="$M/tmp/m0rk-mix.gguf"
[ -f "$out" ] || { cpu_take; step "assemble m0rk-mix"; $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/m0rk-mix.json" "$out"; cpu_give; }
score m0rk-mix-S "$out" S
score final-m0rk-mix-T1-P3 "$out" T1
score final-m0rk-mix-T2-P3 "$out" T2
$PY "$ROOT/analysis/kltable.py" > /dev/null
$PY - <<PYEOF | tee "$ROOT/results/step3/m0rk-mix.txt"
import sys; sys.path.insert(0, "$ROOT/analysis")
from compare import load, paired
k = load()
for s, a, b in (("S", "m0rk-mix-S", "m0rk-S"), ("T1", "final-m0rk-mix-T1-P3", "final-cand-T1-P3"), ("T2", "final-m0rk-mix-T2-P3", "final-cand-T2-P3")):
    d, lo, hi = paired(k[a]["chunks"], k[b]["chunks"])
    print(f"{s} m0rk-mix vs m0rk: {k[a]['chunks'].mean():.5f} vs {k[b]['chunks'].mean():.5f}, diff {d:+.5f} [{lo:+.5f}, {hi:+.5f}], top-1 {k[a]['top']:.2f}% vs {k[b]['top']:.2f}%")
PYEOF
step "m0rk-mix done"
