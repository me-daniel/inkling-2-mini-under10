#!/bin/sh
# concept-v3: the smaller Mac-track comparators (S, T1, T2) and step 3's single promotions on M0rk (S), as
# pre-registered in CHANGELOG 22 Nov 19:40. Starts after step2.sh. Builds take the CPU lock one at a time; scores take the GPU lock.
# Each promotion file is deleted after its S score (its json, splice log and score log stay). Resumes.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
E="$M/ext"; R="$ROOT/results/step3"; mkdir -p "$R"
CAP=9138715328
while pgrep -f "scripts/step2.sh" > /dev/null; do sleep 60; done
# 1. comparators, each checked against its Hugging Face LFS oid
for spec in "AD-IQ1_M:$E/AD-IQ1_M.gguf:53261931d91fa1547efa4288a7c596b8d4987db94252f08f316357cc67ea355f" \
            "AD-IQ1_S:$E/AD-IQ1_S.gguf:4d3cab7e88016548136e13f7ec465db7fc49ab8cf9559a60f3daad1f3f15c9d5" \
            "UD-IQ1_S:$E/mainline/UD-IQ1_S.gguf:17a7ff34acb0e1fe878309962458e0633172c7f93ffe3b1c4f5f6b317dc317db"; do
    name=${spec%%:*}; r=${spec#*:}; f=${r%%:*}; oid=${r#*:}
    while [ ! -f "$f" ]; do sleep 60; done
    [ -f "$R/sha-$name.txt" ] || { h=$(shasum -a 256 "$f" | cut -c1-64); [ "$h" = "$oid" ] || { echo "$name sha mismatch"; exit 1; }; echo "$name $h MATCH" > "$R/sha-$name.txt"; }
    [ -f "$ROOT/results/types-$name.txt" ] || $PY "$ROOT/scripts/gguf-types.py" "$f" > "$ROOT/results/types-$name.txt"
    [ -e "$E/$name.gguf" ] || ln -s "$f" "$E/$name.gguf"
    score "ext-$name-S" "$E/$name.gguf" S
    score "final-$name-T1-P3" "$E/$name.gguf" T1
    score "final-$name-T2-P3" "$E/$name.gguf" T2
done
$PY "$ROOT/analysis/kltable.py" > /dev/null
$PY "$ROOT/analysis/compare.py" cand AD-IQ1_M AD-IQ1_S UD-IQ1_S | tee "$R/compare-iq1.txt"
# 2. promoted gate_up donors with M0rk's recipe: mean-aware refit of the llama-quantize pool, then each row back to
#    that pool's own gain. IQ3_XXS has no refit (meanfit.py), so p-g2023 takes the plain pool.
IM="$SIB/models/ggml-org/imatrix.gguf"
for spec in "4:$SIB/models/pool-2bit.gguf" "6:$SIB/models/pool-2bit.gguf" "13:$M/pools/gu-iq2_s.gguf" "18:$M/pools/gu-iq2_s.gguf"; do
    l=${spec%%:*}; pool=${spec#*:}; f="$M/down/rfx-L$l.gguf"
    [ -f "$f" ] && continue
    cpu_take; step "refit gate_up $l from $(basename "$pool")"
    $PY "$ROOT/scripts/meanfit.py" "$pool" "$IM" "$M/meanfit/rfx-L$l.gguf" --layers "$l" >> "$L/step3-meanfit.log"
    $PY "$ROOT/scripts/degain2.py" "$M/meanfit/rfx-L$l.gguf" "$f" "$l" --kind ffn_gate_up_exps --target-from "$pool" >> "$L/step3-degain.log"
    cpu_give
done
tail -n 4 "$L/step3-meanfit.log" "$L/step3-degain.log"
# 3. single promotions, S only
for spec in "p-d13:down:13=iq4_nl" "p-d21:down:21=iq4_nl" "p-g46:gate_up:4=rfx,6=rfx" "p-g1318:gate_up:13=rfx,18=rfx" "p-g2023:gate_up:20=iq3_xxs,23=iq3_xxs"; do
    name=${spec%%:*}; r=${spec#*:}; kind=${r%%:*}; sets=${r#*:}; out="$M/tmp/$name.gguf"
    grep -q 'Mean *KLD' "$L/ppl-$name-S.log" 2>/dev/null && continue
    $PY -c "
import json, sys
m = json.load(open('$ROOT/configs/m0rk.json'))
for kv in '$sets'.split(','):
    l, src = kv.split('='); m['$kind'][l] = src
json.dump(m, open('$ROOT/configs/$name.json', 'w'))"
    if [ ! -f "$out" ]; then
        cpu_take; step "assemble $name"; $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/$name.json" "$out" --over; cpu_give
    fi
    sz=$(stat -f %z "$out"); echo "$name $sz bytes" | tee -a "$R/sizes.txt"
    [ "$sz" -le "$CAP" ] || { echo "$name over the cap"; exit 1; }
    score "$name-S" "$out" S
    rm -f "$out"
done
$PY "$ROOT/analysis/kltable.py" > /dev/null
$PY - <<EOF | tee "$R/promotions.txt"
import sys; sys.path.insert(0, "$ROOT/analysis")
from compare import load, paired
k = load(); base = k["m0rk-S"]["chunks"]; M0 = 8486214944
sizes = dict(l.split()[:2] for l in open("$R/sizes.txt"))
print("promotion\tKL\tdiff vs m0rk [95% CI]\tbytes added\tKL change per 100 MB")
for n in ("p-d13", "p-d21", "p-g46", "p-g1318", "p-g2023"):
    c = k[n + "-S"]["chunks"]; d, lo, hi = paired(c, base); add = int(sizes[n]) - M0
    print(f"{n}\t{c.mean():.5f}\t{d:+.5f} [{lo:+.5f}, {hi:+.5f}]\t{add:,}\t{d / add * 1e8:+.5f}")
EOF
step "step 3 done"
