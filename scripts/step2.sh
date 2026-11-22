#!/bin/sh
# concept-v3 step 2: measure what exists. Window granted by the sibling (22 Nov ~18:50).
# Lock order as the sibling's bench: GPU first, then CPU. Both are held for the speed and memory runs (llama-bench is
# clean only while no refiner shares memory bandwidth). The CPU lock is released before the KL scores, which
# need the GPU alone. Same binary and settings as the sibling's gate and depth benches. Resumes.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
E="$M/ext"; R="$ROOT/results/step2"; mkdir -p "$R" "$L"
# check each download against its Hugging Face LFS oid (written by the download job)
while [ ! -f "$E/GSQ-RCO-IQ2_XS.gguf" ] || [ ! -f "$E/mainline/UD-Q2_K_XL.gguf" ]; do sleep 60; done
for spec in "GSQ-RCO-IQ2_XS:$E/GSQ-RCO-IQ2_XS.gguf:${GSQ_OID:?}" "UD-Q2_K_XL:$E/mainline/UD-Q2_K_XL.gguf:${UDQ2K_OID:?}"; do
    name=${spec%%:*}; r=${spec#*:}; f=${r%%:*}; oid=${r#*:}
    [ -f "$R/sha-$name.txt" ] || { h=$(shasum -a 256 "$f" | cut -c1-64); [ "$h" = "$oid" ] || { echo "$name sha mismatch"; exit 1; }; echo "$name $h $(stat -f %z "$f") MATCH" > "$R/sha-$name.txt"; }
    [ -f "$ROOT/results/types-$name.txt" ] || $PY "$ROOT/scripts/gguf-types.py" "$f" > "$ROOT/results/types-$name.txt"
    [ "$f" = "$E/$name.gguf" ] || [ -e "$E/$name.gguf" ] || ln -s "$f" "$E/$name.gguf"  # never over the file itself (21:17 bug)
done
gpu_wait; GPU_HOLD=1; cpu_take
step "locks held; memory: $(memory_pressure | grep -o 'free percentage: [0-9]*%')"
FILES="m0rk:$M/tmp/m0rk.gguf UD-IQ2_XXS:$E/UD-IQ2_XXS.gguf UD-Q2_K_XL:$E/UD-Q2_K_XL.gguf GSQ-RCO-IQ2_XS:$E/GSQ-RCO-IQ2_XS.gguf"
for spec in $FILES; do
    name=${spec%%:*}; f=${spec#*:}
    [ -f "$R/bench-$name.txt" ] && continue
    step "bench $name"
    "$BIN/llama-bench" -m "$f" -ngl 99 -p 512 -n 128 -r 3 -o md > "$R/bench-$name.txt.part" 2> "$L/step2-bench-$name.log"
    mv "$R/bench-$name.txt.part" "$R/bench-$name.txt"; cat "$R/bench-$name.txt"
done
# peak memory: one llama-bench run at depth D (n_ctx = D + 512 + 128); /usr/bin/time -l gives max RSS and peak
# footprint, and the log keeps llama.cpp's Metal buffer sizes. M0rk at 8K, 16K and 32K; the others at 16K.
for spec in m0rk:8192 m0rk:16384 m0rk:32768 UD-Q2_K_XL:16384 GSQ-RCO-IQ2_XS:16384 UD-IQ2_XXS:16384; do
    name=${spec%%:*}; d=${spec#*:}; f=$(for s in $FILES; do [ "${s%%:*}" = "$name" ] && echo "${s#*:}"; done)
    [ -f "$R/memory-$name-$d.txt" ] && continue
    step "peak memory $name at depth $d"
    /usr/bin/time -l "$BIN/llama-bench" -m "$f" -ngl 99 -p 512 -n 128 -d "$d" -r 1 -o md > "$R/depth-$name-$d.txt" 2> "$L/step2-memory-$name-$d.log"
    { grep -E "maximum resident set size|peak memory footprint" "$L/step2-memory-$name-$d.log"; grep -iE "MTL0_Mapped|Metal.*buffer size|KV.*MiB|compute buffer" "$L/step2-memory-$name-$d.log" | head -12; } > "$R/memory-$name-$d.txt"
    cat "$R/memory-$name-$d.txt"
done
cpu_give; step "CPU lock released; scores with the GPU lock alone"
for name in UD-Q2_K_XL GSQ-RCO-IQ2_XS; do
    score "ext-$name-S" "$E/$name.gguf" S
    score "final-$name-T1-P3" "$E/$name.gguf" T1
    score "final-$name-T2-P3" "$E/$name.gguf" T2
done
GPU_HOLD=; gpu_done
$PY "$ROOT/analysis/kltable.py" > /dev/null
$PY "$ROOT/analysis/compare.py" cand UD-Q2_K_XL GSQ-RCO-IQ2_XS | tee "$R/compare.txt"
step "step 2 done"
