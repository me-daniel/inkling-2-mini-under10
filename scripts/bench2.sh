#!/bin/sh
# Speed of the four files back to back on a quiet machine (concept-v3 step 2, phase B). Takes the GPU lock and then
# the CPU lock only when both are free at once, and gives the GPU back at once if the CPU is taken in between, so it
# never holds an idle GPU (23 Nov 02:20: step2.sh held it for hours waiting for the CPU). Resumes.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
E="$M/ext"; R="$ROOT/results/step2"
until both_try; do sleep 30; done
mem_wait; step "both locks held; memory: $(memory_pressure | grep -o 'free percentage: [0-9]*%')"
for spec in "m0rk:$M/tmp/m0rk.gguf" "UD-IQ2_XXS:$E/UD-IQ2_XXS.gguf" "UD-Q2_K_XL:$E/UD-Q2_K_XL.gguf" "GSQ-RCO-IQ2_XS:$E/GSQ-RCO-IQ2_XS.gguf"; do
    name=${spec%%:*}; f=${spec#*:}
    [ -f "$R/bench2-$name.txt" ] && continue
    step "bench $name"
    "$BIN/llama-bench" -m "$f" -ngl 99 -p 512 -n 128 -r 3 -o md > "$R/bench2-$name.txt.part" 2> "$L/step2-bench2-$name.log"
    mv "$R/bench2-$name.txt.part" "$R/bench2-$name.txt"; cat "$R/bench2-$name.txt"
done
step "bench2 done"
