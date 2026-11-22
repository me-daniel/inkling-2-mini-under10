#!/bin/sh
# G5 (projection; no 16 GB Mac here): speed at depth 0 and 16K, and peak memory at 16K, for the frozen candidate
# (CAND=path) and UD-IQ2_XXS. Same binary, under the GPU lock. Metal buffer sizes from llama.cpp's log are summed
# and compared with 11,453 MB (recommendedMaxWorkingSetSize of a 16 GB Mac). -> results/fit16-<name>.txt
set -eu
. "$(dirname "$0")/lib.sh"
trap 'gpu_done; cpu_give' EXIT
R="$ROOT/results"
for spec in "cand:$CAND" "UD-IQ2_XXS:$M/ext/UD-IQ2_XXS.gguf"; do
    name=${spec%%:*}; f=${spec#*:}; out="$R/fit16-$name.txt"; [ -f "$out" ] && continue
    cpu_take; gpu_wait; step "bench $name"  # both locks: timings need an otherwise idle machine
    "$BIN/llama-bench" -m "$f" -ngl 99 -p 512 -n 128 -d 0,16384 -r 2 -o md > "$out.part" 2> "$L/bench-$name.log"
    /usr/bin/time -l "$BIN/llama-bench" -m "$f" -ngl 99 -p 512 -n 128 -d 16384 -r 1 -o md > "$L/memory-$name.out" 2> "$L/memory-$name.log"
    gpu_done; cpu_give
    {
        grep -E "maximum resident set size|peak memory footprint" "$L/memory-$name.log"
        grep -E "buffer size" "$L/memory-$name.log" | sed 's/^[^a-zA-Z]*//' | sort -u
        grep -E "buffer size" "$L/memory-$name.log" | awk '{for (i = 1; i < NF; i++) if ($(i+1) == "MiB") s += $i} END {printf "sum of logged buffers: %.0f MiB = %.0f MB (16 GB Mac default limit 11,453 MB)\n", s, s * 1.048576}'
    } >> "$out.part"
    mv "$out.part" "$out"; cat "$out"
done
step "fit16 done"
