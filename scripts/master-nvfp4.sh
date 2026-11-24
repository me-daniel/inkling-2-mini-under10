#!/bin/sh
# Does upstream's new NVFP4 Metal support (ggml-org/llama.cpp #32825, merged 23 Nov 19:02 UTC as 51831d47d, in no
# release tag yet; b12670 is 2 commits before it) remove the speed case for our all-Metal files?
# 1. CPU lock: build vendor/llama.cpp-master (unpatched master, no local patches) for llama-perplexity and llama-bench.
# 2. GPU lock: graph splits of UD-IQ2_XXS on master (2 = all on Metal, as c1 and M0rk on our build; 32 on our build),
#    and UD-IQ2_XXS on S with master: KL must stay at backend-noise level of ext-UD-IQ2_XXS-S (CPU NVFP4 path).
# 3. Both locks: c1 and UD-IQ2_XXS benched back to back on master (-r 3).
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
SRC="$ROOT/vendor/llama.cpp-master"; R="$ROOT/results/master"; mkdir -p "$R"
git -C "$SRC" merge-base --is-ancestor 51831d47d HEAD || { echo "master clone predates 51831d47d"; exit 1; }
if [ ! -x "$SRC/build/bin/llama-bench" ]; then
    cpu_take; step "build master $(git -C "$SRC" rev-parse --short HEAD)"
    cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF > "$L/master-build.log" 2>&1
    cmake --build "$SRC/build" --target llama-perplexity llama-bench -j "$(sysctl -n hw.ncpu)" >> "$L/master-build.log" 2>&1
    cpu_give
fi
BIN="$SRC/build/bin"; UD="$M/ext/UD-IQ2_XXS.gguf"; C1="$M/release/c1.gguf"
echo "master $(git -C "$SRC" rev-parse --short HEAD) $(git -C "$SRC" log -1 --format=%cs)" > "$R/commit.txt"
if [ ! -f "$R/splits-UD-IQ2_XXS.txt" ]; then
    gpu_wait; step "splits UD-IQ2_XXS on master"
    "$BIN/llama-perplexity" -m "$UD" -f "$ROOT/data/S.txt" -c 512 --chunks 1 -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" -v > "$L/master-splits-UD.log" 2>&1
    grep -E "splits = |CPU_Mapped|MTL0_Mapped" "$L/master-splits-UD.log" | sort -u > "$R/splits-UD-IQ2_XXS.txt"; gpu_done
fi
cat "$R/splits-UD-IQ2_XXS.txt"
score master-UD-IQ2_XXS-S "$UD" S
$PY "$ROOT/analysis/kltable.py" > /dev/null
$PY - <<PYEOF | tee "$R/kl-UD.txt"
import sys; sys.path.insert(0, "$ROOT/analysis")
from compare import load, paired
k = load(); a, b = k["master-UD-IQ2_XXS-S"], k["ext-UD-IQ2_XXS-S"]
d, lo, hi = paired(a["chunks"], b["chunks"])
print(f"S UD-IQ2_XXS on master (NVFP4 on Metal) vs our build (NVFP4 on CPU): {a['chunks'].mean():.5f} vs {b['chunks'].mean():.5f}, diff {d:+.5f} [{lo:+.5f}, {hi:+.5f}], top-1 {a['top']:.2f}% vs {b['top']:.2f}%")
PYEOF
until both_try; do sleep 30; done
mem_wait; step "both locks held; memory: $(memory_pressure | grep -o 'free percentage: [0-9]*%')"
for spec in "c1:$C1" "UD-IQ2_XXS:$UD"; do
    n=${spec%%:*}; f=${spec#*:}; [ -f "$R/bench-$n.txt" ] && continue
    step "bench $n on master"
    "$BIN/llama-bench" -m "$f" -ngl 99 -p 512 -n 128 -r 3 -o md > "$R/bench-$n.txt.part" 2> "$L/master-bench-$n.log"
    mv "$R/bench-$n.txt.part" "$R/bench-$n.txt"; cat "$R/bench-$n.txt"
done
step "master check done"
