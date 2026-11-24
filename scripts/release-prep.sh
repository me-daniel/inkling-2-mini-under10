#!/bin/sh
# Release preparation (decisions reviews 24 Nov, reviews/2026-11-24-decisions-*.md). Nothing is uploaded.
# 1. Release copies as APFS clones (cp -c: no extra disk until a block changes) with general.file_type set to
#    MOSTLY_IQ2_XXS (19, as UD-IQ2_XXS at the same per-layer map) instead of the inherited Q8_0 (7); every other byte
#    is checked equal to the frozen file; sha256 recorded.
# 2. GPU: S scored on each release copy, must match the frozen file's S chunk for chunk; one verbose run per file
#    (c1, M0rk, UD-IQ2_XXS) logs graph splits as runtime evidence of where the ops run.
# 3. Both locks: c1, M0rk and UD-IQ2_XXS benched back to back in one session (the earlier UD speed is cross-session).
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
R="$ROOT/results/release"; O="$M/release"; mkdir -p "$R" "$O"
for n in c1 m0rk; do
    src="$M/tmp/$n.gguf"; dst="$O/$n.gguf"
    if [ ! -f "$dst" ]; then
        cp -c "$src" "$dst.part"
        $PY - "$dst.part" <<'PYEOF'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1], "r+"); f = r.fields["general.file_type"]; assert int(f.parts[f.data[0]][0]) == 7
f.parts[f.data[0]][0] = 19; r.data.flush() if hasattr(r.data, "flush") else None; del r
r = GGUFReader(sys.argv[1]); f = r.fields["general.file_type"]; assert int(f.parts[f.data[0]][0]) == 19
PYEOF
        n_diff=$(cmp -l "$src" "$dst.part" | wc -l | tr -d ' ')
        [ "$n_diff" = 1 ] || { echo "$n: $n_diff bytes differ, expected 1"; rm -f "$dst.part"; exit 1; }
        mv "$dst.part" "$dst"
    fi
    grep -q "^$n " "$R/sha256.txt" 2>/dev/null || echo "$n $(shasum -a 256 "$dst" | cut -c1-64) $(stat -f %z "$dst") file_type 19, 1 byte differs from the frozen file" >> "$R/sha256.txt"
done
cat "$R/sha256.txt"
for n in c1 m0rk; do score "release-$n-S" "$O/$n.gguf" S; done
$PY - <<PYEOF | tee "$R/identical.txt"
import sys; sys.path.insert(0, "$ROOT/analysis")
import kltable; kltable.collect()
from compare import load
k = load()
for n in ("c1", "m0rk"):
    a, b = k[f"release-{n}-S"]["chunks"], k[f"{n}-S"]["chunks"]
    print(f"{n}: release copy S {a.mean():.6f} vs frozen {b.mean():.6f}, chunks identical: {bool((a == b).all())}")
PYEOF
gpu_wait; GPU_HOLD=1
for spec in "c1:$O/c1.gguf" "m0rk:$O/m0rk.gguf" "UD-IQ2_XXS:$M/ext/UD-IQ2_XXS.gguf"; do
    n=${spec%%:*}; f=${spec#*:}; [ -f "$R/splits-$n.txt" ] && continue
    step "verbose run $n"
    "$BIN/llama-perplexity" -m "$f" -f "$ROOT/data/S.txt" -c 512 --chunks 1 -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" -v > "$L/release-splits-$n.log" 2>&1
    grep -E "graph splits|splits = |model buffer size|CPU_Mapped|MTL0_Mapped" "$L/release-splits-$n.log" | sort -u > "$R/splits-$n.txt"; cat "$R/splits-$n.txt"
done
GPU_HOLD=; gpu_done
until both_try; do sleep 30; done
mem_wait; step "both locks held; memory: $(memory_pressure | grep -o 'free percentage: [0-9]*%')"
for spec in "c1:$O/c1.gguf" "m0rk:$O/m0rk.gguf" "UD-IQ2_XXS:$M/ext/UD-IQ2_XXS.gguf"; do
    n=${spec%%:*}; f=${spec#*:}; [ -f "$R/bench-$n.txt" ] && continue
    step "bench $n"
    "$BIN/llama-bench" -m "$f" -ngl 99 -p 512 -n 128 -r 3 -o md > "$R/bench-$n.txt.part" 2> "$L/release-bench-$n.log"
    mv "$R/bench-$n.txt.part" "$R/bench-$n.txt"; cat "$R/bench-$n.txt"
done
cpu_give; gpu_done
step "release prep done"
