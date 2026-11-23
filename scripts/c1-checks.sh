#!/bin/sh
# v1-frame checks on the frozen c1 (CHANGELOG 23 Nov 08:20). Resumes.
# 1. peak memory at 16K and 32K (GPU lock), 2. S on unpatched upstream c661033 (GPU lock), 3. stability battery
# (GPU lock), 4. speed with c1 and M0rk back to back (both locks, taken only when both are free).
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
F="$M/tmp/c1.gguf"; R="$ROOT/results/c1"; mkdir -p "$R"
[ "$(shasum -a 256 "$F" | cut -c1-64)" = "$(cut -c1-64 "$ROOT/results/step3/c1.sha256")" ] || { echo "c1 changed since freezing"; exit 1; }
gpu_wait; GPU_HOLD=1
for d in 16384 32768; do
    [ -f "$R/memory-$d.txt" ] && continue
    step "peak memory c1 at depth $d"
    /usr/bin/time -l "$BIN/llama-bench" -m "$F" -ngl 99 -p 512 -n 128 -d "$d" -r 1 -o md > "$R/depth-$d.txt" 2> "$L/c1-memory-$d.log"
    grep -E "maximum resident set size|peak memory footprint" "$L/c1-memory-$d.log" | tee "$R/memory-$d.txt"
done
UBIN="$SIB/vendor/llama.cpp-upstream/build/bin"; OBIN=$BIN; BIN=$UBIN
score c1-upstream-S "$F" S
BIN=$OBIN
[ -f "$ROOT/results/stability/stability-c1.json" ] || { step "stability c1"; $PY "$ROOT/scripts/stability.py" c1 "$F"; }
GPU_HOLD=; gpu_done
$PY - <<PYEOF | tee "$R/stability-verdict.md"
import json, sys; sys.path.insert(0, "$ROOT/scripts")
from stability import judge
v = judge(json.load(open("$SIB/results/pilot/stability-ref.json")), json.load(open("$ROOT/results/stability/stability-c1.json")))
print("| check | Q8_0 reference | c1 | rule | result |\n|---|---|---|---|---|")
for c in v["checks"]: print(f"| {c['name']} | {c['ref']} | {c['cand']} | {c['rule']} | {'pass' if c['pass'] else 'FAIL'} |")
print(f"\nverdict: {'PASS' if v['pass'] else 'FAIL'}")
PYEOF
$PY "$ROOT/analysis/kltable.py" > /dev/null
$PY - <<PYEOF | tee "$R/upstream.txt"
import sys; sys.path.insert(0, "$ROOT/analysis")
from compare import load, paired
k = load(); a, b = k["c1-upstream-S"], k["c1-S"]; d, lo, hi = paired(a["chunks"], b["chunks"])
print(f"S c1 on upstream c661033 vs our build: {a['chunks'].mean():.5f} vs {b['chunks'].mean():.5f}, diff {d:+.5f} [{lo:+.5f}, {hi:+.5f}]")
PYEOF
until both_try; do sleep 30; done
mem_wait; step "both locks held for the speed bench"
for spec in "c1:$F" "m0rk:$M/tmp/m0rk.gguf"; do
    name=${spec%%:*}; f=${spec#*:}
    [ -f "$R/bench-$name.txt" ] && continue
    "$BIN/llama-bench" -m "$f" -ngl 99 -p 512 -n 128 -r 3 -o md > "$R/bench-$name.txt" 2> "$L/c1-bench-$name.log"; cat "$R/bench-$name.txt"
done
step "c1 checks done"
