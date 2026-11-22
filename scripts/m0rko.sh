#!/bin/sh
# Own path (CHANGELOG 11:20): imatrix including the output head, non-expert pool at UD's types with it, M0rko on S.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'cpu_give; gpu_done' EXIT
I="$M/imatrix/mine-out.gguf"
if [ ! -f "$I" ]; then
    cpu_take; gpu_wait; step "imatrix with output head"
    "$BIN/llama-imatrix" -m "$SIB/models/cand.gguf" -f "$ROOT/data/calib-fit.txt" --parse-special --process-output --chunks 160 \
        -o "$I.part" -c 512 -ngl 99 > "$L/imatrix-out.log" 2>&1
    gpu_done; cpu_give; mv "$I.part" "$I"
fi
P="$M/pools/ne-ud2.gguf"
if [ ! -f "$P" ]; then
    set -- --tensor-type "ffn_gate_up_exps=q2_0" --tensor-type "ffn_down_exps=q2_0"
    while IFS= read -r o; do [ -n "$o" ] && set -- "$@" --tensor-type "$o"; done < "$ROOT/configs/ne-ud.tensor-types"
    cpu_take; step "quantize ne-ud2"
    "$BIN/llama-quantize" --imatrix "$I" "$@" "$SIB/models/Inkling-2-Mini-BF16.gguf" "$P.part" Q8_0 > "$L/quant-ne-ud2.log" 2>&1
    cpu_give; mv "$P.part" "$P"
fi
grep -iE "output.weight|token_embd" "$L/quant-ne-ud2.log" | head -4
$PY -c "
import json; m = json.load(open('$ROOT/configs/m0rk.json')); m['ne'] = {'attn': 'ud2', 'embd': 'ud2', 'out': 'ud2'}
json.dump(m, open('$ROOT/configs/m0rko.json', 'w'))"
out="$M/tmp/m0rko.gguf"; [ -f "$out" ] || { cpu_take; $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/m0rko.json" "$out"; cpu_give; }
score m0rko-S "$out" S
$PY "$ROOT/analysis/kltable.py" > /dev/null; step "m0rko done"
