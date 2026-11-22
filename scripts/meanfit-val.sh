#!/bin/sh
# Refit layers 6 (IQ1_S) and 20 (IQ2_S) and score llama-quantize, refit and UD-IQ2_XXS on the sibling's full gate_up Grams.
set -eu
. "$(dirname "$0")/lib.sh"
trap cpu_give EXIT
IM="$SIB/models/ggml-org/imatrix.gguf"; GD="$SIB/models/grams/gu-6-20"; UD="$M/ext/mainline/UD-IQ2_XXS.gguf"
cpu_take; $PY "$ROOT/scripts/meanfit.py" "$M/pools/gu-iq1_s.gguf" "$IM" "$M/meanfit/val-L6.gguf" --layers 6-6; cpu_give
cpu_take; $PY "$ROOT/scripts/meanfit.py" "$M/pools/gu-iq2_s.gguf" "$IM" "$M/meanfit/val-L20.gguf" --layers 20-20; cpu_give
cpu_take
$PY "$ROOT/scripts/gu_full_err.py" 6 "$GD" "llama-quantize=$M/pools/gu-iq1_s.gguf" "refit=$M/meanfit/val-L6.gguf" "UD=$UD:scaled"
$PY "$ROOT/scripts/gu_full_err.py" 20 "$GD" "llama-quantize=$M/pools/gu-iq2_s.gguf" "refit=$M/meanfit/val-L20.gguf" "UD=$UD:scaled"
cpu_give
