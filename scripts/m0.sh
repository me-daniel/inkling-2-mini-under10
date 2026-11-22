#!/bin/sh
# M0 and its reference point on S: UD-IQ2_XXS as downloaded, and M0 (configs/m0.json) built from our donors.
set -eu
. "$(dirname "$0")/lib.sh"
score ext-UD-IQ2_XXS-S "$M/ext/UD-IQ2_XXS.gguf" S
mkdir -p "$M/tmp"; [ -f "$M/tmp/m0.gguf" ] || $PY "$ROOT/scripts/assemble.py" "$ROOT/configs/m0.json" "$M/tmp/m0.gguf"
score m0-S "$M/tmp/m0.gguf" S
$PY "$ROOT/analysis/kltable.py"
