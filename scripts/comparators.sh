#!/bin/sh
# Score the comparators on S (for orientation during selection; S never decides between them and us) and, once
# a candidate is frozen (CAND=path), the release test sets T1 and T2 for every file. Tensor types go to
# results/types-<name>.txt. Waits for each download. Resumes.
set -eu
. "$(dirname "$0")/lib.sh"
E="$M/ext"; R="$ROOT/results"; mkdir -p "$R"
for name in UD-IQ2_XXS AD-IQ2_M AD-IQ2_XXS-Q2 UD-IQ1_M; do
    while [ ! -e "$E/$name.gguf" ]; do sleep 60; done
    [ -f "$R/types-$name.txt" ] || $PY "$ROOT/scripts/gguf-types.py" "$E/$name.gguf" > "$R/types-$name.txt"
    [ "${SCORE_S:-0}" = 1 ] && score "ext-$name-S" "$E/$name.gguf" S
    if [ -n "${CAND:-}" ]; then
        score "final-$name-T1-P3" "$E/$name.gguf" T1
        score "final-$name-T2-P3" "$E/$name.gguf" T2
    fi
done
if [ -n "${CAND:-}" ]; then
    score "final-cand-T1-P3" "$CAND" T1
    score "final-cand-T2-P3" "$CAND" T2
    $PY "$ROOT/analysis/kltable.py" > /dev/null
    $PY "$ROOT/analysis/compare.py" cand UD-IQ2_XXS UD-IQ1_M AD-IQ2_XXS-Q2 | tee "$R/release-test.txt"
    $PY "$ROOT/analysis/compare.py" cand AD-IQ2_M | grep -v 'quality gate' > "$R/context-AD-IQ2_M.txt"
fi
$PY "$ROOT/analysis/kltable.py" > /dev/null
step "comparators done"
