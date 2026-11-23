#!/bin/sh
# Code-review follow-up (23 Nov): m0rk-S was signed against an earlier write of models/tmp/m0rk.gguf (22 Nov 10:46),
# re-assembled at 12:38 without a log entry. Rescore S on the current file and compare per chunk with the old run.
set -eu
. "$(dirname "$0")/lib.sh"
trap 'gpu_done' EXIT
[ -f "$L/ppl-m0rk-S-1046.log" ] || { cp "$L/ppl-m0rk-S.log" "$L/ppl-m0rk-S-1046.log"; cp "$L/ppl-m0rk-S.log.sig" "$L/ppl-m0rk-S-1046.log.sig"; }
score m0rk-S "$M/tmp/m0rk.gguf" S
$PY "$ROOT/analysis/kltable.py" > /dev/null
$PY - <<PYEOF | tee "$ROOT/results/m0rk-rescore.txt"
import csv
r = {x["run"]: x for x in csv.DictReader(open("$ROOT/results/kl.tsv"), delimiter="\t")}
a, b = r["m0rk-S"]["per_chunk"].split(","), r["m0rk-S-1046"]["per_chunk"].split(",")
print(f"m0rk-S on the current file: {r['m0rk-S']['mean_kl']} ({r['m0rk-S']['model_state']}); old 10:46 file: {r['m0rk-S-1046']['mean_kl']}; identical chunks {sum(x == y for x, y in zip(a, b))} of {len(a)}")
PYEOF
