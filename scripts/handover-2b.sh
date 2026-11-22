#!/bin/sh
# One-off: once the WikiText sweep is written, stop gptq2.sh before its own encoding and run gptq2b.sh instead.
cd "$(dirname "$0")/.." || exit 1
until [ -f results/alpha-wt.tsv ]; do sleep 15; done
P=$(pgrep -f "scripts/gptq2[.]sh"); [ -n "$P" ] && kill $P; sleep 2
Q=$(pgrep -f "scripts/gptq_q2[.]py"); [ -n "$Q" ] && kill $Q; sleep 2
rm -f models/down/g2v-*.part
exec sh scripts/gptq2b.sh
