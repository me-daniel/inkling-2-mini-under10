#!/bin/sh
# helper for tests/locks.sh: take FILE with our lib.sh (or the sibling's gpu-lock.sh with "sib"), hold SECONDS, give
set -eu
if [ "${3:-}" = sib ]; then . "$(dirname "$0")/../../inkling-2-mini-quant/scripts/gpu-lock.sh"; trap '' EXIT
else . "$(dirname "$0")/../scripts/lib.sh"; fi
_take "$1"; sleep "$2"; _give "$1"
