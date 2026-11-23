#!/bin/sh
# Kernel-lock helpers in scripts/lib.sh on scratch lock files: exclusion, try, re-entry, give, crash release, both_try,
# and exclusion against a holder taken with the sibling's own gpu-lock.sh. Run: sh tests/locks.sh
set -eu
. "$(dirname "$0")/../scripts/lib.sh"
D=$(mktemp -d); LOCK="$D/gpu"; CLOCK="$D/cpu"
ok() { echo "ok   $*"; }; bad() { echo "FAIL $*"; exit 1; }
holder() { sh "$ROOT/tests/lockhold.sh" "$@" & _hp=$!; sleep 2; }  # file seconds [sib]: another process holds FILE
holder "$LOCK" 6; _take "$LOCK" -t 0 && bad "try while held" || ok "try fails while another holds"
t0=$(date +%s); _take "$LOCK"; t1=$(date +%s); [ $((t1 - t0)) -ge 2 ] && ok "blocking take waited $((t1 - t0)) s" || bad "did not wait"
_take "$LOCK" && ok "re-entrant take"
both_try && ok "both_try with the GPU already ours" && cpu_give || bad "both_try with the GPU ours and the CPU free"
gpu_done; _take "$LOCK" -t 0 && ok "retake after give" && gpu_done
wait $_hp
sh "$ROOT/tests/lockhold.sh" "$LOCK" 60 & k=$!; sleep 2; kill -9 $k; sleep 4
_take "$LOCK" -t 0 && ok "kill -9 of a holder frees the lock" && gpu_done || bad "lock stuck after kill -9"
holder "$CLOCK" 4; both_try && bad "both_try with CPU busy" || ok "both_try fails while CPU busy"
_take "$LOCK" -t 0 && ok "both_try gave the GPU back" && gpu_done || bad "both_try kept the GPU"
wait $_hp; both_try && ok "both_try holds both when free" && { cpu_give; gpu_done; }
holder "$LOCK" 5 sib; _take "$LOCK" -t 0 && bad "sibling holder ignored" || ok "excluded by a sibling-code holder"
wait $_hp; _take "$LOCK" -t 0 && ok "free after the sibling holder gave" && gpu_done
sleep 3; n=$(pgrep -f "lockf -k .*$D" | wc -l | tr -d ' '); [ "$n" = 0 ] && ok "no holders left" || bad "$n holders left"
rm -rf "$D"
