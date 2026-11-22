# shared helpers: sourced by the run scripts
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); SIB=$(cd "$ROOT/../inkling-2-mini-quant" && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"; M="$ROOT/models"; L="$ROOT/logs"; PY="$SIB/.venv/bin/python -I"
mkdir -p "$L"
step() { echo "[$(date '+%H:%M:%S')] $*"; }
# Shared GPU lock with the sibling session (agreed 22 Nov): ~/GitHub/.gpu.lock exists while one heavy GPU job runs.
# gpu_wait takes it atomically (noclobber) with "<pid> <repo> <time>" inside; gpu_done releases it. A lock whose pid is
# gone is stale and removed. Never start beside a llama-bench.
LOCK="$HOME/GitHub/.gpu.lock"
gpu_wait() {
    while :; do
        if [ -e "$LOCK" ]; then
            pid=$(awk '{print $1}' "$LOCK" 2>/dev/null || true)
            case "$pid" in ''|*[!0-9]*) ;; *) kill -0 "$pid" 2>/dev/null || { rm -f "$LOCK"; continue; } ;; esac
        elif ! pgrep -x llama-bench > /dev/null && (set -C; echo "$$ under10 $(date '+%H:%M:%S')" > "$LOCK") 2>/dev/null; then
            return 0  # callers own their traps (a trap here replaced theirs and dropped cpu_give)
        fi
        sleep 20
    done
}
gpu_done() { [ -e "$LOCK" ] && [ "$(awk '{print $1}' "$LOCK")" = "$$" ] && rm -f "$LOCK"; return 0; }
# score MODEL on SET (S, T1, T2) against the sibling's Q8_0 reference logits at P = 3 -> logs/ppl-NAME.log
# extra args go to llama-perplexity (e.g. --chunks 8). Skips a run whose log already has a final KL.
score() { # name model set [extra...]
    name=$1 model=$2 set=$3; shift 3
    log="$L/ppl-$name.log"
    # a cached log counts only for the same model file (path, size, mtime) and with every chunk finite
    ref="$SIB/models/logits/ref-$set-P3.kld"
    sig="$(cd "$(dirname "$model")" && pwd -P)/$(basename "$model") $(stat -L -f '%z %m' "$model") $set $* ref $(stat -L -f '%z %m' "$ref") bin $(stat -f '%m' "$BIN/llama-perplexity")"
    if [ -f "$log" ] && grep -q 'Mean *KLD' "$log" && [ "$(cat "$log.sig" 2>/dev/null)" = "$sig" ] && ! grep -Eq '^ *[0-9]+ .*(nan|inf)' "$log"; then return 0; fi
    gpu_wait; step "score $name"
    ctx=4096; [ "$set" = T2 ] && ctx=2048
    "$BIN/llama-perplexity" -m "$model" -f "$ROOT/data/$set.txt" -c $ctx -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
        --kl-divergence-base "$SIB/models/logits/ref-$set-P3.kld" --kl-divergence "$@" > "$log" 2>&1
    gpu_done
    echo "$sig" > "$log.sig"
    grep -E 'Mean +KLD|Same top p' "$log" | tr '\n' ' '; echo
}
# CPU-heavy jobs (llama-quantize, GPTQ) hold 5-8 GB: start one only while at least 30% of memory is free
mem_wait() { while [ "$(memory_pressure | awk '/free percentage/ {sub("%","",$5); print $5}')" -lt "${MEM_FREE:-30}" ]; do sleep 30; done; }
# CPU-heavy jobs (GPTQ sweeps and encodes, splices, quantize) run one at a time across both repos (agreed with the
# sibling 22 Nov 05:25): ~/GitHub/.cpu.lock, same format as .gpu.lock ("<pid> <owner> <time>"), created with set -C,
# stale when the pid is dead, removed only by its owner. mem_wait alone checks only at start (05:20 overlap).
CLOCK="$HOME/GitHub/.cpu.lock"
cpu_take() {
    while :; do
        if (set -C; echo "$$ under10 $(date '+%H:%M:%S')" > "$CLOCK") 2>/dev/null; then mem_wait; return 0; fi
        pid=$(awk '{print $1}' "$CLOCK" 2>/dev/null || true)
        [ "$pid" = "$$" ] && return 0
        case "$pid" in ''|*[!0-9]*) ;; *) kill -0 "$pid" 2>/dev/null || { rm -f "$CLOCK"; continue; } ;; esac
        sleep 20
    done
}
cpu_give() { [ "$(awk '{print $1}' "$CLOCK" 2>/dev/null)" = "$$" ] && rm -f "$CLOCK"; return 0; }
