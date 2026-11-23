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
            [ "$pid" = "$$" ] && return 0  # already ours (a caller holding the lock for a whole window)
            case "$pid" in ''|*[!0-9]*) ;; *) kill -0 "$pid" 2>/dev/null || { rm -f "$LOCK"; continue; } ;; esac
        elif ! pgrep -x llama-bench > /dev/null && (set -C; echo "$$ under10 $(date '+%H:%M:%S')" > "$LOCK") 2>/dev/null; then
            return 0  # callers own their traps (a trap here replaced theirs and dropped cpu_give)
        fi
        sleep 20
    done
}
gpu_done() { [ -e "$LOCK" ] && [ "$(awk '{print $1}' "$LOCK")" = "$$" ] && rm -f "$LOCK"; return 0; }
# score MODEL on SET (S, T1, T2) against the sibling's Q8_0 reference logits at P = 3 -> logs/ppl-NAME.log
# Hardened 23 Nov after the Astra and Fable code reviews (reviews/2026-11-23-code-pipeline-*.md):
# - the key binds model and reference (path, size, mtime, inode), the scorer binary AND its dylibs, and the extra
#   arguments with their boundaries; it is taken again after gpu_wait, so a file replaced while waiting is caught;
# - extra arguments may not override the model, text, reference, context or chunk count (--chunks is a no-op in
#   KL mode: the reference .kld fixes the chunks);
# - the run writes ppl-NAME.log.part, and only a log with a finite final KL and the expected chunk count for the set
#   becomes ppl-NAME.log; the .sig is written before the GPU lock is released.
# Runs from before 23 Nov keep their old key (path size mtime set args ref bin), accepted by _slegacy.
_sfile() { _sf=$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1"); printf '%s %s' "$_sf" "$(stat -L -f '%z %m %i' "$_sf")"; }
_sdylibs() { stat -f '%m' "$BIN"/llama-perplexity "$BIN"/*.dylib 2>/dev/null | cksum | cut -d' ' -f1; }
_schunks() { case $1 in S|T1) echo 16 ;; T2) echo 5 ;; batA-*) echo 32 ;; *) echo "" ;; esac; }
# sets: S, T1, T2 (our data/, the sibling's ref-SET-P3.kld) and batA-<domain> (the sibling's fresh-domain battery A:
# its data/battery/A-<domain>.txt and models/logits/batA-<domain>-P3.kld, read-only, at its settings -c 1024 --chunks 32)
_sset_paths() { case $1 in
    batA-*) _stext="$SIB/data/battery/A-${1#batA-}.txt"; _sref="$SIB/models/logits/$1-P3.kld"; _sctx=1024; _sxtra="--chunks 32" ;;
    *) _stext="$ROOT/data/$1.txt"; _sref="$SIB/models/logits/ref-$1-P3.kld"; _sctx=4096; [ "$1" = T2 ] && _sctx=2048; _sxtra="" ;;
    esac; }
_sok() { # log set: a finite final KL, no nan/inf chunk, and the expected last chunk index
    grep -Eq 'Mean +KLD: +[0-9]+\.[0-9]+' "$1" || return 1
    ! grep -Eq '^ *[0-9]+ .*(nan|inf)' "$1" || return 1
    _sn=$(_schunks "$2"); [ -z "$_sn" ] && return 0
    [ "$(grep -E '^ *[0-9]+ +[0-9.]+ ±' "$1" | tail -1 | awk '{print $1}')" = "$_sn" ]
}
score() { # name model set [extra...]; its variables are _s-prefixed so callers' loop variables survive
    _sname=$1 _smodel=$2 _sset=$3; shift 3
    for _sa in "$@"; do case $_sa in -m|--model|-f|--file|--kl-divergence-base|-c|--ctx-size|--chunks|--override-kv)
        echo "score $_sname: '$_sa' is set by score() and may not be passed"; return 1 ;; esac; done
    _slog="$L/ppl-$_sname.log"; _sset_paths "$_sset"
    _skey() { printf 'v2 %s set %s args' "$(_sfile "$_smodel")" "$_sset"; printf ' [%s]' "$@"; printf ' ref %s bin %s' "$(_sfile "$_sref")" "$(_sdylibs)"; }
    _slegacy="$(cd "$(dirname "$_smodel")" && pwd -P)/$(basename "$_smodel") $(stat -L -f '%z %m' "$_smodel") $_sset $* ref $(stat -L -f '%z %m' "$_sref") bin $(stat -f '%m' "$BIN/llama-perplexity")"
    _scached() { [ -f "$_slog" ] && _sok "$_slog" "$_sset" && { [ "$(cat "$_slog.sig" 2>/dev/null)" = "$(_skey "$@")" ] || [ "$(cat "$_slog.sig" 2>/dev/null)" = "$_slegacy" ]; }; }
    _scached "$@" && return 0
    gpu_wait
    _scached "$@" && { [ -n "${GPU_HOLD:-}" ] || gpu_done; return 0; }  # another process scored it while we waited
    step "score $_sname"; _ssig=$(_skey "$@")
    if ! "$BIN/llama-perplexity" -m "$_smodel" -f "$_stext" -c $_sctx $_sxtra -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
        --kl-divergence-base "$_sref" --kl-divergence "$@" > "$_slog.part" 2>&1 || ! _sok "$_slog.part" "$_sset"; then
        [ -n "${GPU_HOLD:-}" ] || gpu_done; echo "score $_sname FAILED, see $_slog.part"; return 1
    fi
    [ "$(_skey "$@")" = "$_ssig" ] || { [ -n "${GPU_HOLD:-}" ] || gpu_done; echo "score $_sname: model or reference changed during the run"; return 1; }
    echo "$_ssig" > "$_slog.sig.part"; mv "$_slog.part" "$_slog"; mv "$_slog.sig.part" "$_slog.sig"
    [ -n "${GPU_HOLD:-}" ] || gpu_done  # GPU_HOLD=1: the caller keeps the lock across scores
    grep -E 'Mean +KLD|Same top p' "$_slog" | tr '\n' ' '; echo
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
