#!/bin/sh
# Builds the pinned llama.cpp plus patches/llama.cpp/*.patch (Release, Metal) into vendor/llama.cpp/build,
# runs the loop-scale test and writes results/env.md. Safe to rerun; it refuses to reset a tree with uncommitted edits.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BASE=29b52d398b294509746ec0a3f28948a3107c0213 # b12600, PR #32694 "Inkling 2 Mini"
SRC="$ROOT/vendor/llama.cpp"
TARGETS="llama-cli llama-server llama-perplexity llama-quantize llama-imatrix llama-gguf-split llama-bench test-llama-archs"

if command -v cmake >/dev/null 2>&1; then
    CMAKE=cmake
else
    CMAKE="uvx --from cmake==4.4.5 cmake"
fi

if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$ROOT/vendor"
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
git -C "$SRC" cat-file -e "$BASE^{commit}" 2>/dev/null || git -C "$SRC" fetch -q --depth 1 origin "$BASE"

if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]; then
    echo "vendor/llama.cpp has uncommitted changes: commit them and export them to patches/llama.cpp first" >&2
    exit 1
fi

# a fixed committer and dates make the patched HEAD hash reproducible
git -C "$SRC" checkout -q -B inkling-patches "$BASE"
git -C "$SRC" -c user.name=inkling-build -c user.email=build@invalid am -q --committer-date-is-author-date "$ROOT"/patches/llama.cpp/*.patch
HEAD=$(git -C "$SRC" rev-parse HEAD)

$CMAKE -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF >/dev/null
# shellcheck disable=SC2086
$CMAKE --build "$SRC/build" --target $TARGETS -j "$(sysctl -n hw.ncpu)" >/dev/null

TEST_OUT=$( { "$SRC/build/bin/test-llama-archs" --loop-scales -s 42; "$SRC/build/bin/test-llama-archs" --loop-slots -s 42; } 2>&1 | grep -E '^loop-(scales|slots)')
echo "$TEST_OUT"
# 3 scale checks + 9 slot checks; any FAIL or a missing line stops the build
{ ! echo "$TEST_OUT" | grep -q FAIL && [ "$(echo "$TEST_OUT" | grep -c 'OK')" -eq 12 ]; } || { echo "loop tests failed" >&2; exit 1; }

mkdir -p "$ROOT/results"
{
    echo "# Build environment"
    echo
    echo "Written by \`scripts/build-llama.sh\` on $(date -u '+%Y-%m-%d %H:%M UTC'). Do not edit by hand."
    echo
    echo "- llama.cpp base: \`$BASE\` (b12600)"
    echo "- Patched HEAD: \`$HEAD\`"
    echo "- Patches (sha256, first 16):"
    for p in "$ROOT"/patches/llama.cpp/*.patch; do
        echo "  - \`$(basename "$p")\` \`$(shasum -a 256 "$p" | cut -c1-16)\`"
    done
    echo "- Build: Release, GGML_METAL=ON, $($CMAKE --version | head -1), $(cc --version | head -1)"
    echo "- Machine: $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion), iogpu.wired_limit_mb=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo default)"
    echo "- Loop tests (\`test-llama-archs --loop-scales -s 42\`, \`--loop-slots -s 42\`):"
    echo "$TEST_OUT" | sed 's/^/  - /'
} > "$ROOT/results/env.md"
echo "wrote results/env.md (HEAD $HEAD)"
