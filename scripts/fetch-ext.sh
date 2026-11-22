#!/bin/sh
# Comparator GGUFs under the 16 GB Mac limit, pinned revisions, into models/ext/ (resumes). Total about 33 GB.
set -eu
. "$(dirname "$0")/lib.sh"
E="$M/ext"; mkdir -p "$E"
get() { # repo revision file alias
    [ -e "$E/$4.gguf" ] && return 0
    step "download $1/$3"
    HF_HUB_DISABLE_XET=1 hf download "$1" "$3" --revision "$2" --local-dir "$E/$(echo "$1" | tr / _)" > "$L/fetch-$4.log" 2>&1
    ln -sf "$E/$(echo "$1" | tr / _)/$3" "$E/$4.gguf"
}
get unsloth/Inkling-2-Mini-GGUF    ae9327c5fbc7333116ba32eee78bb5d2bbefae4f Inkling-2-Mini-UD-IQ2_XXS.gguf  UD-IQ2_XXS
get AtomicChat/Inkling-2-Mini-GGUF 4f0c56af7e0c242e620be6394983d245b8c3fd27 Inkling-2-Mini-AD-IQ2_M.gguf    AD-IQ2_M
get AtomicChat/Inkling-2-Mini-GGUF 4f0c56af7e0c242e620be6394983d245b8c3fd27 Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf AD-IQ2_XXS-Q2
get unsloth/Inkling-2-Mini-GGUF    ae9327c5fbc7333116ba32eee78bb5d2bbefae4f Inkling-2-Mini-UD-IQ1_M.gguf    UD-IQ1_M
step "downloads done"
