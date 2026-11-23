# Code review: build and scoring pipeline (Fable)

- Date: 2026-11-23
- Model: claude-fable-5-2 (Agent, read-only, small numpy checks in the scratchpad)
- Scope: under10 scripts/meanfit.py, degain2.py, assemble.py, lib.sh, c1.sh (and c1r.sh, analysis/kltable.py, compare.py as read); sibling scripts/refine-codes.py, gram-error.py, splice.py at the sibling's commit recorded in CHANGELOG (23 Nov 08:45). This repo at 0fe5435.
- Asked for by Daniel (23 Nov): "let Astra and Fable do the code review and then ask me if something is still left. no hard release limitations bound to my code review."

## Reply (verbatim)

No correctness bug in the numerical core (meanfit.py, refine-codes.py, gram-error.py pass every check I ran). The real risks are in resume and caching logic, where a reported KL can belong to an older file than claimed.

Findings, ranked:

1. c1.sh:22 (also :9, c1r.sh:19,36), resume by existence only. [ -f "$out" ] || skips assembly whenever models/tmp/c1.gguf exists, and donors likewise. Trigger: delete or regenerate down/rfmx-L4.gguf (or change m0rk-mix.json, a pool, the imatrix) and rerun. c1.json is rewritten, but the old c1.gguf is hashed and scored, and score() then hits its cache because the signature matches the old file. So old KLs are reported as the new c1. c1r.sh likewise splices refined tensors derived from an older c1. Fix: a manifest per artifact (inputs' size+mtime or sha, plus the spec), rebuild on mismatch. Reading only.
2. splice.py:118,133-140 with assemble.py:119, non-atomic output. Splice writes straight to OUT and sys.exits on a --verify failure without removing it. Trigger: verify fails or the run is killed, then c1.sh is rerun. Finding 1 then accepts the unverified file. c1r.sh already uses .part; assemble.py does not. Fix: write OUT.part and rename after verify. Reading only.
3. analysis/kltable.py:46 and compare.py, kl.tsv carries no model identity. Every logs/ppl-*.log is collected by name and .sig is never consulted. Trigger: a baseline such as m0rk-mix-S is rebuilt but not rescored, and c1 is then compared against a KL from the previous baseline file. Fix: skip logs whose .sig no longer matches, and store size, mtime or sha in the tsv. Reading only.
4. lib.sh:30 score(), the signature covers only the llama-perplexity binary, but the build is shared-library (libggml*.dylib). Trigger: a patch rebuild that updates a dylib without relinking the tool gives a stale cache hit. Fix: add the dylib mtimes, or the vendor commit plus patch hash. Also, the cache is not rechecked after gpu_wait, so two scripts scoring the same name both run, and the second truncates a log the first may be parsing. Dylibs verified by listing, the rest by reading.
5. lib.sh:15,51 and the sibling's gpu-lock.sh:17, stale-lock removal is a check-then-remove race. The holder dies, and waiters A and B both see the dead pid. A removes and takes the lock; B's pending rm -f deletes A's fresh lock and B takes it too, so two heavy jobs run at once. Fix: mv to a unique name and confirm the dead pid before deleting, or use lockf/shlock. Related: ownership is $$, which background subshells share (c1r.sh:27). If capture B fails, the parent's EXIT trap releases the CPU lock while the orphaned refine A keeps running, and a rerun writes the same .part. Reading only.
6. degain2.py:29-31, --target-from gain ignores that file's .scale (wrong for Unsloth references). A reference row gain at or below 0 gives fac <= 0, which zeroes or sign-flips the row, and allclose (against q*fac) passes. Rows with gain at or below 0.2 are skipped without a count. fp16 overflow or a subnormal d trips the assert (a crash, not corruption). Fix: apply the scale, require target > 0, print the skipped-row count. Reading only. Not hit in c1.sh (pools have no .scale).
7. meanfit.py:140-143, the "only scale bits changed" check in the docstring is not enforced: changed is computed for the last expert only and printed. No current bug (set_ls is correct), but a regression would go unnoticed. Fix: assert (new ^ blocks) & ~scale_mask == 0 per expert.
8. assemble.py:89-93, the size prediction disagrees with splice on overlapping ne groups (last-wins vs splice's first-match-wins), e.g. a spec with both "all" and "attn". The prediction is then wrong, but the post-build size check still catches an over-limit file.
9. kltable.py:29-32, per-chunk KL is rebuilt from running means printed with %10.5lf, so chunk k carries up to about k*1e-5 of rounding noise. Means are exact and bootstrap CIs widen slightly. Negligible at 16 chunks.

What checked out:
- meanfit.py, on random blocks of all 4 types:
  - set_ls touches only IQ1_S qh high bytes with mask 0x70 (bits 12-14; delta bit 15 and grid bits intact), IQ2_XXS byte 7 of each group with mask 0xF0, IQ2_XS bytes 66-73 and IQ2_S bytes 74-81.
  - Decode scales exactly by f(ls1)/f(ls0), and restoring the old bits reproduces the bytes.
  - Sherman-Morrison equals a dense solve to 1e-15. The reported error equals direct recomputation, and no row gets worse.
  - Zero-token experts and zero-d blocks stay byte-identical. v=0 with m!=0 stays finite.
- refine-codes.py (IQ2_S and IQ3_XXS; full-rank, rank-deficient and zero-column Grams):
  - Decode is bit-equal to gguf-py, and set_group round-trips.
  - IQ3_XXS parity is enforced, and d and the scale bits are unchanged.
  - Tracked J equals recomputed J, and no row gets worse.
  - With the guard on, the minimum per-row output-gain change is at or above 0; with it off, it reaches -0.43.
- gram-error.py: nothing serious by reading (lines 1-330 plus a skim). splice.py: nothing beyond finding 2.

Limits: decoders were checked against gguf-py; the C struct order and dequantize_row_iq1_s were read and agree, but the C decoders were not run. No real model tensors were opened. Test scripts: scratchpad fable-review/t1.py, t2.py.

## Response

See the joint Response in reviews/2026-11-23-code-pipeline-astra.md (Fable numbers in brackets).
