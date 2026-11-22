# Concept v2 (22 Nov 2026, after night 1)

Every number is on S (16 chunks, KL vs the Q8_0 reference at P = 3), paired over chunks. Source: `results/night1.md` (generated from the logs) and `CHANGELOG.md`.

## What we achieved (measured)

1. **Where the gap to Unsloth sits at 8.49 GB.** M0 (UD-IQ2_XXS's exact per-tensor types, every tensor encoded by us) is +0.02727 behind UD. Swapping in UD's own tensors by class splits that gap: gate_up 0.01676 (61%), non-expert 0.00629 (23%), down 0.00422 (15%). This independently replicates the sibling's 11.6 GB result: the edge is UD's encoding, not its type map.
2. **The importance source is not the lever.** gate_up re-encoded with three imatrices (ggml-org, bartowski, ours) differs by less than 0.0004. The sibling found the same for domain and quantity.
3. **Two mechanisms behind "better offline, worse in the model".**
   - Domain shift. GPTQ-Q2_0 had 24-30% lower held-out output error on calibration-like text, but only -6% (layer 12) or +5.5% (layer 0) on S-text inputs. It lost on S three times: G2 +0.00293, X1 +0.00588, X3 +0.00386.
   - Attenuation. A mean-aware scale refit lowered the full input-Gram error (layer 6: 0.129 to 0.107, 128/128 experts) and still raised KL by +0.07237. It shrinks the weights: gain <q,w>/<w,w> is 0.79 vs 0.94 for llama-quantize and UD at IQ1_S. Over 48 layer visits that compounds. UD lowers the error without attenuating.
   - Rescaling our Q2_0 down to gain 1 does not close the down gap (X5 +0.00333 vs X2). Attenuation explains the refit failure, not all of UD's edge.
4. **X2**, UD-IQ2_XXS with its 7 NVFP4 down projections re-encoded as IQ4_NL, runs fully on stock-llama.cpp Metal at +0.00063 [+0.00026, +0.00101] vs UD. That is at the backend noise floor (the sibling measured up to +0.0004).
5. **Tooling that others can reuse.**
   - An imatrix patch that captures full per-expert input Grams, pooled over the loop visits.
   - A GPTQ encoder into plain Q2_0.
   - A local rebuild of Unsloth's mainline files, sha256-identical.
   - Byte-limit-checked assembly and mixing.
   - Shared GPU and CPU locks across both repos.
6. **The audit found one real bug.** The "diag" control was uniform rounding. It is fixed and documented; every other logged number reproduced.

## Where we cannot compete

- **Unsloth's encoder.** It wins in every tensor class at equal types. Its method is not public, and no encoder of ours (GPTQ, imatrix variants, scale refit, de-attenuation) closes any part of the gap.
- **Unsloth's release and runtime.** Their prebuild (b12588-mix) already runs NVFP4 on Metal. "All-Metal" is therefore only a niche, for stock llama.cpp and the apps built on it.
- **GPU-heavy methods.** GSQ/RCO-class methods need a PyTorch forward of the looped model and GPU time we don't have.

## The special win

There is none at the file level. Nothing beats UD-IQ2_XXS on S. What is new and transfers is a finding:

- In this looped MoE, lower per-layer reconstruction error does not mean lower KL, even under the full input Gram.
- Two measured causes: domain shift of the Gram, and attenuation compounding over 48 visits.
- A practical rule follows: any encoder or refiner should log gain and energy per tensor, and reject moves that lower the gain. The sibling adopted this as amendment 9c.
- This bears directly on the thesis question H1 (does error compound over passes).

## Lean order from here

1. **Bench X2 vs UD on stock llama.cpp** (G5: speed at depth 0 and 16K, Metal buffer sizes). This is the only fact X2 still lacks (queued).
2. **M0u** (our q2opt down at gain 1 inside M0), as a mechanism check only (queued).
3. **Decision (Daniel).**
   - (a) No release; publish the findings and the decomposition.
   - (b) X2 as a clearly credited Unsloth repack for stock-llama.cpp Mac users, only if the bench shows a real speed gain. It then still needs G4 on T1 and T2 (T2 with its sign rule) and the label "equal to slightly worse than UD on KL".
   - My recommendation is (a), plus (b) only if the speed gain is large.
4. **Research to pursue only with more time.**
   - Find out how UD's codes differ from llama-quantize's at the same type, offline: code choice, sign patterns, scale bits, with gain and energy logged.
   - MACaQ-R-style full-Hessian code edits for the K-quant non-expert tensors, with a gain guard.

**Park:** GPTQ variants, imatrix sources, donor mixes beyond X2, and refits without a gain constraint.
