# Inkling 2 Mini for a 16 GB Mac: concept

Written 21 Nov 2026, before any candidate file exists. Changes after that go to `CHANGELOG.md` with a date.

## Goal

One GGUF of `thinkingmachines/Inkling-2-Mini` that runs well on a 16 GB Apple Silicon Mac with llama.cpp. It should be closer to the model than any public GGUF of the same size or smaller, and it should be built in a way nobody on Hugging Face has used for this model.

## What sets the size

- A 16 GB Mac gives Metal about 10.67 GiB (11,453 MB) by default (`recommendedMaxWorkingSetSize` is 2/3 of RAM).
- Inkling's KV cache is small. 20 of 24 layers use a 1,024-token sliding window, and only 8 of the 48 layer visits at P = 3 attend over the full context (layers 5, 11 x3, 17 x3, 23). At F16 that is about 32 KiB per token, so 16K context costs about 0.5 GiB plus about 0.16 GiB of window cache.
- Measured on the sibling's 11.5 GB file at 16K depth: maximum RSS 13.9 GB, so about 2.4 GB above the weights.
- Byte limit: **8,486,236,352 bytes**, the size of Unsloth `UD-IQ2_XXS`. Weights plus 2.4 GB stays under 11,453 MB. The file must not exceed it.

## What the public files under the limit do (read from their GGUF headers on 21 Nov)

| File | Bytes | Down projections | gate_up | Off-Metal expert tensors |
|---|---|---|---|---|
| unsloth UD-IQ2_XXS | 8,486,236,352 | Q2_0 x17, NVFP4 x7 | IQ1_S x6, IQ2_XXS x7, IQ2_XS x6, IQ2_S x5 | 7 |
| unsloth UD-IQ1_M | 8,024,441,024 | Q2_0 x21, NVFP4 x3 | IQ1_S x6, IQ2_XXS x9, IQ2_XS x5, IQ2_S x4 | 3 |
| AtomicChat AD-IQ2_XXS-Q2 | 7,900,679,776 | Q2_0 x24 | IQ2_XXS x24 | 0 |
| AtomicChat AD-IQ2_M | 8,665,091,680 (over) | Q2_0 x16, IQ4_NL x8 | IQ2_XXS x24 | 0 |

- The 1,152-wide down rows rule out every 256-block type, so everyone below 10 GB uses Q2_0 (2.25 bpw) for most down projections.
- Their Q2_0 is not llama.cpp's reference encoder (which never uses the +2d level). On expert 0 of layers 0 and 12 their imatrix-weighted weight error is 0.128 to 0.134, ours (`q2opt`) is 0.119 to 0.126. Rounding each weight better on its own is no longer an edge.
- Unsloth keeps 3 to 7 down projections in NVFP4, which the pinned Metal backend cannot multiply. On a Mac those run on the CPU.
- LookAtMyQuant 8.9GB is a custom `.npz` format with its own runtime, not a GGUF. Out of scope.

## The idea: GPTQ rounding inside existing GGUF types, plus a measured allocation

DASLab's GGUF releases (GSQ-RCO, e.g. their Qwen3.8-Flash-Next card) combine better rounding of the codes with an allocation optimised on real loss, and they use Q2_0 and IQ4_NL on non-256 down rows. We cannot run GSQ (no PyTorch forward for this model), but two parts transfer:

1. **GPTQ for the Q2_0 down projections.** Every public file rounds each weight independently, weighted by the imatrix diagonal. GPTQ uses the full input Gram X^T X of each expert to push each rounding error onto the weights not yet rounded. The codes stay plain Q2_0, so the file runs on every backend and on Metal at Q2_0 speed. The Gram comes from a 70-line patch to `llama-imatrix` (patch 0008), pooled over all three visits of a looped layer (the visit-summed Hessian of Li, arXiv 2609.30820).
2. **An allocation chosen by measured KL.** Per-layer types for down (Q2_0 or IQ4_NL), gate_up (IQ1_S to IQ3_XXS) and the non-expert groups, chosen under the byte limit from measured costs, then checked by building the files and scoring them.

All expert tensors stay in types Metal multiplies.

## Gates, in order

- **G1, GPTQ is real (offline).** On held-out calibration text (`data/calib-eval.txt`, 36 conversations never used for fitting), GPTQ-Q2_0 must have lower relative output error than diag-weighted Q2_0 on layers 0 and 12. If not, drop GPTQ and say so. Pilot with 24K tokens: no held-out gain (GPTQ overfits), so this waits for the full capture (about 280K tokens).
- **G2, GPTQ helps the model.** A build with GPTQ down on one block must have lower KL on S than the same build with diag-weighted down. S is the selection set.
- **G3, selection.** Allocations are built and scored on S only. The lowest S KL under the byte limit becomes the candidate. It is frozen before T1 or T2 is scored.
- **G4, release test.** The candidate must have lower KL than UD-IQ2_XXS on both T1 and T2 with the 95% paired-bootstrap CI of the difference below zero, and top-1 agreement not worse. It is also compared with UD-IQ1_M and AD-IQ2_XXS-Q2 (smaller) and AD-IQ2_M (larger, context only).
- **G5, it fits.** Peak Metal memory at 16K context below 11,453 MB, measured with `llama-bench` on this Mac and the sizes read from its log. Speed with `llama-bench -p 512 -n 128` next to UD-IQ2_XXS.

## Measurement

- Reference: the sibling's Q8_0 conversion at P = 3, and its reference logits on S (16 x 4,096), T1 (16 x 4,096) and T2 (5 x 2,048). Same scorer, same commit, so numbers are comparable with the sibling's.
- Statistics: paired bootstrap over chunks, 10,000 resamples, seed 0 (the sibling's `analysis/compare.py`).
- Calibration text for Grams: bartowski v6 prose and conversations rendered with the model's chat template (`scripts/render-calib.py`). The sibling checked that S and T1 share no 13-gram with it.
- Traps carried over: count bytes from the file, not the label; `gguf-dump` the types of every build; `.scale` tensors in Unsloth files need patch 0001; no task-quality claims from KL.

## Not doing, and why

- GSQ or RCO proper: needs a PyTorch forward of the looped model. Not available.
- GPTQ for gate_up: its types (IQ1/IQ2/IQ3) are vector codebooks, so independent scalar codes cannot be packed. Q2_0 or Q2_K for gate_up could take GPTQ codes. Tried only if time allows.
- Expert pruning, QAT, a custom format: not llama.cpp-compatible or not doable in hours.
- Uploads: none without Daniel's go.
