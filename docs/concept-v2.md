# Concept v2 (22 Nov 2026, after night 1 and the morning runs)

KL against the sibling's Q8_0 reference at P = 3. S is the selection set; T1 and T2 are held out. Differences are paired over chunks. Every number comes from `results/night1.md` (generated from the logs) or `CHANGELOG.md`.

## Result

No file of ours beats Unsloth. Our best own-encoding file, M0rk (8,486,214,944 bytes, all-Metal, stock-llama.cpp types), beats both AtomicChat files on both held-out sets and loses to both Unsloth files:

| M0rk vs | T1 | T2 |
|---|---|---|
| UD-IQ2_XXS (Unsloth, 8.49 GB) | +0.01424 [+0.01249, +0.01593] | +0.01662 [+0.01414, +0.01941] |
| UD-IQ1_M (Unsloth, 8.02 GB) | +0.00411 [+0.00193, +0.00616] | +0.01628 [+0.01464, +0.01829] |
| AD-IQ2_XXS-Q2 (AtomicChat, 7.90 GB) | -0.04051 [-0.04712, -0.03499] | -0.02099 [-0.02523, -0.01776] |
| AD-IQ2_M (AtomicChat, 8.67 GB, context) | -0.01555 [-0.01871, -0.01278] | -0.00822 [-0.01193, -0.00498] |

Daniel's policy releases only our own encodings, and those that clearly beat UD. So there is no release. The default is option (a): publish the findings.

## What we achieved (measured)

1. **Where the gap to Unsloth sits.** At UD-IQ2_XXS's exact per-tensor types, llama-quantize (M0) is +0.02727 behind UD on S. By tensor class, UD's edge is gate_up 0.01676 (61%), non-expert 0.00629 (23%) and down 0.00422 (15%). The edge is UD's encoding, not its type map.
2. **Attenuation, the mechanism of the night.** MSE-style fits shrink the weights (row gain <q,w>/<w,w> < 1). The shrink compounds over the 48 layer visits of the looped model.
   - A mean-aware gate_up scale refit lowered the full input-Gram error in 128/128 experts and still raised KL by +0.07237 (M0r, gain 0.79).
   - The same refit with each row restored to llama-quantize's own per-row gain gains -0.00380 (M0rk vs M0u).
   - Our Q2_0 down at gain 0.86 rescaled to 1 gains -0.00698 (M0u vs M0). A sweep puts the optimum at 0.98-1.00 (0.93: +0.00084, 1.07: +0.00284).
   - Gain 1 is not a universal target. It hurts gate_up (M0uu +0.00138) and UD's own Q2_0 down at 0.90 (X6d +0.00185).
   - Rule: refit for output error, then restore the reference encoder's per-row gain.
3. **40% of the gap closed with own encoders.** M0rk - M0 = -0.01078 [-0.01193, -0.00960] on S. Our gate_up now closes 64% of UD's gate_up share.
4. **UD's remaining edge is invisible to diagonal metrics.**
   - On the 60 lower-bit attention and head matrices, ours beats UD in plain and in imatrix-weighted error 60/60. Every non-matrix tensor is byte-identical.
   - Unsloth's non-expert tensors are worth -0.00246 in our stack, against -0.00629 next to UD's own gate_up. So about 60% of that share is joint with UD's expert tensors.
   - The sibling's full-Gram code refinement matches UD's gate_up on its layers 9-11. That is the method that addresses this kind of edge.
5. **Domain shift.** GPTQ-Q2_0 won offline by 24-30% on calibration-like inputs, only by about -6% to +5.5% on S-text inputs, and lost on S three times (+0.00293, +0.00588, +0.00386).
6. **The importance source is not the lever.** Three imatrices for gate_up differ by less than 0.0004. Covering the output head in the imatrix is neutral (M0rko +0.00036 [-0.00009, +0.00084]).
7. **Tooling, reusable and tested.**
   - Imatrix patch for full per-expert input Grams.
   - GPTQ into Q2_0.
   - Gain/energy rescaler with `--target` / `--target-from` (`degain2.py`, shared with the sibling).
   - Mean-aware refits for IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S and Q2_0.
   - sha256-identical rebuild of Unsloth's mainline files.
   - Byte-limit-checked assembly.
   - Shared GPU/CPU locks with an agreed small-job rule.
8. **Two bugs found and fixed, both documented.** The "diag" control was unweighted rounding. `score()` clobbered its caller's loop variable (it broke one G4 run, no earlier result).

## Where we cannot compete

- **Unsloth's encoder** wins in every tensor class at equal types. Its method is not public. Its remaining edge needs full-Gram code search, which at our types (IQ1_S, IQ2_XXS, IQ2_XS) has no refiner yet. It would also need Gram captures and many CPU hours.
- **Unsloth's runtime.** Their prebuild already runs NVFP4 on Metal, so "all-Metal" alone is only a niche.
- **GPU-heavy methods** (GSQ/RCO class) need GPU time we don't have.

## What transfers

- The attenuation rule, measured on this looped MoE: lower per-layer error is not lower KL, but a gain-restored refit is. This bears directly on the thesis question H1 (does error compound over passes).
- The sibling adopted it as amendments 9c/9e and dropped its gain-1 mirror run because of X6d.

## If work continues

1. Full-Gram code refinement for IQ1_S, IQ2_XXS and IQ2_XS gate_up (extend the sibling's `refine-codes.py`), followed by the gain-restored scale refit. This is the only lever left with a measured share (gate_up about 0.006).
2. Full-Gram refinement of the attention matrices (dense, one Gram each; ceiling about 0.0025 in our stack).
3. A combined test at the sibling's size of its code refinement and our gain-restored scale refit.

**Park:** GPTQ variants, imatrix sources, gain-1 rescales, mean-aware Q2_0 down refit (surrogate -2%), donor mixes.
