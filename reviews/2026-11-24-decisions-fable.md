# Review of the release decisions (c1, M0rk)

- Date: 2026-11-24, ~05:30
- Model: Claude Fable 5.2, effort medium (subagent, prompt file only, no other tools)
- Repo HEAD: 3becc2f (tracked files changed: 0)
- Why: Daniel, relayed by the sibling: "I decide the default switch at the very end. until then keep running, your sibling as well. Ask Astra high and Fable medium for second opinion to answer the rest".

## Prompt

You are a skeptical reviewer of release decisions for a model-quantization project. You did not do the work. Answer in under 500 words. For each numbered decision give: your recommendation (yes / no / change it, and how), the main risk, and one thing the author must check first. Then add at most 3 points the decision list is missing. No praise.

## Setup (public facts)

- Model: Inkling 2 Mini, a looped MoE (24 layers; layers 6-17 run three times per token on the same experts; 128 experts, top-8). Quantized to llama.cpp GGUF for 16 GB Macs. A sibling project already published an 11.5 GB file on a "Mac track": lowest KL among files whose every tensor runs on the Mac GPU (Metal) in stock llama.cpp, with the gap to Unsloth stated.
- Stock llama.cpp Metal has no matmul kernel for NVFP4. Unsloth's files keep 1-11 of their 48 expert tensors in NVFP4, which then run off the GPU.
- Metric: KL divergence of next-token distributions against a Q8_0 reference at 3 loop passes, paired bootstrap over chunks (10,000 resamples). Sets: S (16 x 4096 WikiText-2 test tokens, used for selection), T1 (16 x 4096 WikiText-2 test, disjoint), T2 (5 x 2048 reasoning traces the model wrote on GSM8K train prompts), T3 (16 x 4096 WikiText-2 test, sealed, opened once after the tests below were written down). S, T1 and T2 informed choices during development. Off-domain battery: python, rust, German, maths texts (8 x 4096 tokens each, scored as 32 x 1024), never used in development.
- Two candidate files, both our own encodings (llama-quantize pools, a mean-aware refit of gate_up block scales with each row's output gain restored to llama-quantize's, Q2_0 down rows rescaled to gain 1), every tensor in a Metal type:
  - M0rk: 8,486,214,944 B, Unsloth UD-IQ2_XXS's per-layer type map with NVFP4 downs replaced by IQ4_NL (same bytes).
  - c1: 8,932,121,888 B, M0rk's recipe plus 446 MB of per-layer upgrades chosen one by one on S and T2.
- KL (T1 / T2 / T3), lower is better:
  - c1 0.0668 / 0.0489 / 0.0703. M0rk 0.0801 / 0.0593 / 0.0857.
  - All-Metal comparators: GSQ-RCO-IQ2_XS (8.95 GB) 0.0890 / 0.0557 / 0.0937; AD-IQ2_M (8.67 GB) 0.0956 / 0.0676 / 0.0992; AD-IQ2_XXS-Q2 (7.90 GB) 0.1206 / 0.0803 / 0.1239; two smaller AD files far worse.
  - Unsloth (not all-Metal): UD-IQ2_XXS (8.49 GB, 7 NVFP4 tensors) 0.0658 / 0.0427 / 0.0691; UD-IQ1_M (8.02 GB, 3 NVFP4) 0.0760 / 0.0431 / not scored on T3; UD-Q2_K_XL (9.14 GB, 11 NVFP4) 0.0510 / 0.0349 / not scored.
- Pre-registered T3 tests: c1 beats each of the 5 all-Metal files at or under 9 GB, M0rk each of the 3 at or under 8.49 GB, CI below 0 and top-1 not lower. Both PASS, all CIs far from 0 (closest: c1 vs GSQ-RCO -0.0233 [-0.0263, -0.0203]).
- c1 vs UD-IQ2_XXS: T1 +0.0010 [-0.0009, +0.0028], T3 +0.0013 [-0.0004, +0.0029] (both span 0), T2 +0.0062 [+0.0032, +0.0087], off-domain KL ratio 1.03-1.13 (python and rust CIs span 0, German and maths UD better). M0rk vs UD-IQ2_XXS: ratio 1.24 on T3. c1 also beats both all-Metal comparators in every off-domain text.
- Stability battery (both files, against the Q8_0 reference): no degenerate repetition in long generations, planted fact recalled, 5/5 valid tool calls, 2/2 non-English, GSM8K 30/30 with 0 capped. PASS for both.
- Unpatched upstream llama.cpp: identical KL to our build, chunk for chunk.
- Speed on one M-series Mac, stock llama.cpp: c1 268 pp512 / 21.6 tg128 t/s, M0rk 269 / 21.9, UD-IQ2_XXS 129 / 12.4 (that one from a different session).
- Memory: default GPU budget of a 16 GB Mac is 11,453 MB. Measured on a 48 GB Mac (llama-bench at depth, peak footprint plus weights), projected: at 16K context c1 uses 11,408 MB (45 MB spare), M0rk 10,963 MB (490 MB spare). Neither fits 32K. No real 16 GB Mac and no LM Studio load test yet.
- Known metric limitation: per-chunk KL is reconstructed from running means printed with 5 decimals, so chunk k carries up to about k x 1e-5 of rounding noise. Means are exact; CIs widen slightly. A fix needs reruns with a patched printer.
- The GGUF header's general.file_type of both files still says "Q8_0" (inherited from the base file the tensors were spliced into); llama-bench and possibly LM Studio display that label.

## The decisions

1. Release c1, M0rk, or both, on the Mac track with the gap to Unsloth stated. Proposed: both; c1 recommended for 16 GB Macs at up to 8K-12K context, M0rk for users who need 16K or run other apps.
2. Fix the KL rounding (patched printer, rerun all T3 and T2 scores) before any release. Proposed: no; state the limitation, since no verdict is within the bound.
3. Advertised context for c1 given 45 MB projected headroom at 16K. Proposed: advertise 8K as safe, 16K as measured-but-tight.
4. Required before upload: an LM Studio load test on this Mac, correcting general.file_type, and recomputing sha256 after any header change. Proposed: yes, all three.
5. Wording of the Unsloth comparison: "ties UD-IQ2_XXS on held-out prose (T3), trails on reasoning traces and German/maths, about 2x faster on stock llama.cpp". Proposed as is.


## Reply

## Decisions

1. Release both: change it. Release c1 as the file; ship M0rk only as the "more headroom" variant, with its limits stated.
   - Risk: M0rk's pre-registered win is over three weak, smaller files. It loses to GSQ-RCO on T2 (0.0593 vs 0.0557), and to UD-IQ2_XXS by 1.24x at identical bytes. UD-IQ1_M (8.02 GB) also beats it on T1 and T2.
   - Check first: a paired c1 vs M0rk CI on T3, and whether M0rk's type map needs Unsloth credit on the card.

2. Skip the rounding fix: change it. Skip the full rerun, but rerun c1 and UD-IQ2_XXS on T1 and T3 with the patched printer.
   - Risk: "no verdict is within the bound" holds for the pre-registered tests, not for the tie. The T3 CI lower edge is -0.0004, and paired chunk noise reaches about 3e-4 at chunk 16. The noise widens CIs, so the clean CI may exclude 0 and turn "ties" into "trails".
   - Check first: the worst-case paired noise against that -0.0004 edge.

3. Context wording: change it. Advertise 8K; call 16K "projected, untested on 16 GB hardware", not "measured".
   - Risk: 45 MB is inside the error of a projection from a 48 GB machine. The 11,453 MB budget is shared with other GPU users, so a real load may fail or swap.
   - Check first: load c1 on the 48 GB Mac with the GPU wired limit lowered to 11,453 MB, at 8K, 12K and 16K.

4. Pre-upload gates: yes, in a fixed order. Fix the header, recompute sha256, then run the load test and one KL chunk on the final bytes.
   - Risk: the scored file and the uploaded file differ. No file_type enum fits a mixed-type file, so the label may still mislead.
   - Check first: chunk-identical KL after the header edit, and what LM Studio then displays.

5. Unsloth wording: no, rewrite it.
   - Risk: "ties" reads a non-significant result as equivalence. The point estimate is positive on both T1 and T3. The sentence omits that c1 is 446 MB (5%) larger, and "2x faster" compares across sessions on one Mac.
   - Suggested: "within about 2% KL of UD-IQ2_XXS on held-out prose (difference not resolved) at 5% more bytes; about 15% worse on reasoning traces, worse on German and maths; about 2x faster on stock llama.cpp Metal (one Mac)".
   - Check first: rerun UD-IQ2_XXS speed in the same session as c1, same thermal state and flags.

## Missing from the list

- Per-byte framing: the encoding does not beat Unsloth per byte at any tested size. The release case rests on all-Metal execution and speed, and speed rests on one cross-session number. The card should say so, and decision 5's rerun should be a gate.
- Shelf life: the track exists only while stock Metal lacks an NVFP4 kernel. Pin the llama.cpp commit and the LM Studio runtime version, and state that the speed claim expires if a kernel lands.
- Task-level evidence: the stability battery cannot discriminate (both files pass, GSM8K saturates at 30/30) and T2 has 5 chunks. c1's known weakness is reasoning traces, so a larger reasoning set or an accuracy test with headroom is needed before recommending it for that use.

## Response (joint, both reviews)

1. Release both. **Accepted**, both reviewers: c1 is the default, and M0rk is the lower-memory variant with its limits stated on the card. Those limits are UD-IQ2_XXS at 1.24x on T3, losing to GSQ-RCO on T2, and UD-IQ1_M (smaller) better on T1/T2. The card credits Unsloth for M0rk's per-layer type map (Fable). Paired c1 minus M0rk on T3: -0.01534 [-0.01725, -0.01333], sign robust (results/rounding-bound.txt).
2. Rounding. **Accepted (bound instead of rerun)**: worst-case CI shift is 2(2N-1) x 5e-6, i.e. ±0.00031 at 16 chunks, ±0.00009 at 5 and ±0.00063 at 32. Every pre-registered verdict and c1/UD on T2, German and maths keep their sign. c1/UD on T1 and T3 spans 0 even at worst. c1/UD on python and rust is **not robust**, so they are worded "not resolved at the printed precision". A clean rerun is not possible now, because the T3 and battery A references were deleted after scoring. Open: a patched printer for future runs.
3. Context. **Accepted**: advertise 8K. Describe 16K as "projected from a 48 GB Mac, 45 MB under the default 16 GB budget, untested on 16 GB hardware". Fable's check, lowering the GPU wired limit on this Mac, needs sudo and would affect the sibling's runs. **Open**, for Daniel.
4. Pre-upload gates. **Accepted**, in Fable's order (scripts/release-prep.sh):
   - Header general.file_type set to 19 (MOSTLY_IQ2_XXS, as UD at the same map) on APFS clones, exactly 1 byte different from the frozen files.
   - sha256 recorded, and S checked identical chunk for chunk on the release copies.
   - A verbose run per file logs graph splits as runtime evidence (Astra).
   - The LM Studio load test is **open**: LM Studio is not installed on this Mac, and installing it is Daniel's call.
5. Unsloth wording. **Accepted**: no "ties". The wording is "on held-out WikiText (T3) c1's KL is 0.0013 above UD-IQ2_XXS (95% CI -0.0004 to +0.0029, not resolved) at 5% more bytes; UD is lower on reasoning traces (T2, +14%), German and maths". Speed goes as two same-session ratios (pp and tg), from release-prep.sh's bench.
- Missing points:
  - Per-byte framing: **accepted**. The card says our encoding does not beat Unsloth's per byte, and that the case rests on all-Metal execution and speed.
  - Shelf life: **accepted**. Pin the llama.cpp commit, and say the speed claim expires once stock Metal gets an NVFP4 kernel.
  - Runtime Metal evidence: **accepted** (item 4).
  - Task-level evidence: **open**. A task battery with headroom (e.g. the sibling's battery B) on c1, M0rk and UD-IQ2_XXS is hours of GPU and needs Daniel's go. Until then the card scopes the claim to KL plus a stability smoke test.
