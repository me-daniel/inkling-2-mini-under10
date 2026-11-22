# Concept v3: a 16 GB Mac file, sized by measurement (22 Nov 2026)

Replaces the run order of concept-v2 for the next days. Reviewed by Fable (`reviews/2026-11-22-plan-9gb-fable.md`), Astra (`reviews/2026-11-22-plan-9gb-astra.md`) and Sol (`reviews/2026-11-22-research-9gb-sol.md`), with a file census and a literature pass by two Sonnet agents (`results/landscape-7.5-9.6GB.md`, `reviews/2026-11-22-research-levers-sonnet.md`). The draft they reviewed is `docs/draft-9gb-brief.md`.

## Route decision

- Stay on our route: M0rk (8,486,214,944 B, own encoding, all-Metal) is the base. Every reviewer arrived at this independently.
- Transfer from the sibling:
  - v1's release frame: the Mac track, the 7 stability checks, a score on unpatched upstream, speed at depth and peak memory.
  - Its attenuation instrument, as a per-block gain sweep in looped versus non-looped layers.
  - Its full-Gram refiner, as a pilot only.
- No broad new research. The Sonnet pass found no measured lever beyond what both repos already have. The one same-format refiner with a public result is MACaQ-R, and it covers K-quants only. Its other hint, Grams summed over loop passes, we already do. More instances would buy little.
- Dropped from the draft:
  - The M1 ladder at UD-Q2_K_XL's types. It has no spare bytes, since NVFP4 and IQ4_NL are both 4.5 bpw.
  - The block-map rule "looped and last first". Block KLs do not add, and B4 is out of the loop.

## What the reviews established

- **Fit (open conflict).** A 16 GB Mac gives Metal 11,453 MB by default (PLAN.md).
  - PLAN.md's measured overhead is 2.4 GB above weights at 16K, on the sibling's 11.5 GB file. That caps a 16K file at about 9.05 GB, and UD-Q2_K_XL (9.14 GB) would not fit.
  - Sol's KV formula gives (8C + 40·1536) × 4,096 B at f16. That is 496 MiB at 8K and 1,264 MiB at 32K, so most of the 2.4 GB would be compute buffers and other allocations.
  - Only a measurement settles it.
- **Speed.** Stock Metal rejects only NVFP4, TQ1_0 and F8_E4M3 for mul_mat (Sol, and the census agent, from source). The NVFP4 Metal PR is open upstream (#20456). Unsloth's fork has it (#331). Our speed edge over UD on stock is therefore real but not durable, and it is unmeasured against UD-Q2_K_XL (11 NVFP4 down tensors, 27 down visits per token).
- **Correctness.** Upstream 690b87d (#32704, 21 Nov) fixes the pass-2/3 scales for UD files. The sibling's upstream clone (c66103334) contains it, so stock mainline should run UD correctly. Our edge over UD on stock is speed, not correctness.
- **Competitors.** All-Metal files of the same weights at or under 9.14 GB:
  - AtomicChat AD-IQ2_M (8.67 GB) and AD-IQ2_XXS-Q2 (7.90 GB). We beat both on T1 and T2.
  - **abenzerps GSQ-RCO-IQ2_XS** (8,949,801,504 B, claims "16 GB Mac", unscored).
  - Smaller files not yet scored: UD-IQ1_S 7.76 (1 NVFP4 tensor), AD-IQ1_S 6.88 and AD-IQ1_M 7.26.
  - Published KLs use different references, so nobody has measured these files on a common reference.
- **Unique strength (the claim to aim at).** On one common reference, among stock GGUFs of this model that run fully on the GPU and hold a declared context inside a 16 GB Mac's default budget, ours is closest to the original. Speed and the gap to UD are both stated. Quality is measured on prose and on reasoning traces. The census found no file that makes this claim with comparable numbers.

## Steps (each ends in a check; GPU steps wait for windows the sibling grants)

1. **Freeze the workload and the comparators** (no GPU, now).
   - Workload: one text stream, P = 3, default macOS memory, 8K and 16K, f16 KV.
   - Comparators: every same-weights GGUF at or under the cap, found by header and hash, including the files under 8 GB and Unsloth's `Mainline_Support` copies.
   - Price the promotions on M0rk from tensor shapes. A Q2_0→IQ4_NL down costs 95.6 MB per layer. A gate_up step costs, per layer, +42.5 MB for IQ1_S→IQ2_XXS, +21.2 MB each for IQ2_XXS→IQ2_XS and IQ2_XS→IQ2_S, and +42.5 MB for IQ2_S→IQ3_XXS (computed from M0rk's tensor shapes).
   - Pre-register the gate and the stop rules in CHANGELOG.
2. **Measure what exists** (GPU, short).
   - Peak Metal memory of M0rk at 8K, 16K and 32K.
   - llama-bench: M0rk vs UD-IQ2_XXS and UD-Q2_K_XL, on mainline with 690b87d and the same settings.
   - Score UD-Q2_K_XL and GSQ-RCO-IQ2_XS on S, T1 and T2. That is two downloads of about 9 GB each.
   - Stop and decide: if M0rk beats every all-Metal comparator at or under 8.49 GB, it can go to T3 and release checks now (Daniel). If GSQ-RCO beats M0rk, the 9 GB tier must beat GSQ-RCO.
3. **Equal-byte promotions on M0rk** (GPU, S as development).
   - The headroom up to the measured cap is about 0.55 GB at 9.05 GB.
   - Two competing spends: gate_up steps versus down promotions. Locations are nominated in B3, B4 and B5.
   - Rank by S KL gain per byte on M0rk itself, then verify the chosen set together.
   - Stop after two promotions without a CI below 0.
4. **Per-block gain sweep** (GPU, byte edits by `degain2.py`).
   - Gain targets 0.97, 1.00 and 1.03 relative to the restore rule, for looped (6-17) versus non-looped layers, on the step-3 winner.
   - Stop if no CI is below 0. This is also the loop-specific finding for the sibling's H1.
5. **Refiner pilot** (CPU, then GPU).
   - Scope: one looped IQ2_S layer and one tail layer on the step-3 winner.
   - Diagonal shrinkage sweep: λ in {0, 0.1, 0.3, 0.6, 1}.
   - Prose-only versus mixed Grams (prose, chat, reasoning traces) at equal tokens, logging the lower tail of tokens per expert.
   - Decide on T2 (development use): continue only if T2 keeps at least half of the S gain.
   - If it passes, extend the refiner to IQ2_XXS and IQ2_XS, which Sol specified as 16,384 and 32,768 pair scores per 8-weight group.
6. **Freeze and test.**
   - T3 sealed: new prose documents and new reasoning problems, bootstrapped by document.
   - The 7 stability checks, a score on unpatched upstream, and speed at depth.
   - A load test on LM Studio, if available. Ollama's GGUF path does not support `inkling2_loop` yet (Sol 1.6).

## Decisions for Daniel

- The workload, 8K or 16K, and so the size cap after step 2.
- Whether M0rk ships on the Mac track if step 2 shows it beats every all-Metal comparator at its size.
- The release bar. All reviewers say an own-encoded release does not need to beat UD on KL. It does need KL wins over every eligible all-Metal file, a measured speed edge over UD on stock, and the gap to UD stated.
- GPU windows shared with the sibling (steps 2-5), and access to a real 16 GB Mac for the fit claim (otherwise the fit is a projection).
- An all-Metal repack of UD stays excluded as a derivative: it beats us by about 0.014, but it is not ours.
