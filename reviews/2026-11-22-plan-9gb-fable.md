# Review: plan (9 GB concept draft), Fable

- Date: 2026-11-22
- Model: claude-fable-5-2 (Agent, read-only)
- Input: scratchpad brief-9gb.md (now docs/draft-9gb-brief.md), this repo's PLAN.md, CHANGELOG.md, docs/concept-v2.md, results/night1.md; sibling docs/report.md, docs/concept-v3.md, CLAUDE.md

## Reply (summary of the numbered points; the full report is in the session transcript)

1. Fit: PLAN.md gives 11,453 MB default Metal on 16 GB and 2.4 GB above weights at 16K (measured on the 11.5 GB file). 9.14 + 2.4 = 11.54 GB, over; implied 16K cap about 9.05 GB. Test: peak memory of M0rk at 8K/16K/32K.
2. Speed vs UD-Q2_K_XL unmeasured (sibling's 1.75x is vs 20 NVFP4 tensors; Q2_K_XL has 11). Test: llama-bench M0rk, UD-IQ2_XXS, UD-Q2_K_XL.
3. Stock correctness of UD files unchecked (patch 0001). Test: score UD-Q2_K_XL on unpatched current build.
4. M0rk already beats AD-IQ2_M on T1/T2 KL; fails only T2 top-1 (91.44 vs 91.48).
5. All-Metal is not ours: UD with NVFP4 -> IQ4_NL costs +0.00063 (X2), so a repack of UD beats us by about 0.014.
6. Step 4 has no spare bytes at UD's map (byte-neutral swap). Compute M1 size first.
7. UD's type map may be wrong for our encoder; mix-early lost +0.00515. One byte-neutral swap.
8. Mixed Grams unproven; only 5 of 11 IQ2_S layers are looped. Three-layer pilot on T2 first.
9. Transfer: patch-0001 check, the pure-attenuation instrument (per-block gain targets, looped vs not), v1's release frame (7 stability checks, unpatched-upstream score, speed at depth, peak memory). Per-pass slots are not stock. Block KLs do not add.
10. Niche: speed is not durable (NVFP4 Metal in Unsloth's fork, upstream PR pending). Defensible point: the largest own file that holds 16K in the default budget.
11. Refined plan: census, three GPU measurements, M1/M1rk, per-block gain sweep, one byte-neutral swap, 3-layer refiner pilot, freeze and T1/T2/T3. Owner decisions: size cap, M0rk now, release bar, 16 GB Mac access, UD repack exclusion.

## Response

- 1 accepted, as an open conflict: Sol's KV formula gives about 0.5 GiB of KV at 8K, far below 2.4 GB. Measure (concept-v3 step 2).
- 2 accepted (step 2).
- 3 resolved without a run: upstream 690b87d (#32704, 21 Nov) shares loop slots after the scale pass, and the upstream clone c66103334 contains it. Stock mainline is expected to run UD's scales correctly; scoring on that build stays in step 2.
- 4 accepted: M0rk's Mac-track status depends on the full comparator list (step 1-2).
- 5 accepted: the claim cannot rest on all-Metal alone.
- 6, 7 accepted: promotions on M0rk ranked by KL per byte replace the M1 ladder.
- 8 accepted (step 5).
- 9 accepted: gain instrument (step 4), release frame (step 6).
- 10 accepted in part: the claim is fit at a declared context plus common-reference quality; the cap follows the measurement.
