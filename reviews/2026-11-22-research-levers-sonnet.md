# Research: encoder levers in stock GGUF types (Sonnet, web, abstracts mostly)

- Date: 2026-11-22
- Model: claude-sonnet-5-5 (Agent, web only)

## Findings (condensed; most sources read as abstracts only)

1. Domain robustness: DASH-Q (arXiv 2604.13806) diagonal-curvature, stable with tiny calibration sets; MoEQuant (2505.03804) expert-balanced self-sampling plus router-affinity Hessians; EAQuant (2506.13329); self-calibration (NAACL 2025). No source measured held-out-domain gains from pooled Grams or domain mixing.
2. Same-format refiners: MACaQ-R (huawei-csl/MACaQ) refines GGUF codes keeping formats and scales, reports KL 0.0134 vs Unsloth UD-Q4_K_XL 0.0157 on Qwen3.6-35B-A3B at equal size; K-quants only, no IQ types documented. GSQ (2604.18556) learned grid assignments, GPU-oriented. No merged llama.cpp PR improves IQ2_XS/IQ2_S/IQ1_S search.
3. Looped models: arXiv 2609.30820, Hessians accumulated across recurrence steps beat step-0 GPTQ and RTN on 9 of 9 checkpoints; Tilted Bowl (2609.39277), settling loops shift the fixed point; LoopQuant, LoopQ not stock.
4. CPU feasibility (guesses): multi-pass Gram plus diagonal shrinkage, hours; self-generated calibration, hours to a day; MACaQ-R on IQ, unlikely; GSQ on CPU, days.

## Response

- 3 already holds for us: llama-imatrix and our Gram patch sum all visits of a looped layer (sibling concept-v3, H2 note).
- 1 feeds the shrinkage sweep in concept-v3 step 5. Nothing found changes the plan's order.
