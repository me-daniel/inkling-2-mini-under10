# Draft concept: an own-encoded ~9 GB GGUF of Inkling 2 Mini for a 16 GB Mac

## Model
- thinkingmachines/Inkling-2-Mini, 26.07 B params, looped MoE: 24 layers, layers 6-17 are the looped core, default P = 3 passes, so 48 layer visits. 128 routed experts per layer in two tensors: ffn_gate_up_exps (2304 inputs) and ffn_down_exps.
- Quality metric: mean KL to a Q8_0 reference at P = 3 (llama-perplexity --kl-divergence). S = selection set (WikiText-2 test text), T1 = held-out prose, T2 = held-out reasoning traces (GSM8K, model-generated). Differences are paired over chunks with bootstrap 95% CI.

## Landscape near 9 GB (HF file sizes)
- Unsloth: UD-IQ1_S 7.76 GB, UD-IQ1_M 8.02, UD-IQ2_XXS 8.49, UD-Q2_K_XL 9.14 (9,138,715,328 B), UD-IQ3_XXS 11.66.
- AtomicChat: AD-IQ2_XXS-Q2 7.90, AD-IQ2_M 8.67, AD-IQ2_XXS 10.43.
- bartowski: smallest is IQ2_XXS 10.32 GB. Many other repos not yet checked (APEX variants, GSQ-RCO, NInfer repacks of UD).
- UD-Q2_K_XL tensor types (from its header): gate_up IQ1_S in layers 0-4 and 6, IQ2_XXS in 5, 7, 12-14, IQ2_XS in 8 and 10, IQ2_S in 9, 11, 15-23. down: Q2_0 in 0-4, 6, 12-14, 21-23; NVFP4 in 5, 7-11, 15-19; MXFP4 in 20. Attention mostly Q8_0 / Q6_K, output Q4_K, embeddings IQ4_XS. Its imatrix: "imx_s0.conv.jsonl", 8238 chunks (conversation data).
- Stock llama.cpp's Metal backend cannot multiply NVFP4, so those tensors run on the CPU. The sibling measured UD-IQ3_XXS (20 NVFP4 expert tensors) at 68 t/s prompt and 10.9 t/s generation versus about 225 / 19 t/s for all-Metal files of similar size on an M5 Pro. Unsloth's own llama.cpp fork already runs NVFP4 on Metal.

## Measured so far (this repo and the sibling repo, same metric)
1. At UD-IQ2_XXS's exact per-tensor types, plain llama-quantize with an imatrix (M0) is +0.02727 KL behind UD on S. UD's edge by tensor class: gate_up 61%, non-expert 23%, down 15%. The edge is the encoder, not the type map or the imatrix (three imatrices differ by < 0.0004).
2. Attenuation: fits that lower per-layer MSE shrink each row's output gain <q,w>/<w,w> below 1, and over 48 visits this costs KL. A mean-aware gate_up scale refit lowered full-Gram error in 128/128 experts and still raised KL by +0.072 (gain 0.79). The same refit with each row restored to llama-quantize's own gain gained -0.0038. Rescaling our Q2_0 down from gain 0.86 to 1 gained -0.0070. Gain 1 is not universal: it hurt gate_up and UD's own Q2_0 down (gain 0.90).
3. Our best own file at 8.49 GB (M0rk: mean-aware gate_up refit + gain restore, gain-1 Q2_0 down, all tensors in Metal types) closes 40% of the gap to UD on S. Held out: M0rk minus UD-IQ2_XXS +0.01424 on T1, +0.01662 on T2; it beats both AtomicChat files on both sets (-0.016 to -0.041).
4. The sibling's full-Gram code refinement (changes grid indices and signs, keeps scales, objective tr(E H E^T) with H = Gram + 0.1 diag shrink, gain floor per row) for IQ2_S / IQ3_XXS gate_up at 11.6 GB: closes 93% of the gap to UD-IQ3_XXS on S, 74% on T1, 18% on T2. Its Grams are WikiText-2 train only (same source as S), about 5k tokens per median expert for 2304 dims. The refiner does not support IQ1_S, IQ2_XXS or IQ2_XS.
5. Sibling's block map (one block of experts quantized, rest at reference, S): looped blocks and the last block are the most sensitive at 2 bits; per-layer KL B1 (0-5) 0.0046, B2 (6-11) 0.0104, B3 (12-17) 0.0142, B4 (18-19) 0.0180, B5 (20-23) 0.0167.
6. The sibling's 11.5 GB file (v1) is public: it failed its gate against UD-IQ3_XXS and was released as the closest-to-original all-Metal file under 12 GB, 1.75x faster generation than UD-IQ3_XXS on stock llama.cpp.

## Proposed plan for the 9 GB tier
- Target: one own-encoded GGUF, every tensor in a type stock llama.cpp runs on Metal, size <= 9,138,715,328 B (UD-Q2_K_XL), so it fits a 16 GB Mac's default GPU memory with room for context.
- Step 0: landscape. List every GGUF of this model in 8.0-9.2 GB on HF with its types and off-Metal tensor count. Score UD-Q2_K_XL and any all-Metal comparator on S, T1, T2. Measure stock-llama.cpp speed of ours vs UD-Q2_K_XL.
- Step 1 (baseline M1): llama-quantize at UD-Q2_K_XL's per-tensor types, with each NVFP4 down replaced by IQ4_NL (same 4.5 bpw, on Metal). Check size and that MXFP4 runs on Metal.
- Step 2 (M1rk): apply the M0rk recipe (mean-aware gate_up refit with per-row gain restore; Q2_0 down rescaled to gain 1).
- Step 3 (M1rr): the sibling's full-Gram code refiner on the 11 IQ2_S gate_up layers, with Grams captured on mixed calibration (prose, chat, reasoning traces) and more tokens, then the gain restore.
- Step 4: spend any bytes left under the limit by the block map (looped and last blocks first).
- Gate: lower KL than every all-Metal file <= the limit on T1 and T2 (CI below 0), and report the difference to UD-Q2_K_XL whatever it is. A held-out T3 for the final file.
- Release bar (open, for the owner): only own encodings. Is "best all-Metal own encoding at this size + faster than UD on stock llama.cpp" enough, as for the sibling's v1, or must it beat UD-Q2_K_XL?

## Constraints
- One Apple M5 Pro Mac, 48 GB, ~30 GB per heavy run, shared with a sibling project that is using the GPU now. No cloud GPU. A full S score takes minutes; a Gram capture for a few layers takes about an hour; the sibling's refiner took hours for 24 layers.
- Stock llama.cpp types only (the file must run in stock llama.cpp, LM Studio, Ollama).
