# Inkling 2 Mini GGUF for 16 GB Macs: two files that run entirely on the Mac GPU

- Two GGUFs of Inkling 2 Mini, 8.49 GB and 8.93 GB. Every tensor is in a type that stock llama.cpp can multiply on Metal.
- Among the files we tested that run entirely on Metal in stock llama.cpp, each is the closest to the original model at or under its size, on three test sets including a held-out one scored once.
- Unsloth's files are closer to the original per byte. In llama.cpp as of 21 Nov 2026 their NVFP4 expert tensors run off the GPU, so there they are slower: on our Mac, Inkling-2-Mini-Metal-8.9G.gguf reads prompts 1.67x and generates 1.51x as fast as UD-IQ2_XXS. Unsloth's own llama.cpp build and later upstream versions have NVFP4 Metal kernels, and the speed difference does not apply there.
- Tested on 23 and 24 Nov 2026 with llama.cpp `fcaddf97a` (21 Nov 2026). Later uploads and llama.cpp versions are not covered.

## Which file

| File | Bytes | sha256 | For |
|---|---|---|---|
| Inkling-2-Mini-Metal-8.9G.gguf | 8,932,121,888 | `a01a915627faad63ffef439fbdaf12cce189dcb864952c00150e073d859bbca9` | the default: lower KL everywhere, at 8K context on a 16 GB Mac |
| Inkling-2-Mini-Metal-8.5G.gguf | 8,486,214,944 | `0d24b04803c597a9362b24647d27931e750e3cd01ca2619a8a55252c7882c660` | more memory headroom (16K context, or other apps open) |

```
llama-server -m Inkling-2-Mini-Metal-8.9G.gguf -ngl 99 --jinja -c 8192 --port 8080
```

Memory, measured on a 48 GB Mac (llama-bench at 16K depth, peak footprint plus weights) and set against the 11,453 MB default GPU budget of a 16 GB Mac: Inkling-2-Mini-Metal-8.9G.gguf needs 11,408 MB at 16K (45 MB spare), and Inkling-2-Mini-Metal-8.5G.gguf 10,963 MB (490 MB spare). Neither fits 32K. This is a projection: neither file has been run on 16 GB hardware, so 8K is the context we advertise and 16K is untested.

## Quality

KL divergence of the next-token distribution against the Q8_0 reference at 3 loop passes (lower is closer). T1: 16 x 4,096 WikiText-2 test tokens. T2: 5 x 2,048 tokens of reasoning traces the model wrote on GSM8K train prompts. T3: 16 x 4,096 WikiText-2 test tokens, held out: scored once, after both files were frozen and the pass rules written down. T1 and T2 informed choices during development; T3 did not.

| File | Bytes | Expert tensors off Metal (llama.cpp of 21 Nov 2026) | KL T1 | KL T2 | KL T3 |
|---|---|---|---|---|---|
| **Inkling-2-Mini-Metal-8.9G.gguf** (this repo) | 8,932,121,888 | 0 | 0.0668 | 0.0489 | 0.0703 |
| **Inkling-2-Mini-Metal-8.5G.gguf** (this repo) | 8,486,214,944 | 0 | 0.0801 | 0.0593 | 0.0857 |
| GSQ-RCO IQ2_XS | 8,949,801,504 | 0 | 0.0890 | 0.0557 | 0.0937 |
| AtomicChat AD-IQ2_M | 8,665,091,680 | 0 | 0.0956 | 0.0676 | 0.0992 |
| AtomicChat AD-IQ2_XXS-Q2 | 7,900,679,776 | 0 | 0.1206 | 0.0803 | 0.1239 |
| AtomicChat AD-IQ1_M | 7,263,669,856 | 0 | 0.1757 | 0.1262 | 0.1797 |
| AtomicChat AD-IQ1_S | 6,881,463,904 | 0 | 0.2050 | 0.1474 | 0.2131 |
| Unsloth UD-IQ2_XXS | 8,486,237,632 | 7 | 0.0658 | 0.0427 | 0.0691 |
| Unsloth UD-IQ1_M | 8,024,441,024 | 3 | 0.0760 | 0.0431 | not scored |
| Unsloth UD-Q2_K_XL | 9,138,715,328 | 11 | 0.0510 | 0.0349 | not scored |

Pre-registered T3 tests (paired bootstrap over chunks, 10,000 resamples): Inkling-2-Mini-Metal-8.9G.gguf against every all-Metal file at or under 9 GB, Inkling-2-Mini-Metal-8.5G.gguf against every one at or under its size. The 95% CI of the difference must lie below 0 and top-1 agreement must not be lower. Both pass:

```
T3 c1 vs GSQ-RCO-IQ2_XS: KL 0.07034 vs 0.09366, diff -0.02332 [-0.02628, -0.02034], top-1 89.08% vs 87.74% -> better
T3 c1 vs AD-IQ2_M: KL 0.07034 vs 0.09917, diff -0.02883 [-0.03196, -0.02563], top-1 89.08% vs 87.56% -> better
T3 c1 vs AD-IQ2_XXS-Q2: KL 0.07034 vs 0.12389, diff -0.05355 [-0.05872, -0.04788], top-1 89.08% vs 85.92% -> better
T3 c1 vs AD-IQ1_M: KL 0.07034 vs 0.17972, diff -0.10938 [-0.11965, -0.09803], top-1 89.08% vs 83.34% -> better
T3 c1 vs AD-IQ1_S: KL 0.07034 vs 0.21305, diff -0.14271 [-0.15595, -0.12746], top-1 89.08% vs 82.10% -> better
T3 m0rk vs AD-IQ2_XXS-Q2: KL 0.08568 vs 0.12389, diff -0.03821 [-0.04187, -0.03431], top-1 88.19% vs 85.92% -> better
T3 m0rk vs AD-IQ1_M: KL 0.08568 vs 0.17972, diff -0.09404 [-0.10267, -0.08449], top-1 88.19% vs 83.34% -> better
T3 m0rk vs AD-IQ1_S: KL 0.08568 vs 0.21305, diff -0.12737 [-0.13905, -0.11392], top-1 88.19% vs 82.10% -> better
```

Against Unsloth UD-IQ2_XXS (8,486,237,632 bytes, 5% smaller than Inkling-2-Mini-Metal-8.9G.gguf):
- On held-out WikiText (T3), Inkling-2-Mini-Metal-8.9G.gguf's KL is +0.0013 above UD's (95% CI -0.0004 to +0.0029, not resolved).
- On reasoning traces (T2), UD is lower by 14% of UD's KL.
- On four texts never used in development (python, rust, German, maths; 32 x 1,024 tokens each), UD is lower in German (8%) and maths (13%). In python (6%) and rust (3%) the difference is not resolved at the precision llama-perplexity prints.
- Inkling-2-Mini-Metal-8.9G.gguf beats GSQ-RCO IQ2_XS and AtomicChat AD-IQ2_M on all four of those texts.
- Inkling-2-Mini-Metal-8.5G.gguf uses UD-IQ2_XXS's per-layer type map and is 1.24x its KL on T3. UD-IQ1_M, which is smaller, is also closer to the original on T1 and T2.

## Speed

One Mac (Apple M5 Pro), stock llama.cpp as of 21 Nov 2026, llama-bench -p 512 -n 128 -r 3, the three files back to back in one session with nothing else running:

| File | pp512 t/s | tg128 t/s |
|---|---|---|
| Inkling-2-Mini-Metal-8.9G.gguf | 249.1 | 21.3 |
| Inkling-2-Mini-Metal-8.5G.gguf | 269.3 | 21.9 |
| UD-IQ2_XXS | 148.8 | 14.1 |

The difference comes from the 7 NVFP4 expert tensors in UD-IQ2_XXS, which llama.cpp as of 21 Nov 2026 cannot multiply on Metal. Later versions and Unsloth's own build add NVFP4 Metal kernels, and there the difference does not hold.

## Stability

Both files pass the same smoke test as the Q8_0 reference: long generations without repetition loops, a planted fact recalled from a long prompt, 5 of 5 valid tool calls, 2 of 2 non-English replies, GSM8K 30 of 30 with none hitting the length cap. The test saturates (the reference also scores 30 of 30), so it shows nothing about task quality beyond that.

## How the files were made

Nothing was retrained. The expert tensors are our own encodings: llama-quantize pools with the ggml-org imatrix, a mean-aware refit of the gate_up block scales with each row's output gain restored to llama-quantize's, and the Q2_0 down projections rescaled per row to unit gain. Inkling-2-Mini-Metal-8.5G.gguf uses Unsloth UD-IQ2_XXS's per-layer type map, with its NVFP4 down projections replaced by IQ4_NL at the same size. Inkling-2-Mini-Metal-8.9G.gguf adds 446 MB of per-layer upgrades to that recipe, chosen one at a time. Header: general.file_type is set to MOSTLY_IQ2_XXS (the files mix types); the release files are byte-identical to the scored ones apart from that field, and score identically chunk for chunk.

## Limits

- Our encoding does not beat Unsloth's per byte at any size we tested. The case for these files is that they run entirely on the GPU in llama.cpp as of 21 Nov 2026, and the speed that follows from it there.
- KL on prose and reasoning traces, plus a smoke test. No task benchmark with headroom was run.
- The per-chunk KL values are rebuilt from running means printed with 5 decimals. Every verdict above keeps its sign under the worst-case rounding.
- Engine: llama.cpp `fcaddf97a` (patched build used for the tables). Stock upstream `c661033` scores Inkling-2-Mini-Metal-8.9G.gguf identically on S.
