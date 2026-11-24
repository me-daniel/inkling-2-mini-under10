# Inkling 2 Mini under 10 GB, for 16 GB Macs

Code, logs and results behind `Inkling-2-Mini-Metal-8.9G.gguf` in [Me-Daniel/Inkling-2-Mini-Metal-GGUF](https://huggingface.co/Me-Daniel/Inkling-2-Mini-Metal-GGUF). This repo follows on from [inkling-2-mini-quant](https://github.com/me-daniel/inkling-2-mini-quant), whose 11.5 GB files need more GPU memory than a 16 GB Mac gives by default.

## The file

- 8,932,121,888 bytes, sha256 `a01a915627faad63ffef439fbdaf12cce189dcb864952c00150e073d859bbca9`.
- Every tensor is in a type that llama.cpp can multiply on Metal. The expert tensors are our own encodings; nothing was retrained.
- Fits the 11,453 MB default GPU budget of a 16 GB Mac at 8K context. 16K is projected from measurements on a 48 GB Mac (45 MB spare) and untested on 16 GB hardware.
- Use it only if the 11.5 GB files don't fit. On the sealed T3 set its KL against the Q8_0 reference is 0.0703, against 0.0285 for `Inkling-2-Mini-Metal.gguf`.

## Results

KL divergence of the next-token distribution against the Q8_0 reference at 3 loop passes; lower is closer. T3 (16 x 4,096 WikiText-2 test tokens) was sealed until the file was frozen and the pass rules were written down, then opened once.

| File | GB | KL T3 |
|---|---|---|
| Inkling-2-Mini-Metal-8.9G.gguf | 8.93 | 0.0703 |
| GSQ-RCO IQ2_XS | 8.95 | 0.0937 |
| AtomicChat AD-IQ2_M | 8.67 | 0.0992 |
| AtomicChat AD-IQ2_XXS-Q2 | 7.90 | 0.1239 |
| Unsloth UD-IQ2_XXS | 8.49 | 0.0691 |

- Among the all-Metal files we tested at or under 9 GB, it is the closest to the original on T1, T2 and T3. The pre-registered T3 tests pass against all five (`results/t3/compare.txt`).
- Unsloth's UD-IQ2_XXS is closer per byte. On T3 the difference is not resolved; on reasoning traces (T2) and in German and maths, UD is better.
- Speed, llama.cpp as of 21 Nov 2026 (`fcaddf97a`), one M5 Pro, same session: this file 249 / 21.3 t/s (pp512 / tg128), UD-IQ2_XXS 149 / 14.1, because that build runs UD's 7 NVFP4 expert tensors off the GPU. Upstream added NVFP4 Metal kernels on 23 Nov (#32825); with them the two files run at the same speed (`results/master/`).

## Where things are

- `CHANGELOG.md`: every registration, result and correction in order, written before the data it refers to where it says so. Start here.
- `drafts/MODEL_CARD.md`: the full write-up generated from `results/` by `scripts/card.py`, for both candidates. The 8.49 GB candidate (M0rk) was not published.
- `scripts/`: the build and scoring pipeline (`lib.sh` holds the shared scoring and lock helpers; `meanfit.py` is the mean-aware scale refit).
- `results/`: generated tables; `results/kl.tsv` has per-chunk KL for every run.
- `reviews/`: second opinions from other models, each answered point by point.

The scripts expect llama.cpp in `vendor/llama.cpp` (`scripts/build-llama.sh`) and the sibling repo with its models and reference logits next to this one. Model files, logs and data are not in git.

## Limits

- KL on prose and reasoning traces plus a saturated smoke test; no task benchmark with headroom.
- Per-chunk KL is rebuilt from running means printed with 5 decimals. Every verdict keeps its sign under worst-case rounding (`results/rounding-bound.txt`).
- One Mac, one llama.cpp build for the tables. Stock upstream `c661033` scores the file identically on S.
