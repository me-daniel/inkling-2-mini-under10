# Build environment

Written by `scripts/build-llama.sh` on 2026-11-21 23:30 UTC. Do not edit by hand.

- llama.cpp base: `29b52d398b294509746ec0a3f28948a3107c0213` (b12600)
- Patched HEAD: `fcaddf97a71d2cb49af9309398c726073e972e87`
- Patches (sha256, first 16):
  - `0001-model-inkling2_loop-re-share-the-loop-slots-after-th.patch` `2bc69d3fbc9920c8`
  - `0002-tests-check-that-inkling2_loop-passes-share-the-expe.patch` `5f507286c2941f7c`
  - `0003-model-inkling2_loop-cache-slots-can-carry-their-own-.patch` `02172c222b6bf257`
  - `0004-tests-check-that-inkling2_loop-slots-use-their-own-r.patch` `7c5f68647812c617`
  - `0005-model-inkling2_loop-find-slot-experts-in-every-split.patch` `fea5c969f6e25108`
  - `0006-tests-cover-pass-3-and-single-projections-in-the-slo.patch` `fbf6185bd57c5797`
  - `0007-convert-inkling2-decode-each-packed-expert-tensor-on.patch` `a7e047af2742be7c`
  - `0008-imatrix-optional-full-per-expert-input-Gram-capture-.patch` `858f36d82bca61ef`
- Build: Release, GGML_METAL=ON, cmake version 4.5.1, Apple clang version 21.0.0 (clang-2100.1.1.101)
- Machine: Apple M5 Pro, 48 GB, macOS 26.5, iogpu.wired_limit_mb=40960
- Loop tests (`test-llama-archs --loop-scales -s 42`, `--loop-slots -s 42`):
  - loop-scales: fold weights/2 scale*2 on blk.1: nmse 0.000e+00 (OK)
  - loop-scales: scale*2 only on blk.1:          nmse 7.000e-09 (OK)
  - loop-scales: scale*2 only on slot blk.3:     nmse 0.000e+00 (OK)
  - loop-slots: slots 3, 4 = physical values       nmse 0.000e+00 (OK)
  - loop-slots: slot 3 gate_up only = physical     nmse 0.000e+00 (OK)
  - loop-slots: slot 3 (pass 2) own values         nmse 6.163e-10 (OK)
  - loop-slots: slot 4 (pass 3) own values         nmse 2.115e-09 (OK)
  - loop-slots: pass 2 and pass 3 slots differ     nmse 2.573e-09 (OK)
  - loop-slots: slot 3 own gate_up only            nmse 1.206e-09 (OK)
  - loop-slots: slot 3 own down only               nmse 6.565e-10 (OK)
  - loop-slots: gate_up-only and down-only differ  nmse 1.472e-09 (OK)
  - loop-slots: slot 4 scale*2                     nmse 2.158e-09 (OK)
