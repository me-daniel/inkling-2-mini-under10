### For 16 GB Macs: Inkling-2-Mini-Metal-8.9G.gguf

- The files above need more than the 11.45 GB a 16 GB Mac gives the GPU by default. This one is 8.93 GB and leaves room for 8K context (16K is projected, untested on 16 GB hardware).
- Use it only if the files above don't fit: its KL on the held-out T3 set is 0.0703, against 0.0285 for Inkling-2-Mini-Metal.gguf.
- It is the closest to the original among the all-Metal files we tested at or under 9 GB. Unsloth's UD-IQ2_XXS is closer per byte; with llama.cpp as of 21 Nov 2026 it runs 7 expert tensors off the GPU, and this file reads prompts 1.67x and generates 1.51x as fast. Newer llama.cpp versions run those tensors on Metal, and the difference does not hold there.
- Own encoding with a different recipe from the files above. Method, tests and numbers: [inkling-2-mini-under10](https://github.com/me-daniel/inkling-2-mini-under10).
- sha256 `a01a915627faad63ffef439fbdaf12cce189dcb864952c00150e073d859bbca9`

```
llama-server -m Inkling-2-Mini-Metal-8.9G.gguf -ngl 99 --jinja -c 8192 --port 8080
```
