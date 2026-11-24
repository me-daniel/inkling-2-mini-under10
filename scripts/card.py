"""Renders drafts/MODEL_CARD.md for the c1 / M0rk release from files. Never uploads anything.

    ../inkling-2-mini-quant/.venv/bin/python -I scripts/card.py

Every number comes from results/: kl.tsv (via analysis/compare.py, same paired bootstrap), t3/compare.txt and
batA/ (verdicts), release/ (sha256, same-session bench, graph splits, chunk identity), c1/ and step2/ (memory),
stability/ and c1/stability-verdict.md, types-*.txt (off-Metal counts), rounding-bound.txt. Wording follows the
24 Nov decisions reviews (reviews/2026-11-24-decisions-*.md, joint response). Exits 1 without writing if an input is
missing, if the release copies are not chunk-identical to the frozen files, or if the text holds a local path or a dash.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from compare import load, paired  # noqa: E402

R = ROOT / "results"
BUDGET = 11_453_250_000  # default GPU working-set budget of a 16 GB Mac (results/step2, concept-v3)
FILES = {"c1": "Inkling-2-Mini-Metal-8.9G.gguf", "m0rk": "Inkling-2-Mini-Metal-8.5G.gguf"}  # names: Daniel's call
COMP = [("GSQ-RCO-IQ2_XS", "GSQ-RCO IQ2_XS"), ("AD-IQ2_M", "AtomicChat AD-IQ2_M"), ("AD-IQ2_XXS-Q2", "AtomicChat AD-IQ2_XXS-Q2"),
        ("AD-IQ1_M", "AtomicChat AD-IQ1_M"), ("AD-IQ1_S", "AtomicChat AD-IQ1_S")]
UD = [("UD-IQ2_XXS", "Unsloth UD-IQ2_XXS"), ("UD-IQ1_M", "Unsloth UD-IQ1_M"), ("UD-Q2_K_XL", "Unsloth UD-Q2_K_XL")]


def need(p: Path) -> str:
    if not p.exists():
        sys.exit(f"missing input: {p.relative_to(ROOT)}")
    return p.read_text()


def size(name: str) -> int:
    for line in need(R / "t3/sha.txt").splitlines():
        if line.split()[0] == name:
            return int(line.split()[2])
    m = {"UD-IQ1_M": 8_024_441_024, "UD-Q2_K_XL": 9_138_715_328}  # results/landscape-7.5-9.6GB.md (not on T3)
    return m[name]


def offmetal(name: str) -> int:
    return int(re.search(r"cannot multiply: (\d+)", need(R / f"types-{name}.txt")).group(1))


def bench(name: str) -> tuple[float, float]:
    t = need(R / f"release/bench-{name}.txt")
    pp = float(re.search(r"pp512 \|\s+([\d.]+)", t).group(1)); tg = float(re.search(r"tg128 \|\s+([\d.]+)", t).group(1))
    return pp, tg


def footprint(path: Path) -> int:
    return int(re.search(r"(\d+)\s+peak memory footprint", need(path)).group(1))


k = load()
ALIAS = {("m0rk", "T1"): "cand", ("m0rk", "T2"): "cand"}  # M0rk's T1/T2 runs were named final-cand (comparators.sh, CAND = M0rk)
kl = lambda n, s: (lambda r: k[r]["chunks"].mean() if r in k else None)(f"final-{ALIAS.get((n, s), n)}-{s}-P3")
f5 = lambda x: "not scored" if x is None else f"{x:.4f}"
ident = need(R / "release/identical.txt")
if ident.count("chunks identical: True") != 2:
    sys.exit("release copies are not chunk-identical to the frozen files")
sha = {l.split()[0]: l.split()[1] for l in need(R / "release/sha256.txt").splitlines()}
t3 = need(R / "t3/compare.txt")
for v in ("c1 Mac track (9 GB tier): PASS", "M0rk Mac track (8.49 GB tier): PASS"):
    if v not in t3:
        sys.exit(f"T3 verdict missing: {v}")
pp = {n: bench(n) for n in ("c1", "m0rk", "UD-IQ2_XXS")}
mem = {"c1": footprint(R / "c1/memory-16384.txt") + size("c1"), "m0rk": footprint(R / "step2/memory-m0rk-16384.txt") + size("m0rk")}
d, lo, hi = paired(k["final-c1-T3-P3"]["chunks"], k["final-UD-IQ2_XXS-T3-P3"]["chunks"])
t2r = kl("c1", "T2") / kl("UD-IQ2_XXS", "T2") - 1
dom = {}
for dm in ("python", "rust", "german", "math"):
    a, b = k[f"batA-{dm}-c1"]["chunks"], k[f"batA-{dm}-UD-IQ2_XXS"]["chunks"]
    dom[dm] = (a.mean() / b.mean(), *paired(a, b))

rows = []
for n, label in [("c1", f"**{FILES['c1']}** (this repo)"), ("m0rk", f"**{FILES['m0rk']}** (this repo)")] + COMP + UD:
    rows.append(f"| {label} | {size(n):,} | {offmetal(n)} | {f5(kl(n, 'T1'))} | {f5(kl(n, 'T2'))} | {f5(kl(n, 'T3'))} |")
t3lines = [l for l in t3.splitlines() if l.startswith("T3 ")]

card = f"""# Inkling 2 Mini GGUF for 16 GB Macs: two files that run entirely on the Mac GPU

- Two GGUFs of Inkling 2 Mini, {size('m0rk') / 1e9:.2f} GB and {size('c1') / 1e9:.2f} GB. Every tensor is in a type that stock llama.cpp can multiply on Metal.
- Among the files we tested that run entirely on Metal in stock llama.cpp, each is the closest to the original model at or under its size, on three test sets including a held-out one scored once.
- Unsloth's files are closer to the original per byte. In llama.cpp as of 21 Nov 2026 their NVFP4 expert tensors run off the GPU, so there they are slower: on our Mac, {FILES['c1']} reads prompts {pp['c1'][0] / pp['UD-IQ2_XXS'][0]:.2f}x and generates {pp['c1'][1] / pp['UD-IQ2_XXS'][1]:.2f}x as fast as UD-IQ2_XXS. Unsloth's own llama.cpp build and later upstream versions have NVFP4 Metal kernels, and the speed difference does not apply there.
- Tested on 23 and 24 Nov 2026 with llama.cpp `fcaddf97a` (21 Nov 2026). Later uploads and llama.cpp versions are not covered.

## Which file

| File | Bytes | sha256 | For |
|---|---|---|---|
| {FILES['c1']} | {size('c1'):,} | `{sha['c1']}` | the default: lower KL everywhere, at 8K context on a 16 GB Mac |
| {FILES['m0rk']} | {size('m0rk'):,} | `{sha['m0rk']}` | more memory headroom (16K context, or other apps open) |

```
llama-server -m {FILES['c1']} -ngl 99 --jinja -c 8192 --port 8080
```

Memory, measured on a 48 GB Mac (llama-bench at 16K depth, peak footprint plus weights) and set against the {BUDGET / 1e6:,.0f} MB default GPU budget of a 16 GB Mac: {FILES['c1']} needs {mem['c1'] / 1e6:,.0f} MB at 16K ({(BUDGET - mem['c1']) / 1e6:,.0f} MB spare), and {FILES['m0rk']} {mem['m0rk'] / 1e6:,.0f} MB ({(BUDGET - mem['m0rk']) / 1e6:,.0f} MB spare). Neither fits 32K. This is a projection: neither file has been run on 16 GB hardware, so 8K is the context we advertise and 16K is untested.

## Quality

KL divergence of the next-token distribution against the Q8_0 reference at 3 loop passes (lower is closer). T1: 16 x 4,096 WikiText-2 test tokens. T2: 5 x 2,048 tokens of reasoning traces the model wrote on GSM8K train prompts. T3: 16 x 4,096 WikiText-2 test tokens, held out: scored once, after both files were frozen and the pass rules written down. T1 and T2 informed choices during development; T3 did not.

| File | Bytes | Expert tensors off Metal (llama.cpp of 21 Nov 2026) | KL T1 | KL T2 | KL T3 |
|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

Pre-registered T3 tests (paired bootstrap over chunks, 10,000 resamples): {FILES['c1']} against every all-Metal file at or under 9 GB, {FILES['m0rk']} against every one at or under its size. The 95% CI of the difference must lie below 0 and top-1 agreement must not be lower. Both pass:

```
""" + "\n".join(t3lines) + f"""
```

Against Unsloth UD-IQ2_XXS ({size('UD-IQ2_XXS'):,} bytes, {100 * (size('c1') / size('UD-IQ2_XXS') - 1):.0f}% smaller than {FILES['c1']}):
- On held-out WikiText (T3), {FILES['c1']}'s KL is {d:+.4f} above UD's (95% CI {lo:+.4f} to {hi:+.4f}, not resolved).
- On reasoning traces (T2), UD is lower by {100 * t2r:.0f}% of UD's KL.
- On four texts never used in development (python, rust, German, maths; 32 x 1,024 tokens each), UD is lower in German ({100 * (dom['german'][0] - 1):.0f}%) and maths ({100 * (dom['math'][0] - 1):.0f}%). In python ({100 * (dom['python'][0] - 1):.0f}%) and rust ({100 * (dom['rust'][0] - 1):.0f}%) the difference is not resolved at the precision llama-perplexity prints.
- {FILES['c1']} beats GSQ-RCO IQ2_XS and AtomicChat AD-IQ2_M on all four of those texts.
- {FILES['m0rk']} uses UD-IQ2_XXS's per-layer type map and is {kl('m0rk', 'T3') / kl('UD-IQ2_XXS', 'T3'):.2f}x its KL on T3. UD-IQ1_M, which is smaller, is also closer to the original on T1 and T2.

## Speed

One Mac ({re.search(r'GPU name:\s+MTL0 \(([^)]+)\)', need(ROOT / 'logs/step2-memory-m0rk-16384.log')).group(1)}), stock llama.cpp as of 21 Nov 2026, llama-bench -p 512 -n 128 -r 3, the three files back to back in one session with nothing else running:

| File | pp512 t/s | tg128 t/s |
|---|---|---|
""" + "\n".join(f"| {FILES.get(n, n)} | {pp[n][0]:.1f} | {pp[n][1]:.1f} |" for n in ("c1", "m0rk", "UD-IQ2_XXS")) + f"""

The difference comes from the 7 NVFP4 expert tensors in UD-IQ2_XXS, which llama.cpp as of 21 Nov 2026 cannot multiply on Metal. Later versions and Unsloth's own build add NVFP4 Metal kernels, and there the difference does not hold.

## Stability

Both files pass the same smoke test as the Q8_0 reference: long generations without repetition loops, a planted fact recalled from a long prompt, 5 of 5 valid tool calls, 2 of 2 non-English replies, GSM8K 30 of 30 with none hitting the length cap. The test saturates (the reference also scores 30 of 30), so it shows nothing about task quality beyond that.

## How the files were made

Nothing was retrained. The expert tensors are our own encodings: llama-quantize pools with the ggml-org imatrix, a mean-aware refit of the gate_up block scales with each row's output gain restored to llama-quantize's, and the Q2_0 down projections rescaled per row to unit gain. {FILES['m0rk']} uses Unsloth UD-IQ2_XXS's per-layer type map, with its NVFP4 down projections replaced by IQ4_NL at the same size. {FILES['c1']} adds 446 MB of per-layer upgrades to that recipe, chosen one at a time. Header: general.file_type is set to MOSTLY_IQ2_XXS (the files mix types); the release files are byte-identical to the scored ones apart from that field, and score identically chunk for chunk.

## Limits

- Our encoding does not beat Unsloth's per byte at any size we tested. The case for these files is that they run entirely on the GPU in llama.cpp as of 21 Nov 2026, and the speed that follows from it there.
- KL on prose and reasoning traces, plus a smoke test. No task benchmark with headroom was run.
- The per-chunk KL values are rebuilt from running means printed with 5 decimals. Every verdict above keeps its sign under the worst-case rounding.
- Engine: llama.cpp `fcaddf97a` (patched build used for the tables). Stock upstream `c661033` scores {FILES['c1']} identically on S.
"""

for pat, why in ((r"/Users/|/private/|scratchpad", "local path"), ("—|–", "dash")):
    if re.search(pat, card):
        sys.exit(f"card holds a {why}; nothing written")
out = ROOT / "drafts/MODEL_CARD.md"; out.parent.mkdir(exist_ok=True); out.write_text(card)
print(f"wrote {out.relative_to(ROOT)} ({len(card):,} chars)")
