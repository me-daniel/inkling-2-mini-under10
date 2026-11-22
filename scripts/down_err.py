"""Relative output error of Q2_0 down encodings on several held-out Grams (no bias split), layers given.
    python scripts/down_err.py LAYER [LAYER ...]   -> rows: layer, gram, encoding, err"""
import sys, pathlib
import numpy as np, gguf
from gguf.quants import dequantize
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gptq_q2 as g
ROOT = pathlib.Path(__file__).resolve().parents[1]; SM = pathlib.Path("/Users/daniel/GitHub/inkling-2-mini-quant/models")
GRAMS = {"S-text": ROOT / "models/grams/s-x2", "wikitext-train": ROOT / "models/grams/wt-x2", "conversations": ROOT / "models/grams/eval"}
src = {t.name: t for t in gguf.GGUFReader(SM / "Inkling-2-Mini-BF16.gguf").tensors}
print("layer\tgram\tencoding\terr")
for L in map(int, sys.argv[1:]):
    n = f"blk.{L}.ffn_down_exps.weight"
    W = (np.asarray(src[n].data).view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    E, R, C = W.shape
    enc = {}
    def load(label, path, scaled=False):
        t = {x.name: x for x in gguf.GGUFReader(path).tensors}
        q = dequantize(np.asarray(t[n].data), t[n].tensor_type).reshape(E, R, C).astype(np.float64)
        if scaled: q = q * np.asarray(t[n.replace(".weight", ".scale")].data, dtype=np.float64)[:, None, None]
        enc[label] = q - W
    load("q2opt (imatrix-weighted)", SM / f"q2opt/L{L}.gguf")
    load("UD-IQ2_XXS Q2_0", ROOT / "models/ext/mainline/UD-IQ2_XXS.gguf", scaled=True)
    if (ROOT / f"models/down/g2p-L{L}.gguf").exists(): load("GPTQ v2 (alpha 1)", ROOT / f"models/down/g2p-L{L}.gguf")
    for gname, gd in GRAMS.items():
        G, _ = g.load_gram(gd, n, E, C)
        den = sum(np.sum((W[e] @ G[e]) * W[e]) for e in range(E))
        for label, dw in enc.items():
            num = sum(np.sum((dw[e] @ G[e]) * dw[e]) for e in range(E))
            print(f"{L}\t{gname}\t{label}\t{num / den:.5f}", flush=True)
