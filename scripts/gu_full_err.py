"""Full-Gram output error of gate_up encodings, per expert (memory ~1 GB): err = tr(dW G dW^T) / tr(W G W^T),
bias = n ||dW m||^2 / same denominator, using the sibling's Grams (read-only) and their imatrix (in_sum / counts).
    python scripts/gu_full_err.py LAYER GRAM_DIR label=FILE[:scaled] ...   ->  TSV rows; 'wins' = experts better than the first file"""
import sys, pathlib
import numpy as np, gguf
from gguf.quants import dequantize
L, GD = int(sys.argv[1]), pathlib.Path(sys.argv[2]); n = f"blk.{L}.ffn_gate_up_exps.weight"
E, C = 128, 2304
g = np.memmap(GD / f"{n}.gram.f32", dtype=np.float32, mode="r").reshape(E, C, C)
im = {t.name: t for t in gguf.GGUFReader(GD / "imatrix.gguf").tensors}
cnt = np.asarray(im[n + ".counts"].data, dtype=np.float64).reshape(-1)
mean = np.asarray(im[n + ".in_sum"].data, dtype=np.float64).reshape(E, C) / np.maximum(cnt, 1)[:, None]
src = {t.name: t for t in gguf.GGUFReader(pathlib.Path(__file__).resolve().parents[2] / "inkling-2-mini-quant/models/Inkling-2-Mini-BF16.gguf").tensors}[n]
W = (np.asarray(src.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
enc = []
for arg in sys.argv[3:]:
    label, _, f = arg.partition("="); scaled = f.endswith(":scaled"); f = f.removesuffix(":scaled")
    t = {x.name: x for x in gguf.GGUFReader(f).tensors}
    sc = np.asarray(t[n.replace(".weight", ".scale")].data, dtype=np.float64) if scaled else None
    enc.append((label, t[n], sc))
num = np.zeros((len(enc), E)); bias = np.zeros((len(enc), E)); den = np.zeros(E)
for e in range(E):
    G = np.triu(g[e]).astype(np.float64); G = G + np.triu(G, 1).T
    w = W[e].astype(np.float64); den[e] = np.sum((w @ G) * w)  # BLAS, not a 3-operand einsum
    for k, (label, t, sc) in enumerate(enc):
        q = dequantize(np.asarray(t.data[e]), t.tensor_type).reshape(w.shape).astype(np.float64)
        if sc is not None: q = q * sc[e]
        dw = q - w
        num[k, e] = np.sum((dw @ G) * dw); bias[k, e] = cnt[e] * np.sum((dw @ mean[e]) ** 2)
print("layer\tencoding\terr\tbias\twins_vs_first")
for k, (label, t, _) in enumerate(enc):
    print(f"{L}\t{label} ({t.tensor_type.name})\t{num[k].sum() / den.sum():.5f}\t{bias[k].sum() / den.sum():.5f}\t{int((num[k] / den < num[0] / den).sum())}/128", flush=True)
