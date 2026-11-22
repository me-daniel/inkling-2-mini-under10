"""Relative output error of several down encodings on the held-out eval Gram, split into the part carried by the
mean input (bias) and the rest. err = sum_e tr(dW G dW^T) / sum_e tr(W G W^T), G uncentered;
bias = sum_e c_e ||dW m_e||^2 / same denominator, m_e = in_sum / count from the eval capture's imatrix.
    python scripts/bias_split.py LAYER   (encodings: diag, GPTQ, UD-IQ2_XXS's own tensor with its .scale)"""
import sys, pathlib
import numpy as np, torch, gguf
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gptq_q2 as g
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize
ROOT = pathlib.Path(__file__).resolve().parents[1]
L = int(sys.argv[1]); name = f"blk.{L}.ffn_down_exps.weight"
src = {t.name: t for t in gguf.GGUFReader("/Users/daniel/GitHub/inkling-2-mini-quant/models/Inkling-2-Mini-BF16.gguf").tensors}[name]
w = torch.from_numpy((np.asarray(src.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32)).double()
E, R, C = w.shape
EV = pathlib.Path(__import__("os").environ.get("EVAL_DIR", ROOT / "models/grams/eval"))
G = torch.from_numpy(g.load_gram(EV, name, E, C)[0]).double()
im = {t.name: t for t in gguf.GGUFReader(EV / "imatrix.gguf").tensors}
cnt = torch.from_numpy(np.asarray(im[name + ".counts"].data, dtype=np.float64).reshape(-1))
m = torch.from_numpy(np.asarray(im[name + ".in_sum"].data, dtype=np.float64).reshape(E, C)) / cnt.clamp(min=1)[:, None]
den = torch.einsum("erc,ecd,erd->", w, G, w)
def dq(path, scale=None):
    t = {x.name: x for x in gguf.GGUFReader(path).tensors}
    d = torch.from_numpy(dequantize(np.asarray(t[name].data), t[name].tensor_type).reshape(E, R, C).astype(np.float64))
    if scale:
        d = d * torch.from_numpy(np.asarray(t[name.replace(".weight", ".scale")].data, dtype=np.float64)).view(E, 1, 1)
    return d, t[name].tensor_type.name
print("layer\tencoding\ttype\terr\tbias\trest\tbias_share")
for label, (wq, ty) in {"diag (ours)": dq(ROOT / f"models/down/diag-L{L}.gguf"), "GPTQ (ours)": dq(ROOT / f"models/down/gptq-L{L}.gguf"),
                        "UD-IQ2_XXS": dq(ROOT / "models/ext/mainline/UD-IQ2_XXS.gguf", scale=True)}.items():
    dw = wq - w
    err = torch.einsum("erc,ecd,erd->", dw, G, dw) / den
    bias = (cnt[:, None] * (torch.einsum("erc,ec->er", dw, m) ** 2)).sum() / den
    print(f"{L}\t{label}\t{ty}\t{err:.5f}\t{bias:.5f}\t{err - bias:.5f}\t{bias / err:.3f}", flush=True)
