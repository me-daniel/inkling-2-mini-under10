"""Offline bound for GPTQ on the down projections: relative output error of each encoding on a held-out Gram.

    python scripts/gptq_bound.py BF16.gguf IQ4_NL.gguf FIT_DIR EVAL_DIR --layers 0 12 [--damp 0.01 ...]

err = sum_e tr(dW_e G_e dW_e^T) / sum_e tr(W_e G_e W_e^T), with G_e the unnormalised Gram of expert e, so experts
count by how often they were routed. Prints one TSV row per layer, encoding and Gram set (fit = in sample).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gptq_q2 as g  # noqa: E402


def rel_err(w, wq, gram):
    dw = (wq - w).double()
    w = w.double()
    gram = gram.double()
    return (torch.einsum("erc,ecd,erd->", dw, gram, dw) / torch.einsum("erc,ecd,erd->", w, gram, w)).item()


def main() -> None:
    import gguf
    from gguf import GGMLQuantizationType as Q
    from gguf.quants import dequantize

    ap = argparse.ArgumentParser()
    ap.add_argument("bf16")
    ap.add_argument("iq4nl")
    ap.add_argument("fit", type=Path)
    ap.add_argument("eval", type=Path)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--damp", type=float, nargs="+", default=[0.01])
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    src = {t.name: t for t in gguf.GGUFReader(args.bf16).tensors}
    ref = {t.name: t for t in gguf.GGUFReader(args.iq4nl).tensors}
    print("layer\tencoding\tbpw\terr_fit\terr_eval")
    for layer in args.layers:
        name = f"blk.{layer}.ffn_down_exps.weight"
        raw = np.asarray(src[name].data)
        w = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        E, R, C = w.shape
        gf, nf = g.load_gram(args.fit, name, E, C)
        ge, _ = g.load_gram(args.eval, name, E, C)
        wt = torch.from_numpy(w)
        gft, get = torch.from_numpy(gf), torch.from_numpy(ge)
        rows = {}
        amax = wt.view(E, R, -1, 64).abs().amax(3, keepdim=True)
        d16 = amax.to(torch.float16).to(torch.float32)
        rows["Q2_0 stock"] = (2.25, (torch.clamp(torch.round(wt.view(E, R, -1, 64) / torch.where(amax == 0, 1, amax)), -1, 2) * d16).view(E, R, C))
        diag = torch.diag_embed(torch.diagonal(gft, dim1=1, dim2=2))
        d, c = g.gptq_q2(wt.to(dev), diag, 0.0)
        rows["Q2_0 diag-weighted (q2opt)"] = (2.25, g.dequant(d, c).cpu())
        for damp in args.damp:
            d, c = g.gptq_q2(wt.to(dev), gft, damp)
            rows[f"Q2_0 GPTQ damp {damp}"] = (2.25, g.dequant(d, c).cpu())
        t4 = ref[name]
        rows["IQ4_NL (llama-quantize, imatrix)"] = (4.5, torch.from_numpy(dequantize(np.asarray(t4.data), t4.tensor_type).reshape(E, R, C).astype(np.float32)))
        for k, (bpw, wq) in rows.items():
            print(f"{layer}\t{k}\t{bpw}\t{rel_err(wt, wq, gft):.5f}\t{rel_err(wt, wq, get):.5f}", flush=True)
        print(f"# layer {layer}: fit rows per expert median {int(np.median(nf))}, min {int(nf.min())}", file=sys.stderr)


if __name__ == "__main__":
    main()
