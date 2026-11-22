"""GPTQ rounding into the existing Q2_0 GGUF layout (block of 64, one FP16 scale d, levels d * {-1, 0, 1, 2}).

    python scripts/gptq_q2.py BF16.gguf OUT.gguf --layer L --grams DIR [DIR ...] [--mode gptq|diag] [--alpha 1.0]

For each routed expert, W [rows, cols] is rounded column by column along its input dimension. The error of each
column is pushed onto the not-yet-rounded columns through the Cholesky factor of H^-1 (Frantar et al., GPTQ), where
H is the expert's input Gram X^T X captured by llama-imatrix (patch 0008), pooled over every visit of the layer.
At the start of each 64-column block every row gets its scale from the weighted scale search of scripts/q2opt.py
(weights: diag H), run on the error-updated weights. Codes and scales are packed into block_q2_0 unchanged, so the
file runs on every backend that runs Q2_0. Experts with fewer than --min-rows routed rows fall back to diag-weighted
rounding (H replaced by its diagonal). --alpha shrinks H toward its diagonal (H + alpha * diag H) against
overfitting; --mode diag zeroes the off-diagonals (the matched control: same Gram, same scale search, no error
feedback). Writes a one-tensor donor GGUF for scripts/splice.py, then decodes it and checks it against the solver.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

QK = 64
FACTORS = [round(0.30 + 0.05 * i, 2) for i in range(15)]  # 0.30 .. 1.00, as in q2opt.py


def load_gram(gram_dir: Path, name: str, n_expert: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.fromfile(gram_dir / f"{name}.gram.f32", dtype=np.float32).reshape(n_expert, cols, cols)
    n = np.fromfile(gram_dir / f"{name}.gram.counts.i64", dtype=np.int64)
    up = np.triu(g)  # the capture fills the upper triangle only
    return up + np.transpose(np.triu(g, 1), (0, 2, 1)), n


def block_scales(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """x [E, R, 64] weights, h [E, 1, 64] channel weights -> FP16-representable scale d [E, R, 1] with the smallest
    h-weighted squared error over the candidates of q2opt.py (stock d = max|w|, and +-f * max|w| refit by WLS)."""
    amax = x.abs().amax(dim=2, keepdim=True)
    best_err = torch.full_like(amax, float("inf"))
    best_d = torch.zeros_like(amax)

    def consider(d: torch.Tensor, refit: bool) -> None:
        nonlocal best_err, best_d
        c = torch.clamp(torch.round(x / torch.where(d == 0, torch.ones_like(d), d)), -1, 2)
        if refit:
            den = (h * c * c).sum(2, keepdim=True)
            num = (h * x * c).sum(2, keepdim=True)
            d = torch.where(den > 0, num / torch.where(den > 0, den, torch.ones_like(den)), d)
        d = d.to(torch.float16).to(torch.float32)
        c = torch.clamp(torch.round(x / torch.where(d == 0, torch.ones_like(d), d)), -1, 2)
        err = (h * (x - d * c) ** 2).sum(2, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_d = torch.where(better, d, best_d)

    consider(amax, refit=False)
    for f in FACTORS:
        for sign in (1.0, -1.0):
            consider(sign * f * amax, refit=True)
    return best_d


def gptq_q2(w: torch.Tensor, gram: torch.Tensor, damp: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """w [E, R, C] float32, gram [E, C, C] (any positive scale) -> (d [E, R, C/64] float32 holding FP16 values,
    codes [E, R, C] int8 in -1..2). All on w.device."""
    E, R, C = w.shape
    w = w.clone()
    h = gram.to(torch.float64).cpu()
    diag = torch.diagonal(h, dim1=1, dim2=2)
    dead = diag <= 0
    h[torch.diag_embed(dead)] = 1.0
    diag = torch.diagonal(h, dim1=1, dim2=2)
    h += torch.diag_embed(damp * diag.mean(dim=1, keepdim=True).expand(E, C))
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    hinv = hinv.to(torch.float32).to(w.device)
    hd = torch.diagonal(h, dim1=1, dim2=2).to(torch.float32).to(w.device)  # channel weights for the scale search

    scales = torch.empty(E, R, C // QK, device=w.device)
    codes = torch.empty(E, R, C, dtype=torch.int8, device=w.device)
    for b in range(C // QK):
        lo, hi = b * QK, (b + 1) * QK
        d = block_scales(w[:, :, lo:hi], hd[:, None, lo:hi])
        scales[:, :, b] = d[..., 0]
        safe = torch.where(d == 0, torch.ones_like(d), d)
        err_blk = torch.empty(E, R, QK, device=w.device)
        for j in range(lo, hi):
            col = w[:, :, j : j + 1]
            c = torch.clamp(torch.round(col / safe), -1, 2)
            codes[:, :, j] = c[..., 0].to(torch.int8)
            e = (col - d * c) / hinv[:, j, j].view(E, 1, 1)
            err_blk[:, :, j - lo] = e[..., 0]
            if j + 1 < hi:
                w[:, :, j + 1 : hi] -= e * hinv[:, j : j + 1, j + 1 : hi]
        if hi < C:
            w[:, :, hi:] -= torch.bmm(err_blk, hinv[:, lo:hi, hi:])
    return scales, codes


def dequant(scales: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    E, R, C = codes.shape
    return (codes.to(torch.float32).view(E, R, C // QK, QK) * scales[..., None]).view(E, R, C)


def pack(d: np.ndarray, c: np.ndarray) -> np.ndarray:
    """d [R, nb] (FP16 values), c [R, C] int8 in -1..2 -> uint8 [R, nb * 18] in block_q2_0 layout."""
    rows, nb = d.shape
    q = (c.reshape(rows, nb, 16, 4) + 1).astype(np.uint8)
    qs = q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)
    out = np.empty((rows, nb, 18), dtype=np.uint8)
    out[..., :2] = d.astype("<f2").reshape(rows, nb, 1).view(np.uint8)
    out[..., 2:] = qs
    return out.reshape(rows, nb * 18)


def main() -> None:
    import gguf
    from gguf import GGMLQuantizationType as Q
    from gguf.quants import dequantize

    ap = argparse.ArgumentParser()
    ap.add_argument("bf16")
    ap.add_argument("out")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--grams", type=Path, nargs="+", required=True, help="the capture dir holding this layer's Gram")
    ap.add_argument("--kind", default="ffn_down_exps")
    ap.add_argument("--mode", choices=["gptq", "diag"], default="gptq")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--min-rows", type=int, default=256)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    name = f"blk.{args.layer}.{args.kind}.weight"
    t = {x.name: x for x in gguf.GGUFReader(args.bf16).tensors}[name]
    raw = np.asarray(t.data)
    w = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32) if t.tensor_type == Q.BF16 else dequantize(raw, t.tensor_type)
    E = w.shape[0]
    w = np.ascontiguousarray(w.reshape(E, -1, w.shape[-1]), dtype=np.float32)
    gdir = next(d for d in args.grams if (d / f"{name}.gram.f32").exists())
    gram, n = load_gram(gdir, name, E, w.shape[2])
    diag = np.einsum("eii->ei", gram)
    few = n < args.min_rows
    if args.mode == "diag":
        few[:] = True
    idx = np.arange(w.shape[2])
    for e in np.flatnonzero(few):  # diag fallback: keep only the diagonal
        gram[e] = 0
        gram[e, idx, idx] = diag[e]
    gram += args.alpha * np.einsum("ei,ij->eij", diag, np.eye(w.shape[2], dtype=np.float32))
    scales, codes = gptq_q2(torch.from_numpy(w).to(dev), torch.from_numpy(gram), args.damp)
    d, c = scales.cpu().numpy(), codes.cpu().numpy()
    packed = np.stack([pack(d[e], c[e]) for e in range(E)])
    writer = gguf.GGUFWriter(args.out + ".part", arch="inkling2_loop")
    writer.add_tensor(name, packed, raw_dtype=Q.Q2_0)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    back = {x.name: x for x in gguf.GGUFReader(args.out + ".part").tensors}[name]
    deq = dequantize(np.asarray(back.data), Q.Q2_0).reshape(w.shape)
    assert np.array_equal(deq, dequant(scales, codes).cpu().numpy()), "decoded donor differs from the solver output"
    Path(args.out + ".part").rename(args.out)
    print(f"{name}\tmode {args.mode}\talpha {args.alpha}\tgram {gdir.name}\trows min {n.min()} median {int(np.median(n))}"
          f"\tdiag fallback {int((n < args.min_rows).sum())}", flush=True)


if __name__ == "__main__":
    main()
