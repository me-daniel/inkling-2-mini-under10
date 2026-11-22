"""Imatrix-weighted Q2_0 encoder for the routed experts (block of 64, one FP16 scale d, levels d * {-1, 0, 1, 2}).

    python scripts/q2opt.py BF16.gguf IMATRIX.gguf OUT.gguf --layers 0-23 --kinds ffn_down_exps [ffn_gate_up_exps]

llama.cpp's reference encoder sets d = max|w| and rounds w/d, so the level 2d is never used and the imatrix is
ignored. Here every block tries signed scale candidates d = +-f * max|w| (f on a grid, plus the stock choice), assigns
the nearest legal levels, refits d by weighted least squares (weights: the expert's mean squared input activation
per channel from the imatrix), rounds d to FP16, reassigns, and keeps the candidate with the smallest weighted error.
The stock encoding is one of the candidates, so the weighted error is never worse than stock. Writes a donor GGUF
holding only the encoded tensors (for scripts/splice.py) and a TSV of errors next to it.
"""

import argparse
import sys

import numpy as np

import gguf
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize

QK = 64
FACTORS = np.round(np.arange(0.30, 1.0001, 0.05), 2)


def bf16_rows(t: gguf.ReaderTensor, e: int) -> np.ndarray:
    raw = np.asarray(t.data[e])
    if t.tensor_type == Q.BF16:
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return dequantize(raw, t.tensor_type).astype(np.float32)


def encode(w: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w [rows, cols], h [cols] -> (d float16 [rows, nb], codes int8 [rows, nb, 64] in -1..2)."""
    rows, cols = w.shape
    x = w.reshape(rows, cols // QK, QK).astype(np.float32)
    hw = np.broadcast_to(h.reshape(cols // QK, QK).astype(np.float32), x.shape)
    amax = np.abs(x).max(axis=2, keepdims=True)
    best_err = np.full(amax.shape, np.inf, dtype=np.float32)
    best_d = np.zeros(amax.shape, dtype=np.float16)
    best_c = np.zeros(x.shape, dtype=np.int8)

    def consider(d: np.ndarray, refit: bool) -> None:
        nonlocal best_err, best_d, best_c
        safe = np.where(d == 0, 1, d)
        c = np.clip(np.rint(x / safe), -1, 2)
        if refit:
            den = np.sum(hw * c * c, axis=2, keepdims=True)
            num = np.sum(hw * x * c, axis=2, keepdims=True)
            d = np.where(den > 0, num / np.where(den > 0, den, 1), d)
        d16 = d.astype(np.float16)
        d32 = d16.astype(np.float32)
        c = np.clip(np.rint(x / np.where(d32 == 0, 1, d32)), -1, 2)
        err = np.sum(hw * (x - d32 * c) ** 2, axis=2, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d16, best_d)
        best_c = np.where(better, c.astype(np.int8), best_c)

    consider(amax, refit=False)  # the stock llama.cpp encoding
    for f in FACTORS:
        for sign in (1.0, -1.0):
            consider(sign * f * amax, refit=True)
    return best_d[..., 0], best_c


def pack(d: np.ndarray, c: np.ndarray) -> np.ndarray:
    """-> uint8 [rows, nb * 18] in block_q2_0 layout: fp16 d, then 64 2-bit codes (q + 1), 4 per byte, low bits first."""
    rows, nb, _ = c.shape
    q = (c + 1).astype(np.uint8).reshape(rows, nb, 16, 4)
    qs = q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)
    out = np.empty((rows, nb, 18), dtype=np.uint8)
    out[..., :2] = d.astype("<f2").reshape(rows, nb, 1).view(np.uint8)
    out[..., 2:] = qs
    return out.reshape(rows, nb * 18)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bf16")
    ap.add_argument("imatrix")
    ap.add_argument("out")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--kinds", nargs="+", default=["ffn_down_exps"])
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.layers.split("-"))

    src = {t.name: t for t in gguf.GGUFReader(args.bf16).tensors}
    imat = {t.name: t for t in gguf.GGUFReader(args.imatrix).tensors}
    writer = gguf.GGUFWriter(args.out, arch="inkling2_loop")
    rows_out = ["layer\tkind\tt_stock\tt_opt\ttw_stock\ttw_opt"]
    for layer in range(lo, hi + 1):
        for kind in args.kinds:
            name = f"blk.{layer}.{kind}.weight"
            t = src[name]
            sums = np.asarray(imat[name + ".in_sum2"].data, dtype=np.float64)  # [expert, in]
            counts = np.asarray(imat[name + ".counts"].data, dtype=np.float64).reshape(-1)
            packed, acc = [], np.zeros(4)
            for e in range(t.data.shape[0]):
                w = bf16_rows(t, e)
                h = sums[e] / counts[e] if counts[e] > 0 else np.ones(w.shape[1])
                h = np.maximum(h, 1e-12)
                d, c = encode(w, h)
                block = pack(d, c)
                packed.append(block)
                deq = dequantize(block, Q.Q2_0).reshape(w.shape)
                amax = np.abs(w.reshape(w.shape[0], -1, QK)).max(axis=2, keepdims=True)
                stock = (np.clip(np.rint(w.reshape(w.shape[0], -1, QK) / np.where(amax == 0, 1, amax)), -1, 2) * amax.astype(np.float16).astype(np.float32)).reshape(w.shape)
                ww = np.broadcast_to(h, w.shape)
                acc += [np.sum((stock - w) ** 2), np.sum((deq - w) ** 2), np.sum(ww * (stock - w) ** 2), np.sum(ww * (deq - w) ** 2)]
                norm = np.sum(w.astype(np.float64) ** 2)
                normw = np.sum(ww * w.astype(np.float64) ** 2)
                acc_norm = (norm, normw)
                if e == 0:
                    totals = np.zeros(2)
                totals += acc_norm
            writer.add_tensor(name, np.stack(packed), raw_dtype=Q.Q2_0)
            row = f"{layer}\t{kind}\t{acc[0] / totals[0]:.4f}\t{acc[1] / totals[0]:.4f}\t{acc[2] / totals[1]:.4f}\t{acc[3] / totals[1]:.4f}"
            rows_out.append(row)
            print(row, file=sys.stderr, flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with open(args.out + ".tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(rows_out) + "\n")


if __name__ == "__main__":
    main()
