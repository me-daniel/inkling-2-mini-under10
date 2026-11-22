"""Mean-aware scale refit for IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S expert tensors (codes and fp16 d unchanged).

    python scripts/meanfit.py SRC.gguf IMATRIX.gguf OUT.gguf --layers 0-23 [--kind ffn_gate_up_exps] [--bf16 BF16.gguf]

llama-quantize picks codes and sub-block scales under a diagonal importance weight. The expert inputs have a large
common mean (the sibling measured about 20% of input energy in it), so the output error of a row is closer to
    n * [ sum_j v_j dw_j^2 + (m . dw)^2 ],   v = E[x^2] - m^2,  m = E[x]   (both per expert, from the imatrix)
than to the diagonal term alone. With the codes fixed, each sub-block's dequantized values are sigma_b * u_b, so the
error is quadratic in the sub-block scales sigma. This solves (A + p p^T) sigma = r + q p per row in closed form
(Sherman-Morrison), rounds each sigma_b to the type's scale grid under the block's fp16 d, runs two passes of
coordinate descent on the discrete scale bits, keeps the original bits where they score better, and writes only the
scale bits back. Each tensor is checked: only scale bits changed, decode == sigma_new * u. Writes a donor GGUF.
"""

import argparse
import sys

import numpy as np

import gguf
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize

QK = 256
# per type: block bytes, sub-block size, number of scale levels, scale factor f(ls) (sigma = d * f), get/set of ls
def _f_iq2(ls): return (0.5 + ls) * 0.25
def _f_iq1(ls): return 2.0 * ls + 1.0


def get_ls(t: Q, b: np.ndarray) -> np.ndarray:
    if t == Q.IQ2_XXS:   # 8 bytes per 32 weights: the top nibble of byte 7 of each group
        return (b[:, 2:].reshape(len(b), 8, 8)[:, :, 7] >> 4).astype(np.int64)
    if t in (Q.IQ2_XS, Q.IQ2_S):  # scales[8] bytes, low nibble then high nibble, one per 16 weights
        off = 66 if t == Q.IQ2_XS else 74
        s = b[:, off:off + 8]
        return np.stack([s & 15, s >> 4], axis=2).reshape(len(b), 16).astype(np.int64)
    if t == Q.IQ1_S:     # qh uint16[8] at byte 34: bits 12-14 per 32 weights
        qh = b[:, 34:50].copy().view(np.uint16)
        return ((qh >> 12) & 7).astype(np.int64)
    raise SystemExit(f"type {t.name} not supported")


def set_ls(t: Q, b: np.ndarray, ls: np.ndarray) -> np.ndarray:
    b = b.copy()
    if t == Q.IQ2_XXS:
        g = b[:, 2:].reshape(len(b), 8, 8)
        g[:, :, 7] = (g[:, :, 7] & 15) | (ls.astype(np.uint8) << 4)
        b[:, 2:] = g.reshape(len(b), 64)
    elif t in (Q.IQ2_XS, Q.IQ2_S):
        off = 66 if t == Q.IQ2_XS else 74
        l2 = ls.reshape(len(b), 8, 2).astype(np.uint8)
        b[:, off:off + 8] = l2[:, :, 0] | (l2[:, :, 1] << 4)
    elif t == Q.IQ1_S:
        qh = b[:, 34:50].copy().view(np.uint16)
        qh = (qh & np.uint16(0x8FFF)) | (ls.astype(np.uint16) << 12)
        b[:, 34:50] = qh.view(np.uint8)
    return b


SPEC = {Q.IQ2_XXS: (66, 32, 16, _f_iq2), Q.IQ2_XS: (74, 16, 16, _f_iq2), Q.IQ2_S: (82, 16, 16, _f_iq2), Q.IQ1_S: (50, 32, 8, _f_iq1)}


def refit_expert(t: Q, blocks: np.ndarray, w: np.ndarray, v: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, float, float]:
    """blocks [R*nb, bytes], w [R, C] source weights, v, m [C]. -> (new blocks, err_before, err_after) in the model."""
    nbytes, sb, nlev, f = SPEC[t]
    R, C = w.shape
    nsub = QK // sb
    d = blocks[:, :2].copy().view(np.float16).astype(np.float64).reshape(R, -1, 1)       # [R, nb, 1]
    ls0 = get_ls(t, blocks).reshape(R, -1, nsub)                                         # [R, nb, nsub]
    deq = dequantize(blocks, t).astype(np.float64).reshape(R, -1, nsub, sb)
    sig0 = d * f(ls0)
    safe = np.where(sig0 == 0, 1, sig0)
    u = deq / safe[..., None]                                                            # unit codes
    wb = w.astype(np.float64).reshape(R, -1, nsub, sb)
    vb, mb = v.reshape(1, -1, nsub, sb), m.reshape(1, -1, nsub, sb)
    a = (vb * u * u).sum(-1); r = (vb * u * wb).sum(-1); p = (mb * u).sum(-1)            # [R, nb, nsub]
    q = (m.reshape(1, -1) * w).sum(-1)                                                    # [R]
    A, Rr, P = a.reshape(R, -1), r.reshape(R, -1), p.reshape(R, -1)
    def err(sig):  # up to a constant per row
        s = sig.reshape(R, -1)
        return (A * s * s - 2 * Rr * s).sum(-1) + ((P * s).sum(-1) - q) ** 2
    const = (v.reshape(1, -1) * w * w).sum(-1)
    Aa = np.where(A > 0, A, 1e-30)
    x = (Rr + q[:, None] * P) / Aa; y = P / Aa
    sig = x - y * ((P * x).sum(-1) / (1 + (P * y).sum(-1)))[:, None]                     # continuous optimum
    dd = np.broadcast_to(d, ls0.shape).reshape(R, -1)
    lev = f(np.arange(nlev))                                                              # [nlev]
    ls = np.abs(sig[..., None] - dd[..., None] * lev).argmin(-1)                          # nearest level
    for _ in range(2):  # coordinate descent on the discrete scale bits
        s = dd * f(ls); g = (P * s).sum(-1) - q
        for j in range(ls.shape[1]):
            best_d, best_l = np.zeros(R), ls[:, j].copy()
            for step in (-1, 1):
                l2 = np.clip(ls[:, j] + step, 0, nlev - 1)
                delta = dd[:, j] * (f(l2) - f(ls[:, j]))
                de = A[:, j] * delta * delta + 2 * delta * (A[:, j] * s[:, j] - Rr[:, j]) + (g + P[:, j] * delta) ** 2 - g ** 2
                better = de < best_d
                best_d = np.where(better, de, best_d); best_l = np.where(better, l2, best_l)
            delta = dd[:, j] * (f(best_l) - f(ls[:, j]))
            g = g + P[:, j] * delta; s[:, j] = s[:, j] + delta; ls[:, j] = best_l
    e0, e1 = err(sig0) + const, err(dd * f(ls)) + const
    keep = e1 > e0                                                                        # never worse than llama-quantize
    ls = np.where(keep[:, None], ls0.reshape(R, -1), ls)
    e1 = np.where(keep, e0, e1)
    new = set_ls(t, blocks, ls.reshape(len(blocks), nsub))
    chk = dequantize(new, t).astype(np.float64).reshape(R, -1, nsub, sb)
    assert np.allclose(chk, (dd * f(ls)).reshape(R, -1, nsub)[..., None] * u, rtol=1e-5, atol=1e-12), "decode mismatch"
    return new, float(e0.sum()), float(e1.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("imatrix"); ap.add_argument("out")
    ap.add_argument("--layers", required=True, help="lo-hi or a comma list"); ap.add_argument("--kind", default="ffn_gate_up_exps")
    ap.add_argument("--bf16", default="/Users/daniel/GitHub/inkling-2-mini-quant/models/Inkling-2-Mini-BF16.gguf")
    a = ap.parse_args()
    if "," in a.layers or "-" not in a.layers:
        todo = [int(x) for x in a.layers.split(",")]
    else:
        lo, hi = (int(x) for x in a.layers.split("-")); todo = list(range(lo, hi + 1))
    src = {t.name: t for t in gguf.GGUFReader(a.src).tensors}
    ref = {t.name: t for t in gguf.GGUFReader(a.bf16).tensors}
    im = {t.name: t for t in gguf.GGUFReader(a.imatrix).tensors}
    wr = gguf.GGUFWriter(a.out + ".part", arch="inkling2_loop")
    for layer in todo:
        n = f"blk.{layer}.{a.kind}.weight"
        t = src[n]; ty = t.tensor_type
        if ty not in SPEC:
            print(f"{n}: {ty.name} not supported, skipped", file=sys.stderr); continue
        cnt = np.asarray(im[n + ".counts"].data, dtype=np.float64).reshape(-1)
        s1 = np.asarray(im[n + ".in_sum"].data, dtype=np.float64).reshape(len(cnt), -1)
        s2 = np.asarray(im[n + ".in_sum2"].data, dtype=np.float64).reshape(len(cnt), -1)
        raw = np.asarray(t.data); E = raw.shape[0]
        wsrc = (np.asarray(ref[n].data).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        out, tot0, tot1 = [], 0.0, 0.0
        for e in range(E):
            c = max(cnt[e], 1.0); m = s1[e] / c; v = np.maximum(s2[e] / c - m * m, 0)
            blocks = np.ascontiguousarray(raw[e]).reshape(-1, SPEC[ty][0])
            new, e0, e1 = refit_expert(ty, blocks, wsrc[e], v, m)
            changed = np.flatnonzero((new != blocks).any(0))
            out.append(new.reshape(raw[e].shape)); tot0 += e0 * cnt[e]; tot1 += e1 * cnt[e]
        wr.add_tensor(n, np.stack(out), raw_dtype=ty)
        print(f"{n}\t{ty.name}\tmodel error after/before {tot1 / tot0:.4f}\tbyte columns touched {changed.tolist()[:12]}", flush=True)
    wr.write_header_to_file(); wr.write_kv_data_to_file(); wr.write_tensors_to_file(); wr.close()
    import os; os.replace(a.out + ".part", a.out)


if __name__ == "__main__":
    main()
