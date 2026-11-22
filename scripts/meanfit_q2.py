"""Mean-aware refit of the fp16 block scales of a Q2_0 expert tensor (codes unchanged), then per-row gain restore.

    python scripts/meanfit_q2.py SRC.gguf IMATRIX.gguf OUT.gguf LAYER [--kind ffn_down_exps] [--target 1.0]

Q2_0 decodes block b (64 weights) as d_b * c_j with c_j in {-1, 0, 1, 2}. With the codes fixed, the per-row surrogate
    sum_j v_j dw_j^2 + (m . dw)^2,   v = E[x^2] - m^2,  m = E[x]   (per expert, from the imatrix)
is quadratic in the block scales s = (d_b): (diag A + p p^T) s = r + q p, solved by Sherman-Morrison. Blocks whose codes
are all zero keep their d. A row keeps the refit only if its surrogate error strictly improves in float64 after fp16
rounding. Then every row is rescaled to the target gain <q,w>/<w,w> (M0rk's lesson: refit, then restore the gain).
Checks decode == new d * codes. Writes a one-tensor donor GGUF.
"""

import argparse
import os

import numpy as np

import gguf
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize

QK, NB = 64, 18  # Q2_0: 64 weights, 2-byte fp16 d + 16 bytes of 2-bit codes


def refit_rows(d: np.ndarray, c: np.ndarray, w: np.ndarray, v: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """d [R, nb], c [R, nb, QK] codes, w [R, C], v, m [C] -> (new d [R, nb], err before [R], err after [R])."""
    R, nb = d.shape
    wb = w.reshape(R, nb, QK); vb = v.reshape(1, nb, QK); mb = m.reshape(1, nb, QK)
    A = (vb * c * c).sum(-1); r = (vb * c * wb).sum(-1); p = (mb * c).sum(-1); q = (m.reshape(1, -1) * w).sum(-1)
    live = A > 0
    Aa = np.where(live, A, 1.0)
    z = np.where(live, r / Aa, 0.0); y = np.where(live, p / Aa, 0.0)
    s = z + y * ((q - (p * z).sum(-1)) / (1.0 + (p * y).sum(-1)))[:, None]
    s = np.where(live, s, d)
    s16 = s.astype(np.float16).astype(np.float64)

    def err(sc):  # exact surrogate from the decoded row
        dw = (sc[..., None] * c).reshape(R, -1) - w
        return (v.reshape(1, -1) * dw * dw).sum(-1) + ((m.reshape(1, -1) * dw).sum(-1)) ** 2

    e0, e1 = err(d), err(s16)
    better = np.isfinite(e1) & (e1 < e0 * (1 - 1e-9))
    return np.where(better[:, None], s16, d), e0, np.where(better, e1, e0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("imatrix"); ap.add_argument("out"); ap.add_argument("layer", type=int)
    ap.add_argument("--kind", default="ffn_down_exps"); ap.add_argument("--target", type=float, default=1.0)
    ap.add_argument("--bf16", default=str(os.path.join(os.path.dirname(__file__), "../../inkling-2-mini-quant/models/Inkling-2-Mini-BF16.gguf")))
    a = ap.parse_args()
    n = f"blk.{a.layer}.{a.kind}.weight"
    t = {x.name: x for x in gguf.GGUFReader(a.src).tensors}[n]
    assert t.tensor_type == Q.Q2_0, t.tensor_type.name
    Wraw = np.asarray({x.name: x for x in gguf.GGUFReader(a.bf16).tensors}[n].data)
    im = {x.name: x for x in gguf.GGUFReader(a.imatrix).tensors}
    cnt = np.asarray(im[n + ".counts"].data, dtype=np.float64).ravel()
    s1 = np.asarray(im[n + ".in_sum"].data, dtype=np.float64).reshape(len(cnt), -1)
    s2 = np.asarray(im[n + ".in_sum2"].data, dtype=np.float64).reshape(len(cnt), -1)
    raw = np.asarray(t.data); out = []; tot = np.zeros(2); acc = np.zeros(5)
    for e in range(raw.shape[0]):
        blocks = np.ascontiguousarray(raw[e]).reshape(-1, NB).copy()
        w = (Wraw[e].view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64); R, C = w.shape
        d = blocks[:, :2].copy().view(np.float16).astype(np.float64).reshape(R, -1)
        deq = dequantize(blocks, Q.Q2_0).reshape(R, -1, QK).astype(np.float64)
        c = np.round(np.where(d[..., None] != 0, deq / np.where(d == 0, 1, d)[..., None], 0))
        assert np.allclose(c * d[..., None], deq, rtol=1e-6, atol=1e-12), "codes not recovered"
        if cnt[e] > 0:
            mu = s1[e] / cnt[e]; v = np.maximum(s2[e] / cnt[e] - mu * mu, 0)
            d1, e0, e1 = refit_rows(d, c, w, v, mu); tot += [e0.sum() * cnt[e], e1.sum() * cnt[e]]
        else:
            d1 = d
        q = (d1[..., None] * c).reshape(R, C)
        gain = (q * w).sum(1) / np.maximum((w * w).sum(1), 1e-30)
        fac = np.where(gain > 0.2, a.target / gain, 1.0)
        d2 = (d1 * fac[:, None]).astype(np.float16)
        blocks[:, :2] = d2.reshape(-1, 1).view(np.uint8)
        chk = dequantize(blocks, Q.Q2_0).reshape(R, -1, QK).astype(np.float64)
        assert np.allclose(chk, d2.astype(np.float64)[..., None] * c, rtol=1e-6, atol=1e-12), "decode mismatch"
        q2 = chk.reshape(R, C)
        acc += [(q * w).sum(), (q * q).sum(), (q2 * w).sum(), (q2 * q2).sum(), (w * w).sum()]
        out.append(blocks.reshape(raw[e].shape))
    wr = gguf.GGUFWriter(a.out + ".part", arch="inkling2_loop"); wr.add_tensor(n, np.stack(out), raw_dtype=Q.Q2_0)
    wr.write_header_to_file(); wr.write_kv_data_to_file(); wr.write_tensors_to_file(); wr.close(); os.replace(a.out + ".part", a.out)
    print(f"{n}\tsurrogate after/before {tot[1] / tot[0]:.4f}\tgain after refit {acc[0] / acc[4]:.4f} -> {acc[2] / acc[4]:.4f}\t"
          f"energy {acc[1] / acc[4]:.4f} -> {acc[3] / acc[4]:.4f}", flush=True)


if __name__ == "__main__":
    main()
