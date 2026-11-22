"""Rescale every output row of an expert tensor to gain 1 against BF16 (codes unchanged): block fp16 d *= 1/gain_row.
    python scripts/degain2.py SRC.gguf OUT.gguf LAYER --kind ffn_down_exps|ffn_gate_up_exps
Supports Q2_0 (64-weight blocks), IQ4_NL (32) and IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S (256-weight blocks); d is the first 2 bytes of
each block. If SRC has the per-expert .scale for the tensor (Unsloth files), gain uses decode * scale and the .scale is
copied into OUT unchanged. Checks the decode; prints gain and energy before and after."""
import argparse, os
import numpy as np, gguf
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize
BLK = {Q.Q2_0: (64, 18), Q.IQ4_NL: (32, 18), Q.IQ1_S: (256, 50), Q.IQ2_XXS: (256, 66), Q.IQ2_XS: (256, 74), Q.IQ2_S: (256, 82)}
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("layer", type=int)
ap.add_argument("--kind", required=True); ap.add_argument("--bf16", default="/Users/daniel/GitHub/inkling-2-mini-quant/models/Inkling-2-Mini-BF16.gguf")
a = ap.parse_args(); n = f"blk.{a.layer}.{a.kind}.weight"; sn = n.replace(".weight", ".scale")
T = {x.name: x for x in gguf.GGUFReader(a.src).tensors}; t = T[n]; ty = t.tensor_type
assert ty in BLK, ty.name
qk, nb = BLK[ty]
sc = np.asarray(T[sn].data, dtype=np.float64) if sn in T else None
W = (np.asarray({x.name: x for x in gguf.GGUFReader(a.bf16).tensors}[n].data).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
raw = np.asarray(t.data); E = raw.shape[0]; out = []; acc = np.zeros(5)
for e in range(E):
    blocks = np.ascontiguousarray(raw[e]).reshape(-1, nb).copy()
    w = W[e].astype(np.float64); R = w.shape[0]; s = sc[e] if sc is not None else 1.0
    q = dequantize(blocks, ty).reshape(w.shape).astype(np.float64) * s
    gain = (q * w).sum(1) / np.maximum((w * w).sum(1), 1e-30)
    fac = np.where(gain > 0.2, 1.0 / gain, 1.0)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float64).reshape(R, -1)
    blocks[:, :2] = (d * fac[:, None]).astype(np.float16).reshape(-1, 1).view(np.uint8)
    q2 = dequantize(blocks, ty).reshape(w.shape).astype(np.float64) * s
    assert np.allclose(q2, q * fac[:, None], rtol=2e-3, atol=1e-9)
    acc += [(q * w).sum(), (q * q).sum(), (q2 * w).sum(), (q2 * q2).sum(), (w * w).sum()]
    out.append(blocks.reshape(raw[e].shape))
wr = gguf.GGUFWriter(a.out + ".part", arch="inkling2_loop"); wr.add_tensor(n, np.stack(out), raw_dtype=ty)
if sc is not None:
    wr.add_tensor(sn, np.asarray(T[sn].data, dtype=np.float32))
wr.write_header_to_file(); wr.write_kv_data_to_file(); wr.write_tensors_to_file(); wr.close(); os.replace(a.out + ".part", a.out)
print(f"{n}\t{ty.name}\tgain {acc[0]/acc[4]:.4f} -> {acc[2]/acc[4]:.4f}\tenergy {acc[1]/acc[4]:.4f} -> {acc[3]/acc[4]:.4f}", flush=True)
