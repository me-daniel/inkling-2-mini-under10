"""De-attenuate Q2_0 tensors: per output row, scale every block's fp16 d by 1/gain, gain = <q,w>/<w,w> against BF16.
    python scripts/degain.py SRC.gguf OUT.gguf LAYER [--kind ffn_down_exps]
Codes are unchanged; the row's decoded weights become q/gain (unbiased in the row direction). Checks the decode."""
import argparse, os
import numpy as np, gguf
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("layer", type=int)
ap.add_argument("--kind", default="ffn_down_exps"); ap.add_argument("--bf16", default="/Users/daniel/GitHub/inkling-2-mini-quant/models/Inkling-2-Mini-BF16.gguf")
a = ap.parse_args(); n = f"blk.{a.layer}.{a.kind}.weight"
t = {x.name: x for x in gguf.GGUFReader(a.src).tensors}[n]; assert t.tensor_type == Q.Q2_0, t.tensor_type.name
W = (np.asarray({x.name: x for x in gguf.GGUFReader(a.bf16).tensors}[n].data).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
raw = np.asarray(t.data); E = raw.shape[0]; out = []; g_before = []; g_after = []
for e in range(E):
    blocks = np.ascontiguousarray(raw[e]).reshape(-1, 18).copy()          # block_q2_0: fp16 d + 16 bytes of codes
    w = W[e].astype(np.float64); R = w.shape[0]
    q = dequantize(blocks, Q.Q2_0).reshape(w.shape).astype(np.float64)
    gain = (q * w).sum(1) / np.maximum((w * w).sum(1), 1e-30)            # per row
    fac = np.where(gain > 0.2, 1.0 / gain, 1.0)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float64).reshape(R, -1)
    blocks[:, :2] = (d * fac[:, None]).astype(np.float16).reshape(-1, 1).view(np.uint8)
    q2 = dequantize(blocks, Q.Q2_0).reshape(w.shape).astype(np.float64)
    assert np.allclose(q2, q * fac[:, None], rtol=2e-3, atol=1e-7)
    g_before.append(((q * w).sum(), (w * w).sum())); g_after.append(((q2 * w).sum(), (w * w).sum()))
    out.append(blocks.reshape(raw[e].shape))
wr = gguf.GGUFWriter(a.out + ".part", arch="inkling2_loop"); wr.add_tensor(n, np.stack(out), raw_dtype=Q.Q2_0)
wr.write_header_to_file(); wr.write_kv_data_to_file(); wr.write_tensors_to_file(); wr.close(); os.replace(a.out + ".part", a.out)
gb, ga = np.array(g_before).sum(0), np.array(g_after).sum(0)
print(f"{n}: gain {gb[0] / gb[1]:.4f} -> {ga[0] / ga[1]:.4f}")
