import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gptq_q2 as g  # noqa: E402


def out_err(w, wq, h):
    dw = (wq - w).double()
    return torch.einsum("erc,ecd,erd->", dw, h.double(), dw).item()


def problem(seed, E=2, R=16, C=128, corr=True):
    rng = np.random.default_rng(seed)
    w = torch.from_numpy(rng.standard_normal((E, R, C)).astype(np.float32))
    x = rng.standard_normal((E, 4 * C, C)).astype(np.float32)
    if corr:
        x = x @ rng.standard_normal((C, C)).astype(np.float32) * 0.3 + x
    h = torch.from_numpy(np.einsum("eni,enj->eij", x, x))
    return w, h


def test_identity_hessian_equals_weighted_rounding():
    w, _ = problem(0, corr=False)
    E, R, C = w.shape
    h = torch.eye(C).expand(E, C, C).clone()
    d, c = g.gptq_q2(w, h, damp=0.0)
    # without off-diagonal curvature nothing propagates: plain rounding at the chosen scales
    rtn = torch.clamp(torch.round(w.view(E, R, -1, 64) / d[..., None]), -1, 2).view(E, R, C)
    assert torch.equal(c.to(torch.float32), rtn)


def test_gptq_beats_rounding_on_correlated_inputs():
    w, h = problem(1)
    E, R, C = w.shape
    d, c = g.gptq_q2(w, h)
    gp = g.dequant(d, c)
    diag_h = torch.diag_embed(torch.diagonal(h, dim1=1, dim2=2))
    d0, c0 = g.gptq_q2(w, diag_h)
    rtn = g.dequant(d0, c0)
    assert out_err(w, gp, h) < 0.9 * out_err(w, rtn, h)


def test_pack_roundtrip():
    from gguf import GGMLQuantizationType as Q
    from gguf.quants import dequantize

    w, h = problem(2, E=1, R=4, C=128)
    d, c = g.gptq_q2(w, h)
    packed = g.pack(d[0].numpy(), c[0].numpy())
    assert np.allclose(dequantize(packed, Q.Q2_0).reshape(4, 128), g.dequant(d, c)[0].numpy())
