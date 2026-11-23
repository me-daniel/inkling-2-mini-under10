"""meanfit.refit_expert after the 23 Nov code reviews: Astra's two counterexamples reach the exact optimum, and on random
blocks of every type the refit never raises the error and changes only scale bits."""

import importlib.util
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as Q

_spec = importlib.util.spec_from_file_location("meanfit", Path(__file__).resolve().parents[1] / "scripts/meanfit.py")
mf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mf)


def iq2s_block(d: float = 1.0) -> np.ndarray:
    b = np.zeros((1, 82), dtype=np.uint8)
    b[0, :2] = np.array([d], dtype=np.float16).view(np.uint8)  # grid index 0 (all lanes 8), signs +, scales 0
    return b


def direct(t, blocks, w, v, m):
    from gguf.quants import dequantize
    dw = dequantize(blocks, t).astype(np.float64).reshape(w.shape) - w
    return float((v * dw * dw).sum() + (((m * dw).sum(-1)) ** 2).sum())  # the mean term is squared per row


def test_mean_only_coordinate_reaches_level_10():
    # Astra 16: v = 0, m[0] = 1, w[0] = 21. Level l gives 1 + 2l on sub-block 0, so l = 10 is exact.
    b = iq2s_block(); w = np.zeros((1, 256)); w[0, 0] = 21.0
    v = np.zeros(256); m = np.zeros(256); m[0] = 1.0
    new, e0, e1 = mf.refit_expert(Q.IQ2_S, b, w, v, m)
    assert mf.get_ls(Q.IQ2_S, new)[0, 0] == 10
    assert e0 == 400.0 and e1 == 0.0


def test_wide_variance_range_keeps_exact_correction():
    # Astra 17: v[0] = 2^60, v[16] = 1, w[0] = 1, w[16] = 3, m = 0. Sub-block 1 at level 1 makes the error 0.
    b = iq2s_block(); w = np.zeros((1, 256)); w[0, 0] = 1.0; w[0, 16] = 3.0
    v = np.zeros(256); v[0] = 2.0 ** 60; v[16] = 1.0; m = np.zeros(256)
    new, e0, e1 = mf.refit_expert(Q.IQ2_S, b, w, v, m)
    assert e0 == 4.0 and e1 == 0.0 and mf.get_ls(Q.IQ2_S, new)[0, 1] == 1


def test_random_blocks_only_scale_bits_and_no_worse():
    rng = np.random.default_rng(0)
    for t in (Q.IQ1_S, Q.IQ2_XXS, Q.IQ2_XS, Q.IQ2_S):
        nbytes = mf.SPEC[t][0]; R, nb = 4, 2
        blocks = rng.integers(0, 256, size=(R * nb, nbytes), dtype=np.uint8)
        blocks[:, :2] = np.full((R * nb, 1), 0.01, dtype=np.float16).view(np.uint8)
        from gguf.quants import dequantize
        w = dequantize(blocks, t).astype(np.float64).reshape(R, -1) * 1.1 + rng.normal(0, 0.002, (R, 256 * nb))
        v = rng.uniform(0, 2, 256 * nb); m = rng.normal(0, 0.5, 256 * nb)
        new, e0, e1 = mf.refit_expert(t, blocks, w, v, m)
        assert not ((new ^ blocks) & ~mf.scale_mask(t)).any(), t.name
        assert e1 <= e0 + 1e-12, t.name
        assert abs(direct(t, new, w, v, m) - e1) <= 1e-9 * max(1.0, e1), t.name
