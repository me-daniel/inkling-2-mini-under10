import importlib.util, pathlib
import numpy as np

spec = importlib.util.spec_from_file_location("mq", pathlib.Path(__file__).parents[1] / "scripts/meanfit_q2.py")
mq = importlib.util.module_from_spec(spec); spec.loader.exec_module(mq)


def toy(seed=0, R=6, nb=4):
    g = np.random.default_rng(seed)
    w = g.normal(size=(R, nb * mq.QK)); c = g.integers(-1, 3, size=(R, nb, mq.QK)).astype(float)
    d = np.abs(g.normal(size=(R, nb))) * 0.5 + 0.1
    return d, c, w


def test_no_mean_is_per_block_least_squares():
    d, c, w = toy()
    v = np.ones(w.shape[1]); m = np.zeros(w.shape[1])
    d1, e0, e1 = mq.refit_rows(d, c, w, v, m)
    ls = (c * w.reshape(c.shape)).sum(-1) / (c * c).sum(-1)
    assert np.allclose(d1, ls.astype(np.float16).astype(float))
    assert (e1 <= e0).all()


def test_mean_term_never_worse_and_matches_dense_solve():
    d, c, w = toy(1)
    g = np.random.default_rng(2); v = g.uniform(0.5, 2, w.shape[1]); m = g.normal(size=w.shape[1])
    d1, e0, e1 = mq.refit_rows(d, c, w, v, m)
    assert (e1 <= e0).all()
    r0 = 0; U = np.zeros((w.shape[1], d.shape[1]))
    for b in range(d.shape[1]): U[b * mq.QK:(b + 1) * mq.QK, b] = c[r0, b]
    H = U.T @ (v[:, None] * U) + np.outer(U.T @ m, U.T @ m)
    s = np.linalg.solve(H, U.T @ (v * w[r0]) + (U.T @ m) * (m @ w[r0]))
    assert np.allclose(d1[r0], s.astype(np.float16).astype(float), rtol=2e-3)
