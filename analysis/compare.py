"""Upload-gate comparison (CHANGELOG.md): paired per-chunk KL of the candidate against each baseline on T1 and T2.

    python analysis/compare.py CAND BASELINE [BASELINE ...]   (run names in results/kl.tsv without the -T1/-T2 suffix)

Prints, per test set and baseline, the mean KL of both, the paired difference (candidate - baseline) with a 95%
bootstrap interval over chunks, and the top-1 agreement of both. The gate needs the interval below zero for every
baseline on both sets and top-1 not worse.
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load() -> dict[str, dict]:
    rows = csv.DictReader(open(ROOT / "results/kl.tsv"), delimiter="\t")
    return {r["run"]: {"chunks": np.array([float(x) for x in r["per_chunk"].split(",")]), "top": float(r["same_top_p"])} for r in rows}


def paired(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    if a.shape != b.shape:
        raise ValueError(f"paired runs need the same chunks: {a.shape} vs {b.shape}")
    d = a - b
    idx = np.random.default_rng(seed).integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    cand, *bases = sys.argv[1:]
    runs = load()
    ok = True
    for t in ("T1", "T2"):
        c = runs[f"final-{cand}-{t}-P3"]
        for b in bases:
            o = runs[f"final-{b}-{t}-P3"]
            diff, lo, hi = paired(c["chunks"], o["chunks"])
            win = hi < 0 and c["top"] >= o["top"]
            ok &= win
            print(f"{t} {cand} vs {b}: KL {c['chunks'].mean():.5f} vs {o['chunks'].mean():.5f}, diff {diff:+.5f} [{lo:+.5f}, {hi:+.5f}], "
                  f"top-1 {c['top']:.2f}% vs {o['top']:.2f}% -> {'better' if win else 'not better'}")
    print("quality gate:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
