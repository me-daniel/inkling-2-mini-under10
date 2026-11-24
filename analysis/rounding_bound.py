"""Worst-case effect of the printed precision on paired CIs (decisions review 24 Nov, both reviewers).

llama-perplexity prints the running mean KL m_k with 5 decimals, so each m_k is off by at most 5e-6. The per-chunk
value k*m_k - (k-1)*m_(k-1) is then off by at most (2k-1)*5e-6, and a paired difference of two runs by twice that.
A bootstrap resample mean is an average of such values, so no CI endpoint can move by more than the largest one,
2*(2N-1)*5e-6 for N chunks. The reported mean itself telescopes to m_N and is off by at most 1e-5.

    python analysis/rounding_bound.py A B [A B ...]   (run names in results/kl.tsv)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import load, paired

k = load(); args = sys.argv[1:]
for a, b in zip(args[::2], args[1::2]):
    x, y = k[a]["chunks"], k[b]["chunks"]; d, lo, hi = paired(x, y); e = 2 * (2 * x.size - 1) * 5e-6
    robust = "sign robust" if hi + e < 0 or lo - e > 0 else ("spans 0 even at worst" if lo + e < 0 < hi - e else "NOT robust")
    print(f"{a} - {b}: {d:+.5f} [{lo:+.5f}, {hi:+.5f}], bound ±{e:.5f} -> CI could be [{lo - e:+.5f}..{lo + e:+.5f}, {hi - e:+.5f}..{hi + e:+.5f}]: {robust}")
