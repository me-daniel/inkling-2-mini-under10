"""Paired comparison of 8-chunk S screens with the all chunks of a full-S baseline run (default x2-S).
    python analysis/screen.py [BASELINE]"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from compare import load, paired
r = load(); base = sys.argv[1] if len(sys.argv) > 1 else "x2-S"
b = r[base]["chunks"]  # KL mode ignores --chunks: screens score all chunks
print(f"baseline {base}, all chunks: {b.mean():.5f}")
for n in sorted(k for k in r if k.endswith("-S8")):
    a = r[n]["chunks"]; d, lo, hi = paired(a, b)
    print(f"{n:22s} {a.mean():.5f}  minus baseline {d:+.5f} [{lo:+.5f}, {hi:+.5f}]  top-1 {r[n]['top']:.2f}")
