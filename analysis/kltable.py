"""Collects every llama-perplexity KL log in logs/ into results/kl.tsv (one row per run, per-chunk values).

The chunk lines of a KL log hold running means; every chunk scores the same number of positions, so the mean of
chunk k is k * m_k - (k - 1) * m_(k-1). Every chunk line must parse (checked against 'computing over N chunks').
The seconds column is nan: llama-perplexity logs carry no total time (wall time is in the chain logs).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM = r"-?(?:[\d.]+|nan|inf)"  # llama-perplexity prints nan for an undefined SE (e.g. chunks 1-2 of identical runs)
CHUNK = re.compile(rf"^\s*(\d+)\s+{NUM}\s+±\s+{NUM}\s+{NUM}\s+±\s+{NUM}\s+([\d.]+)\s+±\s+{NUM}\s+{NUM}\s+±\s+{NUM}\s+%\s+([\d.]+)\s+±")


def parse(log: Path) -> dict | None:
    text = log.read_text(errors="replace")
    mean = re.search(r"Mean\s+KLD:\s+([\d.]+)\s+±\s+([\d.]+)", text)
    if not mean:
        return None
    running = [(int(m.group(1)), float(m.group(2)), float(m.group(3))) for m in map(CHUNK.match, text.splitlines()) if m]
    # the telescoping below needs every chunk; a skipped line would silently shift the next chunk's value
    n = re.search(r"computing over (\d+) chunks", text)
    want = list(range(1, int(n.group(1)) + 1)) if n else list(range(1, len(running) + 1))
    if [k for k, _, _ in running] != want:
        raise ValueError(f"{log.name}: chunk lines {[k for k, _, _ in running]} != 1..{len(want)}")
    per_chunk, prev = [], 0.0
    for k, m, _ in running:
        per_chunk.append(k * m - (k - 1) * prev)
        prev = m
    top = re.search(r"Same top p:\s+([\d.]+)\s+±\s+([\d.]+)", text)
    total = re.search(r"total time\s*=\s*([\d.]+) ms", text)
    return {
        "run": log.stem.removeprefix("ppl-"),
        "mean_kl": float(mean.group(1)),
        "se": float(mean.group(2)),
        "same_top_p": float(top.group(1)) if top else float("nan"),
        "seconds": float(total.group(1)) / 1000 if total else float("nan"),
        "per_chunk": per_chunk,
    }


def model_state(log: Path) -> str:
    """ok / stale / gone / unsigned: does the model file the .sig names still have the signed size and mtime?
    Added 23 Nov after the code reviews: a score whose model was rebuilt later must not pass for the new file."""
    sig = log.with_name(log.name + ".sig")
    if not sig.exists():
        return "unsigned"
    f = sig.read_text().split()
    if f and f[0] == "v2":
        f = f[1:]
    try:
        path, size, mtime = f[0], int(f[1]), int(f[2])
    except (IndexError, ValueError):
        return "unsigned"
    if not os.path.exists(path):
        return "gone"
    st = os.stat(path)
    return "ok" if (st.st_size, int(st.st_mtime)) == (size, mtime) else "stale"


def collect() -> list[dict]:
    rows = []
    for log in sorted((ROOT / "logs").glob("ppl-*.log")):
        if r := parse(log):
            r["model_state"] = model_state(log)
            rows.append(r)
    out = ROOT / "results/kl.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("run\tmean_kl\tse\tsame_top_p\tseconds\tmodel_state\tper_chunk\n")
        for r in rows:
            f.write(f"{r['run']}\t{r['mean_kl']:.6f}\t{r['se']:.6f}\t{r['same_top_p']:.3f}\t{r['seconds']:.1f}\t{r['model_state']}\t"
                    + ",".join(f"{x:.6f}" for x in r["per_chunk"]) + "\n")
    return rows


if __name__ == "__main__":
    for r in collect():
        print(f"{r['run']:28s} KL {r['mean_kl']:.5f} ± {r['se']:.5f}  top-1 {r['same_top_p']:.2f}%  {r['seconds']:.0f} s  chunks {len(r['per_chunk'])}")
