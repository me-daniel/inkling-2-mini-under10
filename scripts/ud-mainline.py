"""Rebuild an Unsloth Mainline_Support GGUF (inkling2_loop layout) from the root file of the same quant.

    python scripts/ud-mainline.py ROOT.gguf REPO REMOTE_PATH OUT.gguf [--checks 24]

Unsloth's README says both files hold the same tensors byte for byte, with different metadata and tensor names.
This fetches the remote header (one range request), copies every tensor whose name, type and shape match from the
root file, fetches the rest (the renamed loop tensors) from the remote file, and then compares
--checks random 64 KiB ranges of the output with the same ranges of the remote file. Any mismatch deletes OUT.
"""

import argparse
import random
import struct
import sys
import urllib.request
from pathlib import Path

import gguf
import numpy as np


def fetch(url: str, a: int, n: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{a + n - 1}"})
    return urllib.request.urlopen(req, timeout=120).read()


def parse_header(b: bytes):
    o = 4
    def rd(f):
        nonlocal o
        v = struct.unpack_from("<" + f, b, o); o += struct.calcsize("<" + f); return v[0]
    def rs():
        nonlocal o
        n = rd("Q"); s = b[o:o + n].decode(); o += n; return s
    sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    def skip(t):
        nonlocal o
        if t == 8: rs(); return
        if t == 9:
            et = rd("I"); n = rd("Q")
            if et in sz: o += sz[et] * n
            else:
                for _ in range(n): skip(et)
            return
        o += sz[t]
    rd("I"); nt = rd("Q"); nkv = rd("Q"); align = 32
    for _ in range(nkv):
        k = rs(); t = rd("I")
        if k == "general.alignment": align = rd("I")
        else: skip(t)
    tensors = []
    for _ in range(nt):
        n = rs(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; t = rd("I"); off = rd("Q")
        tensors.append((n, dims, t, off))
    return tensors, (o + align - 1) // align * align


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("repo"); ap.add_argument("remote"); ap.add_argument("out")
    ap.add_argument("--checks", type=int, default=24)
    args = ap.parse_args()
    url = f"https://huggingface.co/{args.repo}/resolve/main/{args.remote}"
    size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60).headers["Content-Length"])
    head = fetch(url, 0, 48 * 2**20)
    remote, data0 = parse_header(head)
    root = gguf.GGUFReader(args.root)
    rt = {t.name: t for t in root.tensors}
    raw = np.memmap(args.root, dtype=np.uint8, mode="r")
    part = Path(args.out + ".part")
    local = fetched = 0
    ends = sorted(off for _, _, _, off in remote) + [size - data0]
    with open(part, "wb") as f:
        f.write(head[:data0])
        for n, dims, ty, off in remote:
            t = rt.get(n)
            f.seek(data0 + off)
            if t is not None and int(t.tensor_type) == ty and [int(x) for x in t.shape] == dims:
                f.write(raw[t.data_offset:t.data_offset + int(t.n_bytes)].tobytes()); local += int(t.n_bytes)
            else:  # renamed or split loop tensors: take them from the remote file (up to the next tensor's offset)
                nxt = ends[ends.index(off) + 1]
                f.write(fetch(url, data0 + off, nxt - off)); fetched += nxt - off
        f.truncate(size)
    print(f"copied {local:,} bytes from the root file, fetched {fetched:,} bytes", file=sys.stderr)
    out = np.memmap(part, dtype=np.uint8, mode="r")
    rng = random.Random(0)
    for a in [0, data0, size - 65536] + [rng.randrange(0, size - 65536) for _ in range(args.checks)]:
        if fetch(url, a, 65536) != out[a:a + 65536].tobytes():
            part.unlink()
            raise SystemExit(f"byte mismatch at offset {a}; deleted")
    part.rename(args.out)
    print(f"{args.out}: {size:,} bytes, {len(remote)} tensors, {args.checks + 3} remote ranges identical")


if __name__ == "__main__":
    main()
