"""Builds a GGUF from a base file, taking selected tensors byte for byte from donor files.

    python scripts/splice.py BASE OUT [--take DONOR REGEX ...] [--add DONOR REGEX ...] [--slot DONOR PHYS SLOT ...] [--verify]

The header and metadata are copied verbatim from BASE, so nothing is re-encoded. A tensor whose name matches
a --take regex (first match wins) comes from that donor with its own type; logical shapes must agree. Each
tensor-info record is copied verbatim from the tensor's source except its data offset. With no options the
output is byte-identical to BASE. --add appends, byte for byte and in donor order, every donor tensor whose name
matches REGEX; each must be absent from BASE and from earlier appends (e.g. the per-expert .scale tensors that
native NVFP4 experts need, which --take cannot bring in). --slot appends the donor's routed experts of layer PHYS (and their .scale,
if any) under the name of inkling2_loop cache slot SLOT, which then uses them in that pass only (patch 0003).
--verify re-reads OUT and compares every tensor with its source.
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

import gguf


SLOT_TENSORS = ("ffn_gate_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_up_exps.scale", "ffn_down_exps.scale")


def tensor_info_bytes(t: gguf.ReaderTensor, offset: int, name: str | None = None) -> bytes:
    # parts: name length, name, n_dims, dims, type, offset
    parts = t.field.parts
    if name is None:
        head = parts[0].tobytes() + parts[1].tobytes()
    else:
        head = struct.pack("<Q", len(name.encode())) + name.encode()
    return head + b"".join(p.tobytes() for p in parts[2:-1]) + struct.pack("<Q", offset)


def pad(n: int, align: int) -> int:
    return (align - n % align) % align


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--take", nargs=2, action="append", default=[], metavar=("DONOR", "REGEX"))
    ap.add_argument("--add", nargs=2, action="append", default=[], metavar=("DONOR", "REGEX"))
    ap.add_argument("--slot", nargs=3, action="append", default=[], metavar=("DONOR", "PHYS", "SLOT"))
    ap.add_argument("--drop", action="append", default=[], metavar="REGEX",
                    help="leave out base tensors matching REGEX (e.g. a .scale whose weight is replaced by a pre-scaled one)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    base = gguf.GGUFReader(args.base)
    if base.endianess != gguf.GGUFEndian.LITTLE:
        sys.exit("only little-endian GGUF is supported")
    donors = [(re.compile(rx), {t.name: t for t in gguf.GGUFReader(path).tensors}, path) for path, rx in args.take]

    drops = [re.compile(rx) for rx in args.drop]
    for rx in drops:
        if not any(rx.search(t.name) for t in base.tensors):
            sys.exit(f"--drop {rx.pattern} matches nothing in base")
    plan: list[tuple[gguf.ReaderTensor, str]] = []
    for t in base.tensors:
        if any(rx.search(t.name) for rx in drops):
            continue
        src, src_path = t, "base"
        for rx, names, path in donors:
            if rx.search(t.name):
                if t.name not in names:
                    sys.exit(f"{t.name} matches {rx.pattern} but is missing in {path}")
                src, src_path = names[t.name], path
                if list(src.shape) != list(t.shape):
                    sys.exit(f"{t.name}: shape {list(src.shape)} in {path} != {list(t.shape)} in base")
                break
        plan.append((src, src_path, None))

    base_names = {t.name for t in base.tensors if not any(rx.search(t.name) for rx in drops)}
    out_names = set(base_names)
    for path, rx in args.add:
        pat = re.compile(rx)
        hits = [t for t in gguf.GGUFReader(path).tensors if pat.search(t.name)]
        if not hits:
            sys.exit(f"--add: {rx} matches no tensor in {path}")
        for t in hits:
            if t.name in out_names:
                sys.exit(f"--add: {t.name} from {path} already exists")
            out_names.add(t.name)
            plan.append((t, path, None))

    for path, phys, slot in args.slot:
        names = {t.name: t for t in gguf.GGUFReader(path).tensors}
        for suffix in SLOT_TENSORS:
            src_name, dst_name = f"blk.{int(phys)}.{suffix}", f"blk.{int(slot)}.{suffix}"
            if src_name not in names:
                if suffix.endswith(".weight"):
                    sys.exit(f"--slot: {src_name} missing in {path}")
                continue
            if dst_name in out_names:
                sys.exit(f"--slot: {dst_name} already exists")
            out_names.add(dst_name)
            plan.append((names[src_name], path, dst_name))

    align_field = base.fields.get(gguf.Keys.General.ALIGNMENT)
    align = int(align_field.contents()) if align_field is not None else gguf.GGUF_DEFAULT_ALIGNMENT
    kv_end = int(base.tensors[0].field.offset)
    head = bytearray(base.data[:kv_end].tobytes())
    # header: magic (4), version (4), tensor count (8), kv count (8)
    struct.pack_into("<Q", head, 8, len(plan))

    offsets, cur = [], 0
    for src, _, _ in plan:
        offsets.append(cur)
        cur += src.n_bytes + pad(src.n_bytes, align)

    with open(args.out, "wb") as f:
        f.write(head)
        for (src, _, name), off in zip(plan, offsets):
            f.write(tensor_info_bytes(src, off, name))
        f.write(b"\0" * pad(f.tell(), align))
        for src, _, _ in plan:
            f.write(memoryview(np.ascontiguousarray(src.data)).cast("B"))
            f.write(b"\0" * pad(src.n_bytes, align))

    counts: dict[str, int] = {}
    for i, (_, p, name) in enumerate(plan):
        key = f"{p} (as slot tensors)" if name is not None else f"{p} (added)" if i >= len(base.tensors) else p
        counts[key] = counts.get(key, 0) + 1
    manifest = {"base": args.base, "take": args.take, "add": args.add, "drop": args.drop, "slot": args.slot, "tensors_from": counts}

    if args.verify:
        out = gguf.GGUFReader(args.out)
        if len(out.tensors) != len(plan):
            sys.exit(f"verify: {len(out.tensors)} tensors in output, expected {len(plan)}")
        for o, (src, p, name) in zip(out.tensors, plan):
            want = src.name if name is None else name
            if o.name != want or o.tensor_type != src.tensor_type or not np.array_equal(o.data, src.data):
                sys.exit(f"verify: {o.name} differs from its source {p}")
        manifest["verified"] = True

    Path(args.out + ".json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
