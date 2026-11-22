"""Summarises a GGUF for Addendum G: bytes, loop count, tensor types by role, MTP tensors, and tensors Metal cannot multiply.

    python scripts/gguf-types.py FILE.gguf
"""

import re
import sys
from collections import Counter

import gguf

OFF_METAL = ("NVFP4", "F8_E4M3", "TQ1_0")


def main() -> None:
    r = gguf.GGUFReader(sys.argv[1])
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    loop = r.fields.get(f"{arch}.loop.count")
    by = Counter()
    for t in r.tensors:
        role = re.sub(r"^blk\.\d+\.", "", t.name)
        by[(role if "exps" in role or role.startswith(("token_embd", "output")) else "other", t.tensor_type.name)] += int(t.n_bytes)
    print(f"arch {arch}  loop.count {loop.contents() if loop else 'absent'}  tensor bytes {sum(int(t.n_bytes) for t in r.tensors):,}")
    print("mtp tensors:", sum("mtp" in t.name or "nextn" in t.name for t in r.tensors))
    print("tensors in types Metal cannot multiply:", sum(t.tensor_type.name in OFF_METAL for t in r.tensors), "(experts:", sum(t.tensor_type.name in OFF_METAL for t in r.tensors if "exps" in t.name), ")")
    for (role, ty), n in sorted(by.items(), key=lambda x: -x[1]):
        print(f"{n / 1e9:8.3f} GB  {ty:8s} {role}")


if __name__ == "__main__":
    main()
