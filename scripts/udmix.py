"""All-Metal mixes of Unsloth-encoded expert tensors, built by byte copy on top of UD-IQ2_XXS (mainline layout).

    python scripts/udmix.py SPEC.json OUT.gguf [--dry] [--over]

SPEC: {"gate_up": {"1": "UD3", ...}, "down": {"18": "UD1M", ...}}. Layers not listed keep UD-IQ2_XXS's tensor, except
that every NVFP4 down projection (which Metal cannot multiply) becomes IQ4_NL from our llama-quantize pool, as in X2.
Sources: UD2 = UD-IQ2_XXS, UD1M = UD-IQ1_M, UD3 = UD-IQ3_XXS (all sha256-checked mainline files), iq4nl = ours.
A tensor taken from another UD file brings its own per-expert .scale; an IQ4_NL tensor is unscaled, so the base's
.scale for it is dropped. Refuses NVFP4 in the result, and a file above LIMIT unless --over.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import gguf

ROOT = Path(__file__).resolve().parents[1]
SIB = (ROOT / "../inkling-2-mini-quant").resolve()
LIMIT = 8_486_236_352
SRC = {"UD2": ROOT / "models/ext/mainline/UD-IQ2_XXS.gguf", "UD1M": ROOT / "models/ext/mainline/UD-IQ1_M.gguf",
       "UD3": SIB / "models/UD-IQ3_XXS.gguf", "iq4nl": SIB / "models/pool-3bit.gguf"}
_t: dict[str, dict] = {}


def tensors(k: str) -> dict:
    if k not in _t:
        _t[k] = {t.name: t for t in gguf.GGUFReader(SRC[k]).tensors}
    return _t[k]


def plan(spec: dict) -> tuple[list[str], dict[str, tuple[str, str]]]:
    """-> splice args, and the final {tensor: (source, type)} for every expert tensor and scale."""
    base = tensors("UD2")
    final = {n: ("UD2", t.tensor_type.name) for n, t in base.items() if "_exps." in n}
    args: list[str] = []
    choice = {("ffn_gate_up_exps", int(l)): s for l, s in spec.get("gate_up", {}).items()}
    choice |= {("ffn_down_exps", int(l)): s for l, s in spec.get("down", {}).items()}
    for l in range(24):
        for kind in ("ffn_gate_up_exps", "ffn_down_exps"):
            w, sc = f"blk.{l}.{kind}.weight", f"blk.{l}.{kind}.scale"
            src = choice.get((kind, l), "UD2")
            if src == "UD2" and base[w].tensor_type.name == "NVFP4":
                src = "iq4nl"
            if src == "UD2":
                continue
            t = tensors(src)[w]
            args += ["--take", str(SRC[src]), rf"^blk\.{l}\.{kind}\.weight$"]
            final[w] = (src, t.tensor_type.name)
            has_src, has_base = sc in tensors(src), sc in base
            if has_src and has_base:
                args += ["--take", str(SRC[src]), rf"^blk\.{l}\.{kind}\.scale$"]; final[sc] = (src, "F32")
            elif has_base:
                args += ["--drop", rf"^blk\.{l}\.{kind}\.scale$"]; final.pop(sc)
            elif has_src:
                args += ["--add", str(SRC[src]), rf"^blk\.{l}\.{kind}\.scale$"]; final[sc] = (src, "F32")
    return args, final


def size(final: dict) -> int:
    base = tensors("UD2")
    total = SRC["UD2"].stat().st_size
    for n, t in base.items():
        if "_exps." in n:
            total -= int(t.n_bytes) if n not in final or final[n][0] != "UD2" else 0
    for n, (src, _) in final.items():
        if src != "UD2":
            total += int(tensors(src)[n].n_bytes)
    return total  # header deltas of dropped/added scale records are ignored (a few hundred bytes); real size is checked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("out")
    ap.add_argument("--dry", action="store_true"); ap.add_argument("--over", action="store_true")
    a = ap.parse_args()
    args, final = plan(json.loads(Path(a.spec).read_text()))
    if any(ty == "NVFP4" for _, ty in final.values()):
        raise SystemExit("NVFP4 left in the result")
    est = size(final)
    print(f"{Path(a.spec).stem}: about {est:,} bytes, limit {LIMIT:,}, margin {LIMIT - est:+,}")
    if a.dry:
        return
    if est > LIMIT + 4096 and not a.over:
        raise SystemExit("over the limit")
    subprocess.run([sys.executable, "-I", str(SIB / "scripts/splice.py"), str(SRC["UD2"]), a.out, *args, "--verify"],
                   check=True, stdout=open(a.out + ".json", "w"))
    real = Path(a.out).stat().st_size
    print(f"built {a.out}: {real:,} bytes")
    if real > LIMIT and not a.over:
        Path(a.out).unlink()
        raise SystemExit("built file is over the limit; deleted")


if __name__ == "__main__":
    main()
