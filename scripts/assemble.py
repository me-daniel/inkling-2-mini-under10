"""Builds one allocation by byte-copying tensors from donor GGUFs (scripts/splice.py), after checking its size.

    python scripts/assemble.py SPEC.json OUT.gguf [--dry]

SPEC: {"gate_up": {"0-5": "iq2_xxs", ...}, "down": {"0-11": "gptq", ...}, "ne": {"attn": "q5_k", "embd": "q4_k", "out": "q6_k"}}
("ud": the per-tensor types of UD-IQ2_XXS, encoded by our llama-quantize: pools/ne-ud.gguf.) Every layer 0-23 needs a gate_up and a down choice. Levels and their donors are in DONORS below. Tensors not
taken keep the base file's encoding (pool-3bit: Q8_0 non-expert matrices, F32 vectors and loop tensors).
Prints the predicted file size (tensor bytes plus the base file's header and padding overhead) and refuses to build
above LIMIT unless --over is given.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import gguf

ROOT = Path(__file__).resolve().parents[1]
SIB = (ROOT / "../inkling-2-mini-quant").resolve()
M, SM = ROOT / "models", SIB / "models"
LIMIT = 8_486_236_352  # unsloth UD-IQ2_XXS, CHANGELOG 2026-11-21
BASE = SM / "pool-3bit.gguf"
ATT = r"attn_(q|k|v|output|gate|rel|sink_proj|k_conv_proj|v_conv_proj)\.weight$"

GATE_UP = {"iq3_xxs": SM / "pool-3bit.gguf", "iq2_xxs": SM / "pool-2bit.gguf", "iq2_s": M / "pools/gu-iq2_s.gguf",
           "iq2_xs": M / "pools/gu-iq2_xs.gguf", "iq1_m": M / "pools/gu-iq1_m.gguf", "iq1_s": M / "pools/gu-iq1_s.gguf",
           # Unsloth's own gate_up tensors (mainline layout, sha256-checked); their per-expert .scale is appended
           "ud": M / "ext/mainline/UD-IQ2_XXS.gguf",
           # M0's per-layer types in one pool, quantized with another importance matrix (scripts/pool-map.sh)
           "map-bart": M / "pools/map-bart.gguf", "map-mine": M / "pools/map-mine.gguf",
           # mean-aware scale refit of the llama-quantize pools (scripts/meanfit.py)
           "rf-iq1_s": M / "meanfit/rf-iq1_s.gguf", "rf-iq2_xxs": M / "meanfit/rf-iq2_xxs.gguf",
           "rf-iq2_xs": M / "meanfit/rf-iq2_xs.gguf", "rf-iq2_s": M / "meanfit/rf-iq2_s.gguf"}
UD_SCALE = "ffn_gate_up_exps.scale"
DOWN = {"iq4_nl": lambda l: SM / "pool-3bit.gguf", "q2opt": lambda l: SM / f"q2opt/L{l}.gguf",
        "diag": lambda l: M / f"down/diag-L{l}.gguf", "gptq": lambda l: M / f"down/gptq-L{l}.gguf"}
NE = {"udfile": M / "ext/mainline/UD-IQ2_XXS.gguf",  # Unsloth's own non-expert tensors (same names in mainline layout)
      "ud": M / "pools/ne-ud.gguf", "q8_0": SM / "pool-3bit.gguf", "q6_k": SM / "cand.gguf", "q5_k": M / "pools/ne-q5.gguf", "q4_k": M / "pools/ne-q4.gguf"}
NE_RE = {"all": r"^(?!.*_exps\.)",  # every non-expert tensor (never a .scale of an expert)
         "attn": rf"^blk\.\d+\.{ATT}", "embd": r"^token_embd\.weight$", "out": r"^output\.weight$"}


def layers(spec: dict) -> dict[int, str]:
    out = {}
    for k, v in spec.items():
        lo, _, hi = k.partition("-")
        for l in range(int(lo), int(hi or lo) + 1):
            out[l] = v
    if sorted(out) != list(range(24)):
        raise SystemExit(f"layers {sorted(out)} do not cover 0-23 exactly once")
    return out


_sizes: dict[Path, dict[str, int]] = {}


def nbytes(path: Path, name: str) -> int:
    if path not in _sizes:
        _sizes[path] = {t.name: int(t.n_bytes) for t in gguf.GGUFReader(path).tensors}
    return _sizes[path][name]


def _names(path: Path) -> set[str]:
    if path not in _sizes:
        _sizes[path] = {t.name: int(t.n_bytes) for t in gguf.GGUFReader(path).tensors}
    return set(_sizes[path])


def plan(spec: dict) -> tuple[list[tuple[Path, str]], int, list[tuple[Path, str]]]:
    takes, size, adds = [], 0, []
    gu, dn = layers(spec["gate_up"]), layers(spec["down"])
    taken = {}
    for l in range(24):
        n = f"blk.{l}.ffn_gate_up_exps.weight"
        takes.append((GATE_UP[gu[l]], rf"^blk\.{l}\.ffn_gate_up_exps\.weight$")); taken[n] = GATE_UP[gu[l]]
        if gu[l] == "ud" and f"blk.{l}.{UD_SCALE}" in _names(GATE_UP["ud"]):
            adds.append((GATE_UP["ud"], rf"^blk\.{l}\.ffn_gate_up_exps\.scale$"))
            size += nbytes(GATE_UP["ud"], f"blk.{l}.{UD_SCALE}")
        n = f"blk.{l}.ffn_down_exps.weight"
        takes.append((DOWN[dn[l]](l), rf"^blk\.{l}\.ffn_down_exps\.weight$")); taken[n] = DOWN[dn[l]](l)
    import re
    ne = spec.get("ne", {})
    base_names = [t.name for t in gguf.GGUFReader(BASE).tensors]
    for group, lvl in ne.items():
        takes.append((NE[lvl], NE_RE[group]))
        for n in base_names:
            if re.search(NE_RE[group], n):
                taken[n] = NE[lvl]
    for n in base_names:
        size += nbytes(taken.get(n, BASE), n)
    overhead = BASE.stat().st_size - sum(nbytes(BASE, n) for n in base_names)
    return takes, size + overhead, adds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--over", action="store_true")
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    takes, size, adds = plan(spec)
    print(f"{Path(args.spec).stem}: predicted {size:,} bytes, limit {LIMIT:,}, margin {LIMIT - size:+,}")
    if args.dry:
        return
    if size > LIMIT and not args.over:
        raise SystemExit("over the limit")
    cmd = [sys.executable, "-I", str(SIB / "scripts/splice.py"), str(BASE), args.out]
    for donor, rx in takes:
        cmd += ["--take", str(donor), rx]
    for donor, rx in adds:
        cmd += ["--add", str(donor), rx]
    subprocess.run(cmd + ["--verify"], check=True, stdout=open(args.out + ".splice.log", "w"))
    real = Path(args.out).stat().st_size
    print(f"built {args.out}: {real:,} bytes (predicted {size:,})")
    if real > LIMIT and not args.over:
        Path(args.out).unlink()
        raise SystemExit("built file is over the limit; deleted")


if __name__ == "__main__":
    main()
