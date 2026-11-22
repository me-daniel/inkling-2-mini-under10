"""Held-out error of GPTQ-Q2_0 with the Gram shrunk toward its diagonal, H = G + a * diag(G), on one layer.
    python scripts/gptq_shrink.py BF16.gguf FIT_DIR EVAL_DIR LAYER a1 a2 ...   (CPU, so it can run beside GPU jobs)"""
import sys
from pathlib import Path
import numpy as np, torch, gguf
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gptq_q2 as g
from gptq_bound import rel_err
torch.set_num_threads(8)
dev = torch.device(sys.argv.pop(5)) if len(sys.argv) > 5 and sys.argv[5] in ("cpu", "mps") else torch.device("cpu")
bf16, fit, ev, layer = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
name = f"blk.{layer}.ffn_down_exps.weight"
t = {x.name: x for x in gguf.GGUFReader(bf16).tensors}[name]
w = (np.asarray(t.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
E, R, C = w.shape
gf, _ = g.load_gram(fit, name, E, C); ge, _ = g.load_gram(ev, name, E, C)
wt, gft, get = torch.from_numpy(w).to(dev), torch.from_numpy(gf), torch.from_numpy(ge)
dg = torch.diag_embed(torch.diagonal(gft, dim1=1, dim2=2))
print("layer\talpha\terr_fit\terr_eval", flush=True)
d, c = g.gptq_q2(wt, dg, 0.0)  # matched control: same Gram diagonal, same scale search, no error feedback
wq = g.dequant(d, c).cpu()
print(f"{layer}\tdiag\t{rel_err(wt.cpu(), wq, gft):.5f}\t{rel_err(wt.cpu(), wq, get):.5f}", flush=True)
for a in map(float, sys.argv[5:]):
    d, c = g.gptq_q2(wt, gft + a * dg, 0.0)
    wq = g.dequant(d, c).cpu()
    print(f"{layer}\t{a}\t{rel_err(wt.cpu(), wq, gft):.5f}\t{rel_err(wt.cpu(), wq, get):.5f}", flush=True)
