"""Render the bartowski v6 calibration conversations with the model's own chat template (as its format notes ask)
and write two calibration texts for llama-imatrix --parse-special:
  data/calib-fit.txt   prose paragraphs interleaved with the 137 base conversations
  data/calib-eval.txt  the 36 extension conversations (held out from every fit)
    python scripts/render-calib.py MODEL.gguf"""
import json, re, sys
from pathlib import Path
import gguf, jinja2

ROOT = Path(__file__).resolve().parents[1]
tpl = gguf.GGUFReader(sys.argv[1]).fields["tokenizer.chat_template"].contents()
tpl = re.sub(r"{%-?\s*(end)?generation\s*-?%}", "", tpl)  # HF training-mask tag, no output of its own
env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False, **{a: b for a, b in k.items() if a in ("indent", "sort_keys")})
t = env.from_string(tpl)
d = json.load(open(ROOT / "data/raw/bartowski-v6_conversations.json", encoding="utf-8"))


def render(convs):
    out = []
    for c in convs:
        msgs = []
        for m in c["messages"]:
            m = dict(m)
            for tc in m.get("tool_calls") or []:
                if isinstance(tc.get("function", {}).get("arguments"), str):
                    tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
            msgs.append(m)
        out.append(t.render(messages=msgs, tools=c.get("tools"), add_generation_prompt=False, bos_token="", eos_token=""))
    return "\n".join(out)


prose = (ROOT / "data/raw/bartowski-v6_prose.txt").read_text(encoding="utf-8")
# interleave prose paragraphs and conversations, so a capture capped with --chunks keeps the mix of both
paras = [p for p in prose.split("\n") if p.strip()]
convs = [render([c]) for c in d["conversations"]]
k = max(1, len(paras) // len(convs))
mixed = []
for i, c in enumerate(convs):
    mixed += paras[i * k:(i + 1) * k] + [c]
mixed += paras[len(convs) * k:]
(ROOT / "data/calib-fit.txt").write_text("\n".join(mixed), encoding="utf-8")
(ROOT / "data/calib-eval.txt").write_text(render(d["extension_conversations"]), encoding="utf-8")
for n in ("calib-fit.txt", "calib-eval.txt"):
    print(n, len((ROOT / "data" / n).read_text(encoding="utf-8")), "chars")
