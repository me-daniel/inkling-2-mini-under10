"""Copied unchanged from ../inkling-2-mini-quant/scripts/stability.py (same thresholds), with data read from there and
results written here (results/stability/). Stability battery for the upload gate (CHANGELOG.md, 2026-11-21). Thresholds are fixed here, before any candidate.

    python scripts/stability.py NAME MODEL.gguf

Runs llama-server on MODEL at P = 3 and writes results/pilot/stability-NAME.json:
1. GSM8K train items 21-50 (T2 uses 1-20; the test split is reserved), reasoning_effort low, temperature 0, cap 2,048:
   accuracy and the number of replies that hit the cap.
2. Ten 2,000-token generations at temperature 1.0 (seeds 1-10): distinct 4-gram ratio and the longest run of a
   repeated line, the usual signs of a degenerate loop.
3. One prompt of at least 16K tokens (WikiText-2 validation text with a fact planted at 20% depth): is it found.
4. Five tool-call prompts with one function: is a call with valid JSON arguments returned.
5. Two non-English prompts (German, Spanish): replies are non-empty and mostly letters.
A build passes if, against the Q8_0 reference's own results on the same battery: no generation has a distinct 4-gram
ratio below 0.30 or a repeated-line run above 5; the planted fact is found if the reference finds it; valid tool calls
are at most 1 fewer; both non-English replies pass; GSM8K accuracy is at most 4 of 30 below and at most 2 more replies
hit the cap. Thirty items are a smoke test, not an accuracy estimate.
"""

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = (ROOT / "../inkling-2-mini-quant").resolve()  # GSM8K and WikiText parquets live in the sibling repo
BIN = ROOT / "vendor/llama.cpp/build/bin"
PORT = 8092
URL = f"http://127.0.0.1:{PORT}"
THRESHOLDS = {"distinct4_min": 0.30, "repeat_run_max": 5, "tool_drop_max": 1, "gsm8k_drop_max": 4, "cap_rise_max": 2}
NEEDLE = "The archive code for the lighthouse at Point Varn is 4417-QX."
TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["C", "F"]}},
                       "required": ["city"]}}}
TOOL_PROMPTS = ["What's the weather in Lisbon right now?", "Is it cold in Oslo today? Use Celsius.",
                "Check the weather for Nairobi, please.", "I need the current temperature in Denver in Fahrenheit.",
                "What is the weather like in Hanoi?"]
OTHER_LANG = ["Erkläre in drei Sätzen, warum der Himmel blau ist.", "Explica en tres frases por qué el cielo es azul."]
STORY = "Write a long, detailed story about a cartographer who discovers that her maps change overnight."


def chat(messages, **kw):
    body = {"messages": messages, "temperature": 0, "seed": 0, "chat_template_kwargs": {"reasoning_effort": "low"}, **kw}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def last_number(text: str):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return float(nums[-1].replace(",", "")) if nums else None


def distinct4(text: str) -> float:
    w = text.split()
    grams = [tuple(w[i : i + 4]) for i in range(len(w) - 3)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repeat_run(text: str) -> int:
    best = run = 0
    prev = None
    for line in (l.strip() for l in text.splitlines() if l.strip()):
        run = run + 1 if line == prev else 1
        best, prev = max(best, run), line
    return best


def judge(ref: dict, cand: dict, th: dict = THRESHOLDS) -> dict:
    """Applies the docstring thresholds to two stability-*.json results (reference, candidate).

    Returns {"checks": [{"name", "ref", "cand", "rule", "pass"}, ...], "pass": all checks pass}. The two runs must
    cover the same battery (same GSM8K item count, generation count, tool prompts and languages)."""
    for k, size in (("gsm8k", lambda d: d["n"]), ("long_generations", len), ("tool_calls", lambda d: d["n"]),
                    ("other_languages", len)):
        if size(ref[k]) != size(cand[k]):
            raise ValueError(f"{k}: the two runs cover different batteries")
    g_ref, g_cand = ref["gsm8k"], cand["gsm8k"]
    d4 = min(g["distinct4"] for g in cand["long_generations"])
    rr = max(g["repeat_run"] for g in cand["long_generations"])
    langs = sum(x["ok"] for x in cand["other_languages"])
    checks = [
        ("lowest distinct 4-gram ratio over the long generations", min(g["distinct4"] for g in ref["long_generations"]), d4,
         f"no generation below {th['distinct4_min']:.2f}", d4 >= th["distinct4_min"]),
        ("longest repeated-line run over the long generations", max(g["repeat_run"] for g in ref["long_generations"]), rr,
         f"no generation above {th['repeat_run_max']}", rr <= th["repeat_run_max"]),
        ("planted fact found in the long prompt", ref["long_context"]["found"], cand["long_context"]["found"],
         "found if the reference finds it", cand["long_context"]["found"] or not ref["long_context"]["found"]),
        ("valid tool calls", f"{ref['tool_calls']['valid']} of {ref['tool_calls']['n']}",
         f"{cand['tool_calls']['valid']} of {cand['tool_calls']['n']}", f"at most {th['tool_drop_max']} fewer",
         cand["tool_calls"]["valid"] >= ref["tool_calls"]["valid"] - th["tool_drop_max"]),
        ("non-English replies that pass", f"{sum(x['ok'] for x in ref['other_languages'])} of {len(ref['other_languages'])}",
         f"{langs} of {len(cand['other_languages'])}", "all pass", langs == len(cand["other_languages"])),
        ("GSM8K correct", f"{g_ref['correct']} of {g_ref['n']}", f"{g_cand['correct']} of {g_cand['n']}",
         f"at most {th['gsm8k_drop_max']} below", g_cand["correct"] >= g_ref["correct"] - th["gsm8k_drop_max"]),
        ("GSM8K replies that hit the cap", g_ref["capped"], g_cand["capped"], f"at most {th['cap_rise_max']} more",
         g_cand["capped"] <= g_ref["capped"] + th["cap_rise_max"]),
    ]
    out = [dict(name=n, ref=r, cand=c, rule=rule, **{"pass": bool(ok)}) for n, r, c, rule, ok in checks]
    return {"checks": out, "pass": all(c["pass"] for c in out)}


def main() -> None:
    name, model = sys.argv[1], sys.argv[2]
    log = open(ROOT / f"logs/stability-{name}.log", "w")
    server = subprocess.Popen([str(BIN / "llama-server"), "-m", model, "-ngl", "99", "-c", "24576", "-np", "1",
                               "--port", str(PORT), "--override-kv", "inkling2_loop.loop.count=int:3"],
                              stdout=log, stderr=subprocess.STDOUT)
    # repo-relative, so result files carry no home path
    out = {"model": str(Path(model).resolve().relative_to(ROOT)) if Path(model).resolve().is_relative_to(ROOT) else Path(model).name, "thresholds": THRESHOLDS}
    try:
        for _ in range(900):
            try:
                with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                    if json.load(r).get("status") == "ok":
                        break
            except Exception:
                pass
            time.sleep(1)
        else:
            sys.exit("server not ready")

        gsm = pq.read_table(DATA / "data/raw/gsm8k/main/train-00000-of-00001.parquet").to_pylist()[20:50]
        correct = capped = 0
        for item in gsm:
            r = chat([{"role": "user", "content": item["question"] + "\nGive the final answer as a number on the last line."}], max_tokens=2048)
            c = r["choices"][0]
            capped += c.get("finish_reason") == "length"
            want = float(item["answer"].split("####")[-1].strip().replace(",", ""))
            got = last_number(c["message"].get("content") or "")
            correct += got is not None and abs(got - want) < 1e-6
        out["gsm8k"] = {"n": len(gsm), "correct": correct, "capped": capped}

        gens = []
        for seed in range(1, 11):
            r = chat([{"role": "user", "content": STORY}], max_tokens=2000, temperature=1.0, seed=seed,
                     chat_template_kwargs={"reasoning_effort": "none"})
            text = r["choices"][0]["message"].get("content") or ""
            gens.append({"seed": seed, "tokens": r.get("usage", {}).get("completion_tokens"), "distinct4": round(distinct4(text), 4), "repeat_run": repeat_run(text)})
        out["long_generations"] = gens

        val = "".join(pq.read_table(DATA / "data/raw/wikitext/wikitext-2-raw-v1/validation-00000-of-00001.parquet").column("text").to_pylist())
        words = val.split()[:12500]
        cut = len(words) // 5
        doc = " ".join(words[:cut]) + "\n" + NEEDLE + "\n" + " ".join(words[cut:])
        r = chat([{"role": "user", "content": doc + "\n\nWhat is the archive code for the lighthouse at Point Varn? Answer with the code only."}], max_tokens=256)
        out["long_context"] = {"prompt_tokens": r.get("usage", {}).get("prompt_tokens"), "found": "4417-QX" in (r["choices"][0]["message"].get("content") or "")}

        valid = 0
        for p in TOOL_PROMPTS:
            r = chat([{"role": "user", "content": p}], tools=[TOOL], max_tokens=512)
            for call in r["choices"][0]["message"].get("tool_calls") or []:
                try:
                    args = json.loads(call["function"]["arguments"])
                    valid += call["function"]["name"] == "get_weather" and isinstance(args.get("city"), str)
                    break
                except Exception:
                    pass
        out["tool_calls"] = {"n": len(TOOL_PROMPTS), "valid": valid}

        langs = []
        for p in OTHER_LANG:
            text = chat([{"role": "user", "content": p}], max_tokens=512)["choices"][0]["message"].get("content") or ""
            letters = sum(ch.isalpha() for ch in text) / max(len(text), 1)
            langs.append({"chars": len(text), "letter_share": round(letters, 3), "ok": len(text) > 40 and letters > 0.6})
        out["other_languages"] = langs
    finally:
        server.terminate()
        server.wait(timeout=60)

    path = ROOT / f"results/stability/stability-{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
