"""Base model vs fine-tuned model on the same Turkish prompts: side-by-side answers and measurements.

The base model is obtained by disabling the adapter and the fine-tuned model by enabling it, so the
model is loaded only once. Decoding is greedy, so the comparison is deterministic. Besides the
answers, each response is measured for the failure modes that matter for a Turkish assistant:

* answered in Turkish (``bilge.textstats.is_turkish``), and whether it contains CJK characters,
* repetition (share of repeated word 3-grams),
* length in words.

Example::

    python -m bilge.compare --adapter runs/bilge-1.5b-lora/adapter
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prompts import build_messages
from .textstats import has_cjk, is_turkish, repetition, turkish_score, words


def measure(text: str) -> dict:
    return {"turkish": is_turkish(text), "turkish_score": round(turkish_score(text), 3), "cjk": has_cjk(text),
            "repetition": round(repetition(text), 3), "words": len(words(text))}


def summary(rows: list[dict]) -> dict:
    out = {}
    for model in ("base", "tuned"):
        m = [r[model]["metrics"] for r in rows]
        out[model] = {"turkish_rate": sum(x["turkish"] for x in m) / len(m),
                      "cjk_rate": sum(x["cjk"] for x in m) / len(m),
                      "mean_repetition": sum(x["repetition"] for x in m) / len(m),
                      "mean_words": sum(x["words"] for x in m) / len(m)}
    return out


def render(rows, stats, model_name, adapter) -> str:
    lines = ["# Base model vs fine-tuned model (Bilge)", "",
             f"Base model: `{model_name}` · Adapter: `{adapter}` · Decoding: greedy", "",
             "| Model | Answers in Turkish | Answers with CJK characters | Mean repetition | Mean length (words) |",
             "|---|---|---|---|---|"]
    for name, s in (("Base", stats["base"]), ("Bilge", stats["tuned"])):
        lines.append(f"| {name} | {s['turkish_rate']:.0%} | {s['cjk_rate']:.0%} | {s['mean_repetition']:.3f} | "
                     f"{s['mean_words']:.0f} |")
    lines.append("")
    for i, r in enumerate(rows, 1):
        lines += [f"## {i}. {r['category']}", "", f"**Prompt:** {r['instruction']}", "",
                  "**Base model:**", "", r["base"]["text"], "", "**Bilge (fine-tuned):**", "", r["tuned"]["text"], "",
                  "---", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Base vs fine-tuned model comparison")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", required=True)
    p.add_argument("--prompts", default="prompts/evaluation_prompts.json")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--out", default="runs/comparison")
    a = p.parse_args(argv)

    from peft import PeftModel

    from .modeling import generate, load_base_model, load_tokenizer

    tok = load_tokenizer(a.model)
    model = PeftModel.from_pretrained(load_base_model(a.model, four_bit=True), a.adapter).eval()
    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))
    rows = []
    for i, item in enumerate(prompts, 1):
        msgs = build_messages(item["instruction"], item.get("input", ""))
        with model.disable_adapter():
            base = generate(model, tok, msgs, a.max_new_tokens, temperature=0)
        tuned = generate(model, tok, msgs, a.max_new_tokens, temperature=0)
        rows.append({"category": item.get("category", ""), "instruction": item["instruction"],
                     "base": {"text": base, "metrics": measure(base)},
                     "tuned": {"text": tuned, "metrics": measure(tuned)}})
        print(f"[{i}/{len(prompts)}] done")
    stats = summary(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.md").write_text(render(rows, stats, a.model, a.adapter), encoding="utf-8")
    (out / "comparison.json").write_text(json.dumps({"summary": stats, "rows": rows}, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved: {out}/comparison.md")


if __name__ == "__main__":
    main()
