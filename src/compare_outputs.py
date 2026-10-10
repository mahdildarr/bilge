"""Side-by-side comparison of the base model and the fine-tuned model on the same Turkish prompts.

The base model is obtained by disabling the adapter and the fine-tuned model by enabling it, so the model
is loaded only once.

Example:
    python src/compare_outputs.py --adapter runs/bilge-1.5b-lora/adapter
"""

import argparse
import json
import os

from peft import PeftModel

from common import build_messages, generate, load_base_model, load_tokenizer


def main():
    p = argparse.ArgumentParser(description="Base vs. fine-tuned model comparison")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", required=True)
    p.add_argument("--prompts", default="prompts/evaluation_prompts.json")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--out", default="runs/comparison.md")
    a = p.parse_args()

    tok = load_tokenizer(a.model)
    model = PeftModel.from_pretrained(load_base_model(a.model, four_bit=True), a.adapter)
    model.eval()
    prompts = json.load(open(a.prompts, encoding="utf-8"))

    lines = ["# Base model vs. fine-tuned model (Bilge)", "",
             f"Base model: `{a.model}` · Adapter: `{a.adapter}` · Decoding: greedy (temperature 0)", ""]
    for i, item in enumerate(prompts, 1):
        msgs = build_messages(item["instruction"], item.get("input", ""))
        with model.disable_adapter():
            base = generate(model, tok, msgs, a.max_new_tokens, temperature=0)
        tuned = generate(model, tok, msgs, a.max_new_tokens, temperature=0)
        lines += [f"## {i}. {item.get('category', '')}", "", f"**Prompt:** {item['instruction']}", "",
                  "**Base model:**", "", base, "", "**Bilge (fine-tuned):**", "", tuned, "", "---", ""]
        print(f"[{i}/{len(prompts)}] done")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"Comparison saved: {a.out}")


if __name__ == "__main__":
    main()
