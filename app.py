"""Bilge: Gradio chat interface for the fine-tuned Turkish AI assistant.

Example:
    python app.py --adapter runs/bilge-1.5b-lora/adapter --share
"""

import argparse
import sys

import gradio as gr
from peft import PeftModel

sys.path.insert(0, "src")
from common import SYSTEM_PROMPT, generate, load_base_model, load_tokenizer  # noqa: E402


def to_messages(message, history):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        if isinstance(turn, dict):  # Gradio "messages" format
            msgs.append({"role": turn["role"], "content": turn["content"]})
        else:  # legacy (user, assistant) pair format
            user, bot = turn
            msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": bot}]
    msgs.append({"role": "user", "content": message})
    return msgs


def main():
    p = argparse.ArgumentParser(description="Bilge chat interface")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", required=True)
    p.add_argument("--share", action="store_true", help="Create a public link (useful on Colab)")
    a = p.parse_args()

    tok = load_tokenizer(a.model)
    model = PeftModel.from_pretrained(load_base_model(a.model, four_bit=True), a.adapter).eval()

    def chat(message, history):
        return generate(model, tok, to_messages(message, history), max_new_tokens=384)

    gr.ChatInterface(
        chat,
        type="messages",
        title="Bilge · Turkish AI Assistant",
        description=f"Base model: {a.model} · fine-tuned with QLoRA on Turkish instruction data. Ask in Turkish.",
        examples=["Yapay zekâ ile makine öğrenmesi arasındaki farkı basitçe açıkla.",
                  "İnsansız hava araçlarının sivil kullanım alanlarını listele.",
                  "Bir e-postayı resmi bir dille yeniden yaz: Toplantıya gelemiyorum."],
    ).launch(share=a.share)


if __name__ == "__main__":
    main()
