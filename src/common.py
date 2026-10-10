"""Shared model loading and prompt formatting for training, evaluation and the demo."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Bilge is a Turkish assistant, so its system prompt is in Turkish:
# "You are Bilge, a helpful, accurate and clear-spoken Turkish AI assistant."
SYSTEM_PROMPT = "Sen Bilge adında, yardımsever, doğru ve açık konuşan bir Türkçe yapay zekâ asistanısın."


def build_messages(instruction, inp=""):
    user = instruction.strip()
    if inp and inp.strip():
        user += "\n\n" + inp.strip()
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(name, four_bit=True):
    """Load the base model in 4-bit (QLoRA) or 16-bit. fp16 is used because the T4 GPU does not support bf16."""
    if four_bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
    return AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")


@torch.no_grad()
def generate(model, tok, messages, max_new_tokens=256, temperature=0.7):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
