"""Model loading and generation shared by training, evaluation and the demo."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def compute_dtype():
    """bf16 where the GPU supports it (Ampere and newer), fp16 otherwise (e.g. the T4)."""
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def load_base_model(name, four_bit=True):
    """Load the base model in 4-bit NF4 (QLoRA) or in 16 bit."""
    if four_bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype(),
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
    return AutoModelForCausalLM.from_pretrained(name, torch_dtype=compute_dtype(), device_map="auto")


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
