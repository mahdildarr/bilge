"""QLoRA fine-tuning of an open-source language model on Turkish instruction data.

- The base model is quantized to 4-bit; only the LoRA adapter weights are trained.
- The loss is computed on the assistant answer only; the prompt is masked.
- Perplexity is measured on the same held-out validation set before and after training. Because LoRA's
  B matrix is initialized to zero, the pre-training measurement equals the base model's performance.

Example (Colab T4):
    python src/train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset TFLai/Turkish-Alpaca \
        --train-samples 5000 --eval-samples 500 --epochs 1
"""

import argparse
import json
import math
import os

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from common import build_messages, load_base_model, load_tokenizer

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tuning on Turkish instruction data")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dataset", default="TFLai/Turkish-Alpaca", help="Dataset with instruction / input / output columns")
    p.add_argument("--train-samples", type=int, default=5000)
    p.add_argument("--eval-samples", type=int, default=500)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--out", default="runs/bilge-1.5b-lora")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_tokenize(tok, max_len):
    def fn(ex):
        prompt = tok.apply_chat_template(build_messages(ex["instruction"], ex.get("input", "")), tokenize=False, add_generation_prompt=True)
        full = prompt + ex["output"].strip() + tok.eos_token
        p_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
        ids = tok(full, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
        labels = [-100] * min(p_len, len(ids)) + ids[p_len:]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    return fn


def main():
    a = parse_args()
    tok = load_tokenizer(a.model)

    ds = load_dataset(a.dataset, split="train").shuffle(seed=a.seed)
    ds = ds.filter(lambda ex: len(ex["output"].strip()) > 0)
    n_eval = min(a.eval_samples, len(ds) // 10)
    eval_ds = ds.select(range(n_eval))
    train_ds = ds.select(range(n_eval, min(n_eval + a.train_samples, len(ds))))
    cols = train_ds.column_names
    tokenize = make_tokenize(tok, a.max_len)
    has_answer = lambda ex: any(l != -100 for l in ex["labels"])  # noqa: E731  (drops examples whose prompt fills the whole window)
    train_ds = train_ds.map(tokenize, remove_columns=cols).filter(has_answer)
    eval_ds = eval_ds.map(tokenize, remove_columns=cols).filter(has_answer)
    print(f"Training examples: {len(train_ds)} | Validation examples: {len(eval_ds)}")

    model = load_base_model(a.model, four_bit=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="no",
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=a.seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))

    before = trainer.evaluate()["eval_loss"]
    print(f"Validation loss before training: {before:.4f} (perplexity {math.exp(before):.2f})")
    train_out = trainer.train()
    after = trainer.evaluate()["eval_loss"]
    print(f"Validation loss after training: {after:.4f} (perplexity {math.exp(after):.2f})")

    adapter_dir = os.path.join(a.out, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    metrics = {
        "base_model": a.model,
        "dataset": a.dataset,
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds),
        "lora_rank": a.rank,
        "train_runtime_s": round(train_out.metrics.get("train_runtime", 0), 1),
        "loss_before": round(before, 4),
        "perplexity_before": round(math.exp(before), 2),
        "loss_after": round(after, 4),
        "perplexity_after": round(math.exp(after), 2),
    }
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Adapter saved: {adapter_dir}")


if __name__ == "__main__":
    main()
