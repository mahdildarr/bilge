"""QLoRA fine-tuning of an open-source language model on Turkish instruction data.

- The data is cleaned and decontaminated against the evaluation prompts first (``bilge.data``); the
  number of examples removed by each filter is saved with the run.
- The base model is quantised to 4-bit NF4; only the LoRA adapter weights are trained.
- The loss is computed on the assistant answer only; the prompt is masked.
- Validation loss is tracked during training; the best checkpoint is kept (early stopping).
- Perplexity is measured on the same held-out validation set before and after training. Because
  LoRA's B matrix is initialised to zero, the pre-training measurement equals the base model.

Example (Colab T4)::

    python -m bilge.train --config configs/qwen2.5-1.5b.yaml
    python -m bilge.train --config configs/qwen2.5-1.5b.yaml --set train_samples=1000 rank=8 out=runs/quick
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import yaml

DEFAULTS = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "dataset": "TFLai/Turkish-Alpaca",
    "eval_prompts": "prompts/evaluation_prompts.json",
    "train_samples": 5000,
    "eval_samples": 500,
    "max_len": 512,
    "epochs": 1.0,
    "lr": 2e-4,
    "batch": 4,
    "grad_accum": 4,
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "eval_steps": 50,
    "patience": 3,
    "out": "runs/bilge-1.5b-lora",
    "seed": 42,
}


def parse_value(v: str):
    """YAML scalar, with "1e-4"-style floats (strings under YAML 1.1) converted to float."""
    x = yaml.safe_load(v)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            pass
    return x


def load_config(path=None, overrides=()) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text()) or {})
    for kv in overrides:
        k, v = kv.split("=", 1)
        if k not in DEFAULTS:
            raise SystemExit(f"Unknown setting: {k}")
        cfg[k] = parse_value(v)
    return cfg


def prepare_data(cfg, tok):
    from datasets import Dataset, load_dataset

    from .data import clean, has_answer, make_tokenize

    raw = load_dataset(cfg["dataset"], split="train").shuffle(seed=cfg["seed"])
    eval_prompts = json.loads(Path(cfg["eval_prompts"]).read_text(encoding="utf-8")) if cfg["eval_prompts"] else []
    examples, counts = clean(list(raw), eval_prompts)
    n_eval = min(cfg["eval_samples"], len(examples) // 10)
    eval_ds = Dataset.from_list(examples[:n_eval])
    train_ds = Dataset.from_list(examples[n_eval:n_eval + cfg["train_samples"]])
    tokenize = make_tokenize(tok, cfg["max_len"])
    cols = train_ds.column_names
    train_ds = train_ds.map(tokenize, remove_columns=cols).filter(has_answer)
    eval_ds = eval_ds.map(tokenize, remove_columns=cols).filter(has_answer)
    return train_ds, eval_ds, counts


def main(argv=None):
    p = argparse.ArgumentParser(description="QLoRA fine-tuning on Turkish instruction data")
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.set)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer, TrainingArguments

    from .modeling import compute_dtype, load_base_model, load_tokenizer

    tok = load_tokenizer(cfg["model"])
    train_ds, eval_ds, counts = prepare_data(cfg, tok)
    print(f"Data filters: {counts}")
    print(f"Training examples: {len(train_ds)} | Validation examples: {len(eval_ds)}")

    model = load_base_model(cfg["model"], four_bit=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                                             target_modules=cfg["targets"], task_type="CAUSAL_LM"))
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    import torch

    bf16 = compute_dtype() == torch.bfloat16
    args = TrainingArguments(
        output_dir=cfg["out"],
        per_device_train_batch_size=cfg["batch"],
        per_device_eval_batch_size=cfg["batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["eval_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not bf16,
        bf16=bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["patience"])])

    before = trainer.evaluate()["eval_loss"]
    print(f"Validation loss before training: {before:.4f} (perplexity {math.exp(before):.2f})")
    train_out = trainer.train()
    after = trainer.evaluate()["eval_loss"]
    print(f"Validation loss after training: {after:.4f} (perplexity {math.exp(after):.2f})")

    adapter_dir = os.path.join(cfg["out"], "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    metrics = {
        "config": cfg,
        "data_filters": counts,
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds),
        "trainable_parameters": trainable,
        "trainable_share": trainable / total,
        "train_runtime_s": round(train_out.metrics.get("train_runtime", 0), 1),
        "best_step": trainer.state.best_global_step if hasattr(trainer.state, "best_global_step") else None,
        "loss_before": round(before, 4),
        "perplexity_before": round(math.exp(before), 2),
        "loss_after": round(after, 4),
        "perplexity_after": round(math.exp(after), 2),
        "log_history": trainer.state.log_history,
    }
    with open(os.path.join(cfg["out"], "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Perplexity {metrics['perplexity_before']} -> {metrics['perplexity_after']}. Adapter saved: {adapter_dir}")


if __name__ == "__main__":
    main()
