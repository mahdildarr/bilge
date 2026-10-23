"""Package a trained adapter for the Hugging Face Hub: model card, optional merged weights, optional upload.

The adapter alone is small (tens of MB) and loads on top of the base model. ``--merge`` additionally
writes a standalone 16-bit model with the LoRA weights folded in (loaded without quantisation, since
merging into 4-bit weights loses precision).

    python -m bilge.export --run runs/bilge-1.5b-lora --repo <user>/bilge-qwen2.5-1.5b-lora [--merge] [--push]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def model_card(metrics: dict, repo: str) -> str:
    cfg = metrics.get("config", {})
    filters = metrics.get("data_filters", {})
    return f"""---
language: [tr]
license: apache-2.0
base_model: {cfg.get("model", "")}
datasets: [{cfg.get("dataset", "")}]
library_name: peft
tags: [qlora, lora, turkish, instruction-tuning]
---

# Bilge · Turkish AI Assistant (LoRA adapter)

QLoRA adapter for `{cfg.get("model", "")}`, trained on {metrics.get("train_examples", "?"):,} cleaned Turkish
instruction examples from `{cfg.get("dataset", "")}`.

| | Value |
|---|---|
| LoRA rank / alpha | {cfg.get("rank")} / {cfg.get("alpha")} |
| Trainable parameters | {metrics.get("trainable_parameters", 0):,} ({100 * metrics.get("trainable_share", 0):.2f}%) |
| Validation perplexity, base model | {metrics.get("perplexity_before")} |
| Validation perplexity, with adapter | {metrics.get("perplexity_after")} |

Data filters applied before training: {", ".join(f"{k}: {v}" for k, v in filters.items())}.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{cfg.get("model", "")}", device_map="auto")
model = PeftModel.from_pretrained(base, "{repo}")
tok = AutoTokenizer.from_pretrained("{repo}")
```

## Limitations

The training data is machine-translated; answers can contain translation artefacts and factual
errors. Not evaluated for safety-critical use.
"""


def main(argv=None):
    p = argparse.ArgumentParser(description="Export a Bilge adapter")
    p.add_argument("--run", required=True, help="Training output folder (contains adapter/ and metrics.json)")
    p.add_argument("--repo", required=True, help="Hub repository id, e.g. user/bilge-qwen2.5-1.5b-lora")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--out", default=None, help="Default: <run>/export")
    a = p.parse_args(argv)

    run = Path(a.run)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    out = Path(a.out or run / "export")
    shutil.copytree(run / "adapter", out / "adapter", dirs_exist_ok=True)
    (out / "adapter" / "README.md").write_text(model_card(metrics, a.repo), encoding="utf-8")
    if a.merge:
        from peft import PeftModel

        from .modeling import load_base_model, load_tokenizer

        base_name = metrics["config"]["model"]
        base = load_base_model(base_name, four_bit=False)
        merged = PeftModel.from_pretrained(base, run / "adapter").merge_and_unload()
        merged.save_pretrained(out / "merged", safe_serialization=True)
        load_tokenizer(base_name).save_pretrained(out / "merged")
    if a.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(a.repo, exist_ok=True)
        api.upload_folder(folder_path=str(out / "adapter"), repo_id=a.repo)
        print(f"Pushed: https://huggingface.co/{a.repo}")
    print(f"Export: {out}")


if __name__ == "__main__":
    main()
