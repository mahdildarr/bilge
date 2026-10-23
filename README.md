# Bilge · Turkish AI Assistant

[![CI](https://github.com/mahdildarr/bilge/actions/workflows/ci.yml/badge.svg)](https://github.com/mahdildarr/bilge/actions/workflows/ci.yml)

**Bilge** is my own Turkish-speaking AI assistant, which I am building by fine-tuning an open-source large language model on Turkish instruction data with **QLoRA**. The project covers training, a **measurable comparison** against the base model and a browser-based **chat interface**.

```bash
# Chat with Bilge after training:
python app.py --adapter runs/bilge-1.5b-lora/adapter --share
```

<p align="center"><img src="docs/assets/chat_interface.png" width="100%" alt="Example"></p>
<p align="center"><sub>Bilge's chat interface (Gradio).</sub></p>

## Problem

Multilingual open models often answer Turkish instructions with short, incomplete or grammatically weak responses, and small ones drift into English or Chinese mid-answer, while full fine-tuning requires a lot of GPU memory. This project aims to improve Turkish instruction-following on a single free Colab GPU (T4, 16 GB) by training only a small fraction of the model's parameters, and to measure the improvement rather than eyeball it.

## Approach

<p align="center"><img src="docs/assets/pipeline.svg" width="100%" alt="Pipeline"></p>

| Step | Method | Module |
|---|---|---|
| 1. Data cleaning | Turkish-Alpaca is machine-translated Alpaca. Counted filters remove empty and overlong answers, untranslated (non-Turkish) answers, "as an AI language model" boilerplate and duplicates | `bilge.data` |
| 2. Decontamination | Training examples sharing an 8-word n-gram with an evaluation prompt are removed, so the evaluation cannot reward memorisation | `bilge.data` |
| 3. Prompt masking | Examples are rendered with the model's chat template; the loss is computed on the answer only | `bilge.data` |
| 4. QLoRA training | 4-bit NF4 base model; LoRA adapters on the attention and MLP projections; validation loss tracked during training with early stopping (bf16 where supported, fp16 on a T4) | `bilge.train` |
| 5. Perplexity | Before and after training on the same held-out validation set (answer tokens only) | `bilge.train` |
| 6. Answer measurements | Base vs fine-tuned answers on fixed prompts in 8 categories, greedy decoding; share of answers in Turkish, share with CJK characters, repeated 3-grams, length | `bilge.compare` |
| 7. Knowledge retention | Multiple-choice accuracy by option log-likelihood, base vs fine-tuned, on any Turkish MC benchmark: does instruction tuning cost knowledge? | `bilge.mc_eval` |
| 8. Scaling | Training-set size × LoRA rank grid | `bilge.sweep` |
| 9. Release | Model card with the measured numbers, optional merged 16-bit weights, upload to the Hugging Face Hub | `bilge.export` |
| 10. Demo | Bilge's Gradio chat interface | `app.py` |

## Model and data

- **Base model:** [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache-2.0)
- **Data:** [TFLai/Turkish-Alpaca](https://huggingface.co/datasets/TFLai/Turkish-Alpaca) (~52k Turkish instruction examples, Apache-2.0)
- **Default settings:** 5,000 training examples, 500 validation examples, 1 epoch, LoRA rank 16, up to 512 tokens

All settings live in `configs/qwen2.5-1.5b.yaml` and can be overridden with `--set key=value`. Any dataset with `instruction`, `input` and `output` columns works.

## Installation

```bash
git clone https://github.com/mahdildarr/bilge.git
cd bilge
pip install -e ".[gpu,demo,hub,dev]"
pytest -q          # data cleaning, masking, metrics (no GPU or model download needed)
```

The quickest way to run everything is the `notebooks/colab_qlora_training.ipynb` notebook on Google Colab (T4 GPU).

## Usage

```bash
# 1) Clean data + QLoRA training (perplexity before / after is measured automatically)
python -m bilge.train --config configs/qwen2.5-1.5b.yaml

# 2) Base model vs Bilge: answers side by side, with language / repetition measurements
python -m bilge.compare --adapter runs/bilge-1.5b-lora/adapter

# 3) Knowledge retention on a Turkish multiple-choice benchmark (column names are configurable)
python -m bilge.mc_eval --adapter runs/bilge-1.5b-lora/adapter --dataset <hf dataset> --split test

# 4) Training-set size x LoRA rank grid
python -m bilge.sweep --config configs/sweep.yaml

# 5) Model card (+ merged weights) and upload
python -m bilge.export --run runs/bilge-1.5b-lora --repo <user>/bilge-qwen2.5-1.5b-lora --push

# 6) Chat interface
python app.py --adapter runs/bilge-1.5b-lora/adapter --share
```

## Evaluation metrics

- **Perplexity:** on a held-out validation set, over answer tokens only; lower is better
- **Answer language:** share of answers detected as Turkish, and share containing Chinese / Japanese / Korean characters
- **Degeneration:** share of repeated word 3-grams; mean answer length
- **Knowledge retention:** multiple-choice accuracy (option-letter log-likelihood), base vs fine-tuned
- **Qualitative comparison:** answers in explanation, technical knowledge, summarisation, writing, reasoning, classification, code and grammar categories (`prompts/evaluation_prompts.json`)
- **Efficiency:** share of trainable parameters and training time

## Progress

- [x] Turkish instruction data preparation with prompt masking
- [x] QLoRA training on a single T4 GPU
- [x] Perplexity before / after fine-tuning
- [x] Side-by-side comparison with the base model
- [x] Gradio chat interface
- [x] Counted data-cleaning filters and n-gram decontamination against the evaluation prompts
- [x] Early stopping on validation loss
- [x] Answer-language and repetition measurements
- [x] Multiple-choice knowledge-retention evaluation
- [x] Train-size × rank sweep and Hub export with model card
- [x] Unit tests and CI
- [x] Colab notebook

Next:

- [ ] Baseline run with 5,000 examples
- [ ] Effect of training set size and LoRA rank
- [ ] Publish the adapter on the Hugging Face Hub

## Project structure

```
├── app.py                           # Gradio chat interface
├── configs/                         # training config and sweep grid
├── notebooks/colab_qlora_training.ipynb
├── prompts/evaluation_prompts.json  # evaluation prompts (Turkish)
├── src/bilge/
│   ├── data.py                      # cleaning, decontamination, prompt masking
│   ├── textstats.py                 # Turkish detection, repetition
│   ├── prompts.py                   # system prompt and chat formatting
│   ├── modeling.py                  # model loading (4-bit / 16-bit) and generation
│   ├── train.py                     # QLoRA training and perplexity
│   ├── compare.py                   # base ↔ fine-tuned answers and measurements
│   ├── mc_eval.py                   # multiple-choice knowledge retention
│   ├── sweep.py                     # train-size × rank grid
│   └── export.py                    # model card, merge, Hub upload
├── tests/
└── pyproject.toml
```

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023
- [Hugging Face Transformers](https://github.com/huggingface/transformers) · [PEFT](https://github.com/huggingface/peft)

## License

MIT
