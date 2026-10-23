"""Multiple-choice accuracy by log-likelihood, base vs fine-tuned: does instruction tuning cost knowledge?

Fine-tuning on translated instructions can improve fluency while eroding what the model knew
(catastrophic forgetting). Each question is shown with its options, and the option letter with the
highest log-probability is the prediction; no free-form generation or answer parsing is involved,
so the score is deterministic and comparable across models.

Works with any Hugging Face dataset that has a question, a list of choices and an answer (index or
letter); column names are configurable. Example for a Turkish MMLU-style benchmark::

    python -m bilge.mc_eval --adapter runs/bilge-1.5b-lora/adapter --dataset <hf dataset> --split test \
        --question-col question --choices-col choices --answer-col answer --limit 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LETTERS = "ABCDEFGH"


def format_question(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
    # "Answer with the letter of the correct option only."
    return f"{question.strip()}\n\n{opts}\n\nSadece doğru seçeneğin harfiyle cevap ver."


def answer_index(answer, n_choices: int) -> int:
    if isinstance(answer, str):
        a = answer.strip().upper()
        if a and a[0] in LETTERS[:n_choices]:
            return LETTERS.index(a[0])
        return int(a)
    return int(answer)


def letter_logprobs(model, tok, messages, n_choices: int) -> list[float]:
    """Log-probability of each option letter as the first answer token."""
    import torch

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1].float()
    logp = torch.log_softmax(logits, -1)
    ids = [tok(LETTERS[i], add_special_tokens=False)["input_ids"][0] for i in range(n_choices)]
    return [float(logp[i]) for i in ids]


def evaluate(model, tok, rows, build_messages) -> dict:
    correct, by_subject = 0, {}
    for r in rows:
        msgs = build_messages(format_question(r["question"], r["choices"]))
        lp = letter_logprobs(model, tok, msgs, len(r["choices"]))
        ok = int(max(range(len(lp)), key=lp.__getitem__) == r["answer"])
        correct += ok
        s = by_subject.setdefault(r.get("subject", "all"), [0, 0])
        s[0] += ok
        s[1] += 1
    return {"accuracy": correct / max(len(rows), 1), "n": len(rows),
            "per_subject": {k: v[0] / v[1] for k, v in sorted(by_subject.items())}}


def main(argv=None):
    p = argparse.ArgumentParser(description="Multiple-choice evaluation, base vs adapter")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config-name", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--question-col", default="question")
    p.add_argument("--choices-col", default="choices")
    p.add_argument("--answer-col", default="answer")
    p.add_argument("--subject-col", default="subject")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/mc_eval.json")
    a = p.parse_args(argv)

    from datasets import load_dataset
    from peft import PeftModel

    from .modeling import load_base_model, load_tokenizer
    from .prompts import build_messages

    ds = load_dataset(a.dataset, a.config_name, split=a.split).shuffle(seed=a.seed)
    ds = ds.select(range(min(a.limit, len(ds))))
    rows = [{"question": ex[a.question_col], "choices": list(ex[a.choices_col]),
             "answer": answer_index(ex[a.answer_col], len(ex[a.choices_col])),
             "subject": ex.get(a.subject_col, "all")} for ex in ds]
    tok = load_tokenizer(a.model)
    model = PeftModel.from_pretrained(load_base_model(a.model, four_bit=True), a.adapter).eval()
    with model.disable_adapter():
        base = evaluate(model, tok, rows, build_messages)
    tuned = evaluate(model, tok, rows, build_messages)
    res = {"dataset": a.dataset, "split": a.split, "n": len(rows), "base": base, "tuned": tuned}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Accuracy: base {base['accuracy']:.3f} | Bilge {tuned['accuracy']:.3f} (n={len(rows)})")


if __name__ == "__main__":
    main()
