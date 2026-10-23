"""Instruction data: cleaning, decontamination and tokenisation with prompt masking.

Turkish-Alpaca is a machine translation of Stanford Alpaca, so it inherits Alpaca's problems and adds
translation artefacts. ``clean`` applies explicit, counted filters:

* empty or very short answers, and answers that are too long for the context window,
* answers that are not Turkish (left untranslated, or code-only answers when ``keep_code`` is off),
* boilerplate refusals ("bir yapay zeka dil modeli olarak ...", "as an AI language model"),
* exact duplicates after normalisation (case, whitespace, punctuation),
* *decontamination*: training examples whose instruction shares a long word n-gram with an
  evaluation prompt, so that evaluation does not reward memorisation.
"""
from __future__ import annotations

import re
from collections import Counter

from .prompts import build_messages
from .textstats import is_turkish, words

REFUSALS = ("yapay zeka dil modeli olarak", "bir yapay zeka modeli olarak", "as an ai language model",
            "bir dil modeli olarak")


def normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def looks_like_code(text: str) -> bool:
    return "```" in text or bool(re.search(r"^\s*(def |class |import |for |if |#include|public )", text, re.M))


def clean(examples: list[dict], eval_prompts: list[dict] = (), min_answer_chars: int = 2,
          max_answer_chars: int = 4000, contamination_n: int = 8, keep_code: bool = True):
    """Filter a list of {"instruction", "input", "output"} dicts. Returns (kept, counts per filter)."""
    eval_grams = set()
    for p in eval_prompts:
        eval_grams |= ngrams(p["instruction"] + " " + p.get("input", ""), contamination_n)

    def reason(ex) -> str | None:
        out = (ex.get("output") or "").strip()
        prompt = (ex.get("instruction") or "") + " " + (ex.get("input") or "")
        if len(out) < min_answer_chars:
            return "empty_or_short_answer"
        if len(out) > max_answer_chars:
            return "too_long"
        if any(r in out.lower() for r in REFUSALS):
            return "boilerplate_refusal"
        if not is_turkish(out) and not (keep_code and looks_like_code(out)) and len(words(out)) >= 5:
            return "not_turkish"
        if eval_grams and ngrams(prompt, contamination_n) & eval_grams:
            return "overlaps_eval_prompt"
        key = normalise(prompt) + "\x00" + normalise(out)
        if key in seen:
            return "duplicate"
        seen.add(key)
        return None

    counts, seen, kept = Counter(), set(), []
    for ex in examples:
        r = reason(ex)
        if r:
            counts[r] += 1
        else:
            kept.append(ex)
    counts["kept"] = len(kept)
    return kept, dict(counts)


def make_tokenize(tok, max_len: int):
    """Tokenise one example with the chat template; the loss is computed on the answer only."""

    def fn(ex):
        prompt = tok.apply_chat_template(build_messages(ex["instruction"], ex.get("input", "")), tokenize=False,
                                         add_generation_prompt=True)
        full = prompt + ex["output"].strip() + tok.eos_token
        p_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
        ids = tok(full, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
        labels = [-100] * min(p_len, len(ids)) + ids[p_len:]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    return fn


def has_answer(ex) -> bool:
    """False when the prompt fills the whole window and no answer token is left to learn from."""
    return any(label != -100 for label in ex["labels"])
