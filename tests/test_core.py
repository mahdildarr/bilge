import json

import pytest

from bilge.data import clean, has_answer, make_tokenize
from bilge.export import model_card
from bilge.mc_eval import answer_index, format_question
from bilge.prompts import SYSTEM_PROMPT, build_messages, to_messages
from bilge.sweep import grid_points, run_name
from bilge.textstats import has_cjk, is_turkish, repetition, turkish_score
from bilge.train import load_config

TR = "Yapay zekâ, bilgisayarların insan gibi düşünmesini ve öğrenmesini sağlayan bir bilim dalıdır."
EN = "Artificial intelligence is the field of computer science that makes machines think and learn."
ZH = "人工智能是计算机科学的一个分支。"


def test_language_and_repetition_stats():
    assert is_turkish(TR) and not is_turkish(EN) and not is_turkish(ZH)
    assert has_cjk(TR + ZH) and not has_cjk(TR)
    assert turkish_score(TR) > turkish_score(TR + " " + ZH * 3)
    assert repetition("bir iki üç bir iki üç bir iki üç") > 0.5
    assert repetition(TR) == 0


def test_messages():
    m = build_messages("Soru", "Girdi")
    assert m[0] == {"role": "system", "content": SYSTEM_PROMPT} and m[1]["content"] == "Soru\n\nGirdi"
    pairs = to_messages("yeni", [("a", "b")])
    msgs = to_messages("yeni", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    assert pairs == msgs and msgs[-1] == {"role": "user", "content": "yeni"}


def test_cleaning_filters_and_decontamination():
    eval_prompts = [{"instruction": "Yapay sinir ağlarında aşırı öğrenme nedir önlemek için üç yöntem öner"}]
    data = [
        {"instruction": "Bir şey anlat", "input": "", "output": TR},
        {"instruction": "Bir şey anlat", "input": "", "output": TR},  # duplicate
        {"instruction": "Explain", "input": "", "output": EN},  # untranslated
        {"instruction": "Kod yaz", "input": "", "output": "```python\nprint('merhaba dünya')\n```"},  # code is kept
        {"instruction": "Soru", "input": "", "output": ""},
        {"instruction": "Soru", "input": "", "output": "Bir yapay zeka dil modeli olarak buna cevap veremem."},
        {"instruction": "Yapay sinir ağlarında aşırı öğrenme nedir önlemek için üç yöntem öner lütfen", "input": "",
         "output": TR + " Ek."},
    ]
    kept, counts = clean(data, eval_prompts)
    assert counts == {"duplicate": 1, "not_turkish": 1, "empty_or_short_answer": 1, "boilerplate_refusal": 1,
                      "overlaps_eval_prompt": 1, "kept": 2}
    assert [k["instruction"] for k in kept] == ["Bir şey anlat", "Kod yaz"]


class CharTokenizer:
    """Character-level stand-in for a Hugging Face tokenizer (enough for the masking logic)."""

    eos_token = "|"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(c) for c in text]
        return {"input_ids": ids[:max_length] if truncation else ids}


def test_prompt_is_masked_and_answer_is_learned():
    tok = CharTokenizer()
    ex = make_tokenize(tok, max_len=10_000)({"instruction": "Soru", "input": "", "output": "Cevap"})
    learned = [chr(i) for i, lab in zip(ex["input_ids"], ex["labels"]) if lab != -100]
    assert "".join(learned) == "Cevap|"
    truncated = make_tokenize(tok, max_len=20)({"instruction": "Soru", "input": "", "output": "Cevap"})
    assert not has_answer(truncated)  # the prompt fills the window: the example must be dropped


def test_multiple_choice_helpers():
    q = format_question("Başkent?", ["İzmir", "Ankara"])
    assert "A) İzmir" in q and "B) Ankara" in q
    assert answer_index("B", 4) == 1 and answer_index(2, 4) == 2 and answer_index("3", 4) == 3


def test_config_overrides(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("rank: 8\n")
    cfg = load_config(cfg_file, ["train_samples=100", "lr=1e-4"])
    assert cfg["rank"] == 8 and cfg["train_samples"] == 100 and cfg["lr"] == 1e-4
    with pytest.raises(SystemExit):
        load_config(None, ["unknown=1"])


def test_sweep_grid():
    pts = grid_points({"train_samples": [1, 2], "rank": [8, 16]})
    assert len(pts) == 4 and run_name(pts[0]) == "train_samples1_rank8"


def test_model_card_mentions_results():
    metrics = {"config": {"model": "Qwen/Qwen2.5-1.5B-Instruct", "dataset": "TFLai/Turkish-Alpaca", "rank": 16,
                          "alpha": 32}, "train_examples": 5000, "trainable_parameters": 1000, "trainable_share": 0.01,
               "perplexity_before": 5.1, "perplexity_after": 3.2, "data_filters": {"duplicate": 3}}
    card = model_card(metrics, "user/bilge")
    assert "5.1" in card and "3.2" in card and "duplicate: 3" in card and "language: [tr]" in card
    json.dumps(metrics)
