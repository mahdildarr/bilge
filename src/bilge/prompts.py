"""Chat formatting shared by training, evaluation and the demo (no model dependencies)."""
from __future__ import annotations

# Bilge is a Turkish assistant, so its system prompt is in Turkish:
# "You are Bilge, a helpful, accurate and clear-spoken Turkish AI assistant."
SYSTEM_PROMPT = "Sen Bilge adında, yardımsever, doğru ve açık konuşan bir Türkçe yapay zekâ asistanısın."


def build_messages(instruction: str, inp: str = "", system: str = SYSTEM_PROMPT) -> list[dict]:
    user = instruction.strip()
    if inp and inp.strip():
        user += "\n\n" + inp.strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def to_messages(message: str, history, system: str = SYSTEM_PROMPT) -> list[dict]:
    """Gradio chat history (messages or legacy pair format) + new message -> chat messages."""
    msgs = [{"role": "system", "content": system}]
    for turn in history:
        if isinstance(turn, dict):
            msgs.append({"role": turn["role"], "content": turn["content"]})
        else:
            user, bot = turn
            msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": bot}]
    msgs.append({"role": "user", "content": message})
    return msgs
