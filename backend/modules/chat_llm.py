"""
LLM abstraction layer for case-chat (Claude / Mistral swappable).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List


class BaseChatLLM:
    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class MistralChatLLM(BaseChatLLM):
    def __init__(self) -> None:
        from mistralai import Mistral

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY not set")
        self.client = Mistral(api_key=api_key)
        self.model = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")

    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        payload = [{"role": "system", "content": system_prompt}] + messages
        response = self.client.chat.complete(model=self.model, messages=payload)
        return response.choices[0].message.content or ""


class ClaudeChatLLM(BaseChatLLM):
    def __init__(self) -> None:
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=api_key)
        self.model = os.getenv("CLAUDE_CHAT_MODEL", "claude-3-5-sonnet-latest")

    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=900,
            system=system_prompt,
            messages=messages,
        )
        chunks = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        return "\n".join(chunks).strip()


class FallbackGuidanceLLM(BaseChatLLM):
    """
    Deterministic fallback so chat remains functional without API keys.
    """

    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        latest = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return (
            "I can help you structure your case details and next legal steps.\n"
            f"You asked: {latest[:300]}\n\n"
            "- Please share incident date, location, and witness details.\n"
            "- Keep copies of FIR, medical records, and communication proof.\n"
            "- If police are not registering FIR, document refusal and approach senior officer or magistrate.\n\n"
            "This is AI guidance, consult a lawyer for final action."
        )


def build_llm_client() -> BaseChatLLM:
    provider = (os.getenv("CHAT_LLM_PROVIDER", "mistral") or "mistral").lower()
    try:
        if provider == "claude":
            return ClaudeChatLLM()
        if provider == "mistral":
            return MistralChatLLM()
    except Exception:
        return FallbackGuidanceLLM()
    return FallbackGuidanceLLM()
