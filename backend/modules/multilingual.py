"""
backend/modules/multilingual.py
=================================
Multilingual support for Hindi ↔ English translation.
Uses Helsinki-NLP/opus-mt models (local, no API needed).

Models:
  Helsinki-NLP/opus-mt-hi-en  (Hindi → English)
  Helsinki-NLP/opus-mt-en-hi  (English → Hindi)

These are ~300MB models downloaded once via HuggingFace Hub.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


class MultilingualModule:
    """Handles language detection and Hindi↔English translation."""

    _hi_to_en = None
    _en_to_hi = None

    # ── Language Detection ─────────────────────────────────────────────────

    def detect_language(self, text: str) -> str:
        """
        Detects language using script-based heuristic + langdetect fallback.
        Returns ISO 639-1 code: 'en', 'hi', 'mr', etc.
        """
        # Fast Devanagari script check
        devanagari = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
        if devanagari / max(len(text), 1) > 0.2:
            return "hi"

        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return "en"

    # ── Translation ────────────────────────────────────────────────────────

    def translate_to_english(self, text: str, src: str = "hi") -> str:
        """
        Translate text to English using local Helsinki-NLP MarianMT model.
        Downloads model on first use (~300MB).
        """
        if src == "en":
            return text

        try:
            from transformers import MarianMTModel, MarianTokenizer

            model_name = f"Helsinki-NLP/opus-mt-{src}-en"
            logger.info(f"Loading translation model: {model_name}")

            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)

            # Split into chunks (MarianMT has 512 token limit)
            chunks = self._chunk_text(text, max_chars=400)
            translated_chunks = []

            for chunk in chunks:
                inputs = tokenizer([chunk], return_tensors="pt", padding=True, truncation=True)
                translated = model.generate(**inputs)
                result = tokenizer.decode(translated[0], skip_special_tokens=True)
                translated_chunks.append(result)

            return " ".join(translated_chunks)

        except Exception as e:
            logger.warning(f"Translation failed: {e}. Returning original text.")
            return text

    def translate_to_hindi(self, text: str) -> str:
        """Translate English text to Hindi."""
        try:
            from transformers import MarianMTModel, MarianTokenizer
            model_name = "Helsinki-NLP/opus-mt-en-hi"
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)

            inputs = tokenizer([text[:400]], return_tensors="pt", padding=True, truncation=True)
            translated = model.generate(**inputs)
            return tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Hindi translation failed: {e}")
            return text

    # ── Helpers ────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, max_chars: int = 400) -> list[str]:
        """Split text into chunks of max_chars, respecting sentence boundaries."""
        if len(text) <= max_chars:
            return [text]
        parts = text.split(". ")
        chunks, current = [], ""
        for part in parts:
            if len(current) + len(part) < max_chars:
                current += part + ". "
            else:
                if current:
                    chunks.append(current.strip())
                current = part + ". "
        if current:
            chunks.append(current.strip())
        return chunks if chunks else [text[:max_chars]]
