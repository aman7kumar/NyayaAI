"""
Post-processing utilities to improve noisy OCR text readability.
"""

from __future__ import annotations

import os
import re


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _restore_names_from_raw(raw_text: str, processed_text: str) -> str:
    """
    Guardrail: do not let cleanup mutate probable names.
    """
    if not raw_text or not processed_text:
        return processed_text

    raw_names = re.findall(r"\b[A-Z][a-z]{2,}\b", raw_text)
    if not raw_names:
        return processed_text

    unique_raw = list(dict.fromkeys(raw_names))
    token_map: dict[str, str] = {}
    for token in re.findall(r"\b[A-Z][a-z]{2,}\b", processed_text):
        if token in unique_raw:
            continue
        best = None
        best_dist = 99
        for candidate in unique_raw:
            if token[0].lower() != candidate[0].lower():
                continue
            dist = _levenshtein(token.lower(), candidate.lower())
            if dist < best_dist:
                best = candidate
                best_dist = dist
        if best is None:
            continue
        # Tight threshold to avoid false replacements.
        if best_dist <= 2 and (best_dist / max(len(token), len(best))) <= 0.34:
            token_map[token] = best

    fixed = processed_text
    for wrong, correct in token_map.items():
        fixed = re.sub(rf"\b{re.escape(wrong)}\b", correct, fixed)
    return fixed


def _basic_cleanup(text: str) -> str:
    if not text:
        return ""

    cleaned = text
    cleaned = cleaned.replace("\r", "\n")
    cleaned = cleaned.replace("ﬁ", "fi").replace("ﬂ", "fl")
    cleaned = cleaned.replace("’", "'").replace("`", "'")
    cleaned = cleaned.replace("0bservation", "observation")
    cleaned = cleaned.replace("cornplainant", "complainant")
    cleaned = cleaned.replace("comgalnant", "complainant")
    cleaned = cleaned.replace("wtness", "witness")
    cleaned = cleaned.replace("poIice", "police")
    cleaned = cleaned.replace("|", "I")

    # Keep legal tokens intact but reduce random symbols/noise.
    cleaned = re.sub(r"[^\w\s.,:;()/#'\"-]", " ", cleaned)
    cleaned = re.sub(r"\b([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\b", r"\1\2\3", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Split into lighter sentence-like chunks for readability.
    cleaned = re.sub(r"(?<=[a-z0-9])\s+(?=[A-Z][a-z])", ". ", cleaned)
    cleaned = re.sub(r"\.\s*\.", ".", cleaned)
    return cleaned.strip()


def _llm_cleanup(text: str) -> str:
    """
    Optional correction pass using Mistral when key is available.
    Keeps legal facts and section numbers unchanged as much as possible.
    """
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    use_llm = os.getenv("OCR_USE_LLM_CLEANUP", "false").lower() in ("1", "true", "yes")
    if not api_key or not text or not use_llm:
        return text

    try:
        from mistralai import Mistral

        client = Mistral(api_key=api_key)
        model = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")

        prompt = (
            "You are cleaning OCR-extracted FIR/legal text.\n"
            "Rules:\n"
            "- Fix obvious OCR spelling/spacing errors.\n"
            "- Preserve names, numbers, dates, legal section identifiers.\n"
            "- Do not add new facts.\n"
            "- Return plain cleaned text only.\n\n"
            f"Input OCR text:\n{text[:6000]}"
        )

        res = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        out = (res.choices[0].message.content or "").strip()
        return out or text
    except Exception:
        return text


def improve_ocr_text(text: str) -> str:
    cleaned = _basic_cleanup(text)
    llm_cleaned = _llm_cleanup(cleaned)
    return _restore_names_from_raw(text, llm_cleaned)
