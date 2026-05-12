"""
Heuristic FIR analyzer used to enrich chat context.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


FIR_HINTS = ("fir", "first information report", "complaint", "police report")


def _extract_field(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def analyze_fir_text(user_text: str) -> Dict[str, Any]:
    text = (user_text or "").strip()
    normalized = text.lower()
    mentions_fir = any(token in normalized for token in FIR_HINTS)

    if not text:
        return {
            "mentions_fir": False,
            "extracted": {},
            "missing_details": [],
            "one_sided_narration": False,
            "bias_indicators": [],
        }

    extracted = {
        "fir_number": _extract_field(r"fir\s*(?:no|number)?\s*[:#-]?\s*([a-z0-9/-]+)", text),
        "police_station": _extract_field(r"police station\s*[:#-]?\s*([a-zA-Z0-9 ,.-]+)", text),
        "incident_date": _extract_field(r"(?:incident|date)\s*[:#-]?\s*([a-zA-Z0-9, /.-]+)", text),
        "location": _extract_field(r"(?:location|place)\s*[:#-]?\s*([a-zA-Z0-9, /.-]+)", text),
    }

    missing_details: List[str] = []
    for field, value in extracted.items():
        if not value:
            missing_details.append(field)

    # Basic one-sided narration heuristics
    accuser_words = len(re.findall(r"\b(he|she|they|accused|police)\b", normalized))
    self_words = len(re.findall(r"\b(i|me|my|we|our)\b", normalized))
    has_counter_view = bool(re.search(r"\bhowever|but|on the other hand|according to\b", normalized))
    one_sided_narration = (accuser_words > 1 and self_words > 1 and not has_counter_view)

    bias_signals = [
        "police threatened",
        "police refused",
        "forced statement",
        "political pressure",
        "influence used",
        "biased investigation",
    ]
    bias_indicators = [signal for signal in bias_signals if signal in normalized]

    if mentions_fir and "incident_date" in missing_details:
        missing_details.append("timeline_of_events")

    # Keep unique order
    missing_details = list(dict.fromkeys(missing_details))

    return {
        "mentions_fir": mentions_fir,
        "extracted": extracted,
        "missing_details": missing_details,
        "one_sided_narration": one_sided_narration,
        "bias_indicators": bias_indicators,
    }
