"""
Case-chat orchestration: context builder, memory trim, safety filter.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from backend.modules.chat_prompt_config import SYSTEM_INSTRUCTIONS
from backend.modules.fir_analysis import analyze_fir_text

DISCLAIMER = "This is AI guidance, consult a lawyer for final action."


def _compact_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for section in sections or []:
        compact.append(
            {
                "section": section.get("section"),
                "title": section.get("title"),
                "confidence": section.get("confidence"),
            }
        )
    return compact


def build_case_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query_type": payload.get("query_type"),
        "predicted_crime_type": payload.get("query_type"),
        "ipc_sections": _compact_sections(payload.get("ipc_sections", [])),
        "crpc_sections": _compact_sections(payload.get("crpc_sections", [])),
        "fir_data": payload.get("fir_data") or payload.get("entities", {}),
        "previous_guidance": {
            "summary": payload.get("summary"),
            "explanation": payload.get("explanation"),
            "roadmap": payload.get("roadmap", []),
        },
    }


def trim_history(messages: List[Dict[str, str]], max_chars: int = 6000) -> List[Dict[str, str]]:
    total = 0
    kept: List[Dict[str, str]] = []
    for message in reversed(messages):
        content = message.get("content", "")
        total += len(content)
        if total > max_chars:
            break
        kept.append({"role": message.get("role", "user"), "content": content})
    return list(reversed(kept))


def build_prompt(case_context: Dict[str, Any], messages: List[Dict[str, str]], fir_analysis: Dict[str, Any]) -> Dict[str, Any]:
    context_block = (
        "Case Context:\n"
        f"{case_context}\n\n"
        "FIR Analysis:\n"
        f"{fir_analysis}\n"
    )

    system_prompt = f"{SYSTEM_INSTRUCTIONS}\n\n{context_block}"
    return {
        "system_prompt": system_prompt,
        "messages": trim_history(messages),
    }


def apply_response_safety(raw_response: str, case_context: Dict[str, Any]) -> str:
    text = (raw_response or "").strip()
    lowered = text.lower()

    blocked_patterns = (
        "you should destroy evidence",
        "destroy evidence to",
        "you should bribe",
        "pay a bribe",
        "threaten them",
        "take revenge",
        "fake witness",
        "fabricate evidence",
    )
    if any(pattern in lowered for pattern in blocked_patterns):
        text = (
            "I cannot help with illegal or harmful actions.\n"
            "I can help with lawful legal remedies, documentation, and complaint process."
        )

    # Heuristic guard: flag unsupported section references
    valid_sections = {
        str(sec.get("section", "")).upper()
        for sec in (case_context.get("ipc_sections", []) + case_context.get("crpc_sections", []))
        if sec.get("section")
    }
    referenced = set(re.findall(r"\b(?:IPC|CRPC)\s*\d+[A-Z]?\b", text.upper()))
    unknown = [sec for sec in referenced if sec not in valid_sections]
    if unknown:
        text += "\n\nNote: Some legal section references should be verified with official bare acts and a lawyer."

    if DISCLAIMER not in text:
        text = f"{text}\n\n{DISCLAIMER}"

    return text


def compute_fir_analysis(message: str, case_context: Dict[str, Any]) -> Dict[str, Any]:
    fir_seed = case_context.get("fir_data") or {}
    merged_text = f"{message}\n\n{fir_seed}"
    return analyze_fir_text(merged_text)
