"""
backend/modules/explainability.py
===================================
Explainable AI (XAI) module.
Generates human-readable reasoning for why each IPC/CrPC section was predicted.

Approach:
  1. Keyword overlap analysis — which query words match the section's description
  2. RAG chunk attribution — which retrieved statute chunk supports the prediction
  3. Judicial syllogism format — Major Premise (law) + Minor Premise (facts) → Conclusion
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)


class ExplainabilityModule:
    """Generates plain-language explanations for IPC/CrPC predictions."""

    def generate_explanation(
        self,
        query: str,
        predicted_sections: list[dict],
        rag_chunks: list[dict],
    ) -> str:
        """
        Returns a structured explanation in judicial syllogism format.

        Example output:
        ─────────────────────────────────────────────
        📌 Why IPC 323 was identified:
          • MAJOR PREMISE (The Law): IPC 323 states: "Whoever voluntarily causes hurt..."
          • MINOR PREMISE (Your Facts): Your description mentions "he hit me" and "injury".
          • CONCLUSION: The act of hitting causing injury corresponds to IPC 323.
        ─────────────────────────────────────────────
        """
        if not predicted_sections:
            return "No specific IPC/CrPC sections could be identified with high confidence from your description."

        lines = ["📋 LEGAL ANALYSIS — Why these sections apply to your case:\n"]

        query_words = set(re.findall(r"\b\w{4,}\b", query.lower()))

        for section in predicted_sections[:3]:
            sec_id    = section.get("section", "Unknown")
            sec_title = section.get("title", "")
            sec_desc  = section.get("description", "")
            confidence = section.get("confidence", 0.0)

            # Find matching keywords
            desc_words = set(re.findall(r"\b\w{4,}\b", sec_desc.lower()))
            matching   = query_words & desc_words
            match_str  = ", ".join(f'"{w}"' for w in list(matching)[:5]) if matching else "contextual reasoning"

            # Find supporting RAG chunk
            rag_support = ""
            for chunk in rag_chunks:
                chunk_text = chunk.get("text", "").lower()
                if any(w in chunk_text for w in [sec_id.lower(), sec_title.lower().split()[0]]):
                    rag_support = chunk.get("text", "")[:200] + "..."
                    break

            lines.append(f"─" * 50)
            lines.append(f"⚖️  {sec_id} — {sec_title} (Confidence: {confidence:.0%})")
            lines.append("")
            lines.append(f"  📌 MAJOR PREMISE (The Law):")
            lines.append(f"     {sec_desc}")
            lines.append("")
            lines.append(f"  📌 MINOR PREMISE (Your Facts):")
            lines.append(f"     Your description contains keywords matching this section: {match_str}.")
            if rag_support:
                lines.append(f"     Supporting statute reference found in legal corpus.")
            lines.append("")
            lines.append(f"  📌 CONCLUSION:")
            lines.append(f"     Based on the facts described, {sec_id} appears applicable.")
            lines.append("")

        lines.append("─" * 50)
        lines.append(
            "⚠️  DISCLAIMER: This analysis is AI-generated for informational purposes only. "
            "Please consult a qualified lawyer for legal advice."
        )

        return "\n".join(lines)
