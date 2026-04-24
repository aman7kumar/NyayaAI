"""
backend/modules/entity_extractor.py
=====================================
Extracts legal entities from text using spaCy + custom rules.
Entities: PERSON, ORG, LOCATION, DATE, TIME, IPC_SECTION, MONEY, WEAPON, OFFENCE
"""

from __future__ import annotations
import re
import logging

logger = logging.getLogger(__name__)

# Regex patterns for Indian legal entities
PATTERNS = {
    "ipc_section":   re.compile(r"\b(?:ipc|section|sec\.?)\s*(\d+[A-Za-z]?)\b", re.IGNORECASE),
    "crpc_section":  re.compile(r"\b(?:crpc|cr\.?p\.?c\.?)\s*(?:sec(?:tion)?\.?)?\s*(\d+[A-Za-z]?)\b", re.IGNORECASE),
    "date":          re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4})\b", re.IGNORECASE),
    "time":          re.compile(r"\b(\d{1,2}:\d{2}(?:\s?[ap]m)?)\b", re.IGNORECASE),
    "money":         re.compile(r"\b(?:rs\.?|inr|₹)\s*(\d[\d,]*(?:\.\d+)?(?:\s*(?:lakh|crore|thousand))?)\b", re.IGNORECASE),
    "phone":         re.compile(r"\b([6-9]\d{9})\b"),
    "aadhar":        re.compile(r"\b(\d{4}\s\d{4}\s\d{4})\b"),
}

WEAPON_WORDS = {"knife", "gun", "pistol", "rod", "bat", "stick", "blade", "acid", "weapon", "revolver", "firearm"}
OFFENCE_WORDS = {"murder", "assault", "rape", "robbery", "theft", "kidnap", "fraud", "cheating", "extortion", "dacoity", "arson"}


class EntityExtractor:
    """Extracts structured legal entities from query text."""

    def extract(self, text: str) -> dict:
        entities = {
            "ipc_sections_mentioned": [],
            "crpc_sections_mentioned": [],
            "dates": [],
            "times": [],
            "amounts": [],
            "weapons": [],
            "offences": [],
            "locations": [],
            "persons": [],
        }

        for match in PATTERNS["ipc_section"].finditer(text):
            entities["ipc_sections_mentioned"].append(f"IPC {match.group(1)}")

        for match in PATTERNS["crpc_section"].finditer(text):
            entities["crpc_sections_mentioned"].append(f"CrPC {match.group(1)}")

        for match in PATTERNS["date"].finditer(text):
            entities["dates"].append(match.group(1))

        for match in PATTERNS["time"].finditer(text):
            entities["times"].append(match.group(1))

        for match in PATTERNS["money"].finditer(text):
            entities["amounts"].append(f"₹{match.group(1)}")

        text_lower = text.lower()
        entities["weapons"] = [w for w in WEAPON_WORDS if w in text_lower]
        entities["offences"] = [o for o in OFFENCE_WORDS if o in text_lower]

        # Simple NER via spaCy (if available)
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities["persons"].append(ent.text)
                elif ent.label_ in ("GPE", "LOC", "FAC"):
                    entities["locations"].append(ent.text)
        except Exception:
            pass  # spaCy optional

        # Deduplicate
        for k in entities:
            if isinstance(entities[k], list):
                entities[k] = list(dict.fromkeys(entities[k]))

        return entities
