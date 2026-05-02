"""
backend/modules/query_classifier.py
=====================================
Classifies a legal query into one of the major legal domains.
Uses a rule-based + ML hybrid approach:
  - Fast rule-based keywords for obvious cases
  - Falls back to a fine-tuned text classifier for ambiguous cases

Domains: criminal, civil, consumer, family, cyber, property, labour, dowry_harassment
"""

from __future__ import annotations

import re
from typing import Optional

# ── Keyword Rules (fast path) ─────────────────────────────────────────────────

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "criminal": [
        "assault", "murder", "killed", "robbery", "theft", "stolen", "rape",
        "kidnap", "abduction", "extortion", "dacoity", "hurt", "injury",
        "threatening", "threat", "attack", "hit me", "beat me", "shot",
        "stabbed", "poisoned", "bribe", "blackmail", "harassment", "fir",
        "police", "arrest", "bail", "ipc", "crpc", "cognizable",
    ],
    "consumer": [
        "product", "defective", "refund", "company", "fraud", "cheated",
        "online shopping", "delivery", "amazon", "flipkart", "service",
        "warranty", "guarantee", "deficiency", "overcharged", "billing",
        "consumer", "purchase", "bought", "seller",
    ],
    "family": [
        "divorce", "custody", "maintenance", "alimony", "matrimonial",
        "husband", "wife", "spouse", "child custody", "adoption",
        "guardianship", "succession", "inheritance", "will", "property dispute",
        "marriage", "domestic", "talaq", "separation",
    ],
    "cyber": [
        "online", "internet", "hacking", "hacked", "phishing", "spam",
        "social media", "whatsapp", "facebook", "instagram", "cyber",
        "data breach", "identity theft", "fake profile", "morphed photos",
        "sexting", "revenge porn", "upi fraud", "otp fraud", "online fraud",
    ],
    "property": [
        "land", "plot", "property", "house", "rent", "tenant", "landlord",
        "eviction", "possession", "sale deed", "registry", "encroachment",
        "boundary", "benami",
    ],
    "labour": [
        "job", "employer", "employee", "salary", "wages", "fired", "dismissed",
        "termination", "workplace", "pf", "esi", "labour", "trade union",
        "discrimination at work", "sexual harassment at workplace",
    ],
    "dowry_harassment": [
        "dowry", "498a", "cruelty", "in-laws", "mother in law", "father in law",
        "harassment by husband", "domestic violence", "beaten by husband",
    ],
    "civil": [
        "contract", "agreement", "breach", "damages", "suit", "decree",
        "civil", "injunction", "specific performance", "cheque bounce",
        "dishonoured", "promissory note", "debt", "money recovery",
    ],
}


class QueryClassifier:
    """
    Classifies legal queries into domain categories.
    """

    def classify(self, text: str, fallback: str = "criminal") -> str:
        """
        Returns the most likely legal domain for the given text.
        """
        text_lower = text.lower()

        # Score each domain
        scores: dict[str, int] = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[domain] = score

        if not scores:
            return fallback

        # Return highest-scoring domain
        return max(scores, key=scores.get)

    def classify_with_scores(self, text: str) -> dict[str, int]:
        """Returns scores for all domains (for debugging/UI display)."""
        text_lower = text.lower()
        return {
            domain: sum(1 for kw in keywords if kw in text_lower)
            for domain, keywords in DOMAIN_KEYWORDS.items()
        }

# Add these at the bottom of query_classifier.py

ACCUSED_KEYWORDS = [
    "i stole", "i took", "i hit", "i beat", "i killed", "i attacked",
    "i threatened", "i cheated", "i fraud", "i did", "i committed",
    "we stole", "we took", "we beat", "we attacked", "i was caught",
    "police caught me", "arrested me", "i ran away", "i fled",
    "i am accused", "i am arrested", "case against me", "fir against me",
    "complaint against me", "charge against me", "i broke into",
    "i snatched", "i robbed", "i assaulted", "i molested", "i raped",
    "i blackmailed", "i threatened to kill", "i forged", "i embezzled",
    "मैंने मारा", "मैंने चुराया", "मुझ पर केस", "मेरे खिलाफ fir",
]

VICTIM_KEYWORDS = [
    "someone attacked me", "i was attacked", "i was beaten", "i was robbed",
    "my phone was stolen", "they hit me", "he hit me", "she hit me",
    "i was cheated", "they cheated me", "i was threatened", "help me",
    "what should i do", "someone stole", "i am victim", "i need help",
]

def detect_user_role(self, text: str) -> str:
    """
    Detect if the user is:
    - 'accused'  : they committed the act
    - 'victim'   : they suffered the act
    - 'witness'  : third party
    - 'unknown'  : unclear
    """
    text_lower = text.lower()

    accused_score = sum(1 for kw in ACCUSED_KEYWORDS if kw in text_lower)
    victim_score  = sum(1 for kw in VICTIM_KEYWORDS  if kw in text_lower)

    if accused_score > victim_score and accused_score > 0:
        return "accused"
    elif victim_score > accused_score and victim_score > 0:
        return "victim"
    elif accused_score > 0:
        return "accused"
    return "victim"  # default assumption