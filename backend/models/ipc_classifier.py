"""
backend/models/ipc_classifier.py
==================================
Fine-tuned BERT-based multi-label classifier for IPC / CrPC section prediction.

Architecture:
  - Base: bert-base-multilingual-cased  (handles English + Hindi + other Indian langs)
  - Head: Multi-label classification head (sigmoid output, threshold=0.35)
  - Training: See training/scripts/train_classifier.py

Usage after training:
  clf = IPCClassifier.load(model_dir="models/saved/ipc_classifier")
  predictions = clf.predict("Someone assaulted me at the market")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

logger = logging.getLogger(__name__)

# ── IPC / CrPC Section Label Mapping ─────────────────────────────────────────
# This list covers the most commonly invoked sections.
# Expand with your full labeled dataset.
IPC_SECTION_META: dict[str, dict] = {
    "IPC_302": {
        "act": "IPC", "section": "IPC 302",
        "title": "Murder",
        "description": "Punishment for intentionally causing death of another person.",
        "punishment": "Death or imprisonment for life, and fine.",
    },
    "IPC_307": {
        "act": "IPC", "section": "IPC 307",
        "title": "Attempt to Murder",
        "description": "Whoever does any act with such intention or knowledge that if death were caused, the act would constitute murder.",
        "punishment": "Imprisonment up to 10 years, and fine.",
    },
    "IPC_323": {
        "act": "IPC", "section": "IPC 323",
        "title": "Voluntarily Causing Hurt",
        "description": "Whoever voluntarily causes hurt shall be punished.",
        "punishment": "Imprisonment up to 1 year, or fine up to ₹1000, or both.",
    },
    "IPC_324": {
        "act": "IPC", "section": "IPC 324",
        "title": "Voluntarily Causing Hurt by Dangerous Weapons",
        "description": "Voluntarily causing hurt by means of any instrument for shooting, stabbing, or cutting.",
        "punishment": "Imprisonment up to 3 years, or fine, or both.",
    },
    "IPC_354": {
        "act": "IPC", "section": "IPC 354",
        "title": "Assault or Criminal Force on Woman",
        "description": "Assault or use of criminal force on a woman, intending to outrage her modesty.",
        "punishment": "Imprisonment from 1 to 5 years, and fine.",
    },
    "IPC_376": {
        "act": "IPC", "section": "IPC 376",
        "title": "Rape",
        "description": "Punishment for rape.",
        "punishment": "Rigorous imprisonment not less than 10 years, may extend to life.",
    },
    "IPC_379": {
        "act": "IPC", "section": "IPC 379",
        "title": "Theft",
        "description": "Whoever commits theft shall be punished.",
        "punishment": "Imprisonment up to 3 years, or fine, or both.",
    },
    "IPC_380": {
        "act": "IPC", "section": "IPC 380",
        "title": "Theft in Dwelling House",
        "description": "Theft in a building, tent, or vessel used as a human dwelling.",
        "punishment": "Imprisonment up to 7 years, and fine.",
    },
    "IPC_392": {
        "act": "IPC", "section": "IPC 392",
        "title": "Robbery",
        "description": "Whoever commits robbery shall be punished.",
        "punishment": "Rigorous imprisonment up to 10 years, and fine.",
    },
    "IPC_395": {
        "act": "IPC", "section": "IPC 395",
        "title": "Dacoity",
        "description": "Whoever commits dacoity shall be punished.",
        "punishment": "Imprisonment for life, or rigorous imprisonment up to 10 years, and fine.",
    },
    "IPC_406": {
        "act": "IPC", "section": "IPC 406",
        "title": "Criminal Breach of Trust",
        "description": "Punishment for criminal breach of trust.",
        "punishment": "Imprisonment up to 3 years, or fine, or both.",
    },
    "IPC_420": {
        "act": "IPC", "section": "IPC 420",
        "title": "Cheating and Dishonestly Inducing Delivery of Property",
        "description": "Whoever cheats and thereby dishonestly induces the person deceived to deliver property.",
        "punishment": "Imprisonment up to 7 years, and fine.",
    },
    "IPC_498A": {
        "act": "IPC", "section": "IPC 498A",
        "title": "Husband or Relative of Husband Subjecting Woman to Cruelty",
        "description": "Cruelty by husband or relatives — harassment for dowry.",
        "punishment": "Imprisonment up to 3 years, and fine.",
    },
    "IPC_503": {
        "act": "IPC", "section": "IPC 503",
        "title": "Criminal Intimidation",
        "description": "Threatening another with injury to person, reputation, or property.",
        "punishment": "Imprisonment up to 2 years, or fine, or both.",
    },
    "IPC_506": {
        "act": "IPC", "section": "IPC 506",
        "title": "Punishment for Criminal Intimidation",
        "description": "Punishment for criminal intimidation.",
        "punishment": "Imprisonment up to 2 years, or fine, or both.",
    },
    "IPC_509": {
        "act": "IPC", "section": "IPC 509",
        "title": "Word, Gesture or Act Intended to Insult Modesty of a Woman",
        "description": "Any word/gesture intended to insult the modesty of a woman.",
        "punishment": "Imprisonment up to 3 years, and fine.",
    },
    "CrPC_154": {
        "act": "CrPC", "section": "CrPC 154",
        "title": "Information in Cognizable Cases (FIR)",
        "description": "Every information relating to the commission of a cognizable offence shall be recorded.",
        "punishment": "Procedural — compels police to register FIR.",
    },
    "CrPC_156": {
        "act": "CrPC", "section": "CrPC 156",
        "title": "Police Officer's Power to Investigate Cognizable Case",
        "description": "Officer in charge may investigate any cognizable case without order from Magistrate.",
        "punishment": "Procedural.",
    },
    "CrPC_200": {
        "act": "CrPC", "section": "CrPC 200",
        "title": "Examination of Complainant (Magistrate Complaint)",
        "description": "Magistrate taking cognizance of an offence on complaint shall examine the complainant.",
        "punishment": "Procedural.",
    },
    "CrPC_482": {
        "act": "CrPC", "section": "CrPC 482",
        "title": "Inherent Powers of High Court",
        "description": "High Court may make such orders as necessary to prevent abuse of process of court.",
        "punishment": "Procedural — relief from High Court.",
    },
}

LABEL_LIST = list(IPC_SECTION_META.keys())
NUM_LABELS = len(LABEL_LIST)
LABEL2IDX = {label: i for i, label in enumerate(LABEL_LIST)}
IDX2LABEL = {i: label for label, i in LABEL2IDX.items()}


# ── Model Definition ──────────────────────────────────────────────────────────

class BertIPCClassifier(nn.Module):
    """
    Multi-label BERT classifier for IPC/CrPC section prediction.
    """

    def __init__(self, num_labels: int, dropout: float = 0.3):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        hidden_size = self.bert.config.hidden_size  # 768

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.pooler_output          # [batch, 768]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)        # [batch, num_labels]
        return logits                           # raw logits (BCEWithLogitsLoss)


# ── Wrapper Class ─────────────────────────────────────────────────────────────

class IPCClassifier:
    """
    High-level wrapper around BertIPCClassifier.
    Handles tokenization, inference, and decoding to IPC section objects.
    """

    MODEL_DIR = Path("backend/models/saved/ipc_classifier")
    THRESHOLD = 0.15      # sigmoid threshold for multi-label prediction
    MAX_LEN   = 256

    def __init__(self, model: BertIPCClassifier, tokenizer: BertTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"IPCClassifier running on {self.device}")

    # ── Factory Methods ───────────────────────────────────────────────────
    """
    @classmethod
    def load(cls, model_dir: Optional[str] = None) -> "IPCClassifier":
        
        #Load fine-tuned model from disk.
        #Falls back to base (untrained) model if checkpoint not found.
        
        directory = Path(model_dir) if model_dir else cls.MODEL_DIR

        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        pytorch_model = BertIPCClassifier(num_labels=NUM_LABELS)

        checkpoint = directory / "pytorch_model.bin"
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location="cpu")
            pytorch_model.load_state_dict(state)
            logger.info(f"✅  IPCClassifier loaded from {checkpoint}")
        else:
            logger.warning(
                f"⚠️  No checkpoint found at {checkpoint}. "
                "Using untrained model. Run training/scripts/train_classifier.py first."
            )

        return cls(pytorch_model, tokenizer)
        """
    @classmethod
    def load(cls, model_dir: Optional[str] = None) -> "IPCClassifier":
        """
        Load fine-tuned model from disk.
        Auto-detects BERT or DistilBERT based on saved classifier_config.json.
        """
        import json
        from transformers import DistilBertTokenizer, DistilBertModel
        import torch.nn as nn

        directory  = Path(model_dir) if model_dir else cls.MODEL_DIR
        checkpoint = directory / "pytorch_model.bin"
        config_f   = directory / "classifier_config.json"

        # Detect which model was trained
        use_distilbert = False
        if config_f.exists():
            with open(config_f) as f:
                saved_cfg = json.load(f)
            if "distilbert" in saved_cfg.get("base_model", "").lower():
                use_distilbert = True

        if use_distilbert:
            # ── Load DistilBERT classifier ──────────────────────────
            logger.info("Loading DistilBERT classifier ...")

            class _DistilBertCls(nn.Module):
                def __init__(self, num_labels):
                    super().__init__()
                    self.distilbert     = DistilBertModel.from_pretrained(
                        "distilbert-base-multilingual-cased"
                    )
                    hidden              = self.distilbert.config.hidden_size
                    self.pre_classifier = nn.Linear(hidden, 256)
                    self.dropout        = nn.Dropout(0.3)
                    self.classifier     = nn.Linear(256, num_labels)
                    self.relu           = nn.ReLU()

                def forward(self, input_ids, attention_mask):
                    out     = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
                    cls_out = out.last_hidden_state[:, 0, :]
                    x       = self.relu(self.pre_classifier(cls_out))
                    x       = self.dropout(x)
                    return self.classifier(x)

                def predict_proba(self, input_ids, attention_mask):
                    logits = self.forward(input_ids, attention_mask)
                    return torch.sigmoid(logits)

            tokenizer     = DistilBertTokenizer.from_pretrained(
                str(directory) if (directory / "vocab.txt").exists()
                else "distilbert-base-multilingual-cased"
            )
            pytorch_model = _DistilBertCls(NUM_LABELS)

            if checkpoint.exists():
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                pytorch_model.load_state_dict(state)
                logger.info(f"✅  DistilBERT classifier loaded from {checkpoint}")
            else:
                logger.warning("⚠️  No checkpoint found. Using untrained DistilBERT.")

        else:
            # ── Load original BERT classifier ───────────────────────
            logger.info("Loading BERT classifier ...")
            tokenizer     = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
            pytorch_model = BertIPCClassifier(num_labels=NUM_LABELS)

            if checkpoint.exists():
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                pytorch_model.load_state_dict(state)
                logger.info(f"✅  BERT classifier loaded from {checkpoint}")
            else:
                logger.warning("⚠️  No checkpoint found. Using untrained BERT.")

        # Wrap in IPCClassifier — update predict() to handle both architectures
        instance = cls(pytorch_model, tokenizer)
        instance._use_distilbert = use_distilbert
        return instance

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(
        self,
        text: str,
        context_chunks: Optional[list[dict]] = None,
        threshold: float = THRESHOLD,
    ) -> list[dict]:
        """
        Predict IPC/CrPC sections for the given text.

        Args:
            text: Query / FIR text (English)
            context_chunks: RAG-retrieved statute chunks (appended for context)
            threshold: Sigmoid confidence threshold

        Returns:
            List of section dicts sorted by confidence (descending)
        """
        # Optionally append retrieved statute context
        if context_chunks:
            context_str = " ".join(c["text"][:200] for c in context_chunks[:3])
            combined = f"{text} [SEP] {context_str}"
        else:
            combined = text
        """
        encoding = self.tokenizer(
            combined,
            max_length=self.MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        """
        max_len  = 64 if getattr(self, "_use_distilbert", False) else self.MAX_LEN
        encoding = self.tokenizer(
            combined,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device),
            )
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # [num_labels]

        results = []
        for idx, prob in enumerate(probs):
            if prob >= threshold:
                label_key = IDX2LABEL[idx]
                meta = IPC_SECTION_META.get(label_key, {})
                results.append({
                    **meta,
                    "label_key": label_key,
                    "confidence": float(prob),
                    "statute_text": meta.get("description", ""),
                })

        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)

        # If model finds nothing, use keyword-based fallback
        if not results:
            results = self._keyword_fallback(text)

        # Return top-5 at most
        return results[:5]
    def _keyword_fallback(self, text: str) -> list[dict]:
        """Rule-based fallback when model confidence is low."""
        text_lower = text.lower()
        matched = []

        keyword_map = {
            "IPC_323":  ["hit", "beat", "assault", "hurt", "slap", "punch", "attack", "injured", "injury", "beaten"],
            "IPC_324":  ["stick", "rod", "knife", "weapon", "blade", "iron", "bat", "chain", "dangerous"],
            "IPC_379":  ["stolen", "theft", "stole", "pickpocket", "snatched", "missing", "robbed", "took"],
            "IPC_380":  ["broke into", "burglary", "house", "home", "dwelling", "break in"],
            "IPC_392":  ["robbery", "robbed", "rob", "looted"],
            "IPC_420":  ["cheat", "fraud", "cheated", "deceived", "fake", "scam", "money", "online fraud"],
            "IPC_498A": ["dowry", "husband", "in-laws", "cruelty", "wife", "domestic", "harass"],
            "IPC_503":  ["threat", "threaten", "threatened", "kill", "harm", "blackmail"],
            "IPC_506":  ["intimidat", "threaten", "death threat", "life threat"],
            "IPC_354":  ["molestation", "modesty", "outrage", "grope", "sexual harassment", "inappropriate touch"],
            "IPC_376":  ["rape", "sexual assault", "sexually assaulted"],
            "IPC_302":  ["murder", "killed", "kill", "dead", "death"],
            "IPC_307":  ["attempt to murder", "tried to kill", "shot at", "stabbed"],
            "CrPC_154": ["fir", "complaint", "police", "refuse", "register"],
        }

        for label_key, keywords in keyword_map.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                meta = IPC_SECTION_META.get(label_key, {})
                matched.append({
                    **meta,
                    "label_key":    label_key,
                    "confidence":   min(0.5 + score * 0.1, 0.95),
                    "statute_text": meta.get("description", ""),
                })

        matched.sort(key=lambda x: x["confidence"], reverse=True)
        return matched[:5]
