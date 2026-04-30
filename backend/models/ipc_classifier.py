"""
backend/models/ipc_classifier.py
Supports: DistilBERT, BERT, InLegalBERT
Auto-detects from saved classifier_config.json
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    DistilBertTokenizer,
    DistilBertModel,
    BertTokenizer,
    BertModel,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Label definitions ─────────────────────────────────────────────────────────

IPC_SECTION_META: dict[str, dict] = {
    "IPC_302":  {"act": "IPC",  "section": "IPC 302",  "title": "Murder",
                 "description": "Whoever commits murder shall be punished.",
                 "punishment": "Death or imprisonment for life, and fine."},
    "IPC_307":  {"act": "IPC",  "section": "IPC 307",  "title": "Attempt to Murder",
                 "description": "Whoever does any act with intention to cause death.",
                 "punishment": "Imprisonment up to 10 years, and fine."},
    "IPC_323":  {"act": "IPC",  "section": "IPC 323",  "title": "Voluntarily Causing Hurt",
                 "description": "Whoever voluntarily causes hurt shall be punished.",
                 "punishment": "Imprisonment up to 1 year, or fine up to Rs 1000, or both."},
    "IPC_324":  {"act": "IPC",  "section": "IPC 324",  "title": "Hurt by Dangerous Weapons",
                 "description": "Voluntarily causing hurt by dangerous instruments.",
                 "punishment": "Imprisonment up to 3 years, or fine, or both."},
    "IPC_354":  {"act": "IPC",  "section": "IPC 354",  "title": "Assault on Woman",
                 "description": "Assault intending to outrage modesty of a woman.",
                 "punishment": "Imprisonment 1 to 5 years, and fine."},
    "IPC_376":  {"act": "IPC",  "section": "IPC 376",  "title": "Rape",
                 "description": "Punishment for rape.",
                 "punishment": "Rigorous imprisonment not less than 10 years."},
    "IPC_379":  {"act": "IPC",  "section": "IPC 379",  "title": "Theft",
                 "description": "Whoever commits theft shall be punished.",
                 "punishment": "Imprisonment up to 3 years, or fine, or both."},
    "IPC_380":  {"act": "IPC",  "section": "IPC 380",  "title": "Theft in Dwelling House",
                 "description": "Theft in a building used as human dwelling.",
                 "punishment": "Imprisonment up to 7 years, and fine."},
    "IPC_392":  {"act": "IPC",  "section": "IPC 392",  "title": "Robbery",
                 "description": "Whoever commits robbery shall be punished.",
                 "punishment": "Rigorous imprisonment up to 10 years, and fine."},
    "IPC_395":  {"act": "IPC",  "section": "IPC 395",  "title": "Dacoity",
                 "description": "Whoever commits dacoity shall be punished.",
                 "punishment": "Imprisonment for life, or rigorous imprisonment up to 10 years."},
    "IPC_406":  {"act": "IPC",  "section": "IPC 406",  "title": "Criminal Breach of Trust",
                 "description": "Punishment for criminal breach of trust.",
                 "punishment": "Imprisonment up to 3 years, or fine, or both."},
    "IPC_420":  {"act": "IPC",  "section": "IPC 420",  "title": "Cheating",
                 "description": "Whoever cheats and dishonestly induces delivery of property.",
                 "punishment": "Imprisonment up to 7 years, and fine."},
    "IPC_498A": {"act": "IPC",  "section": "IPC 498A", "title": "Cruelty by Husband",
                 "description": "Cruelty by husband or relatives — harassment for dowry.",
                 "punishment": "Imprisonment up to 3 years, and fine."},
    "IPC_503":  {"act": "IPC",  "section": "IPC 503",  "title": "Criminal Intimidation",
                 "description": "Threatening another with injury to person or property.",
                 "punishment": "Imprisonment up to 2 years, or fine, or both."},
    "IPC_506":  {"act": "IPC",  "section": "IPC 506",  "title": "Punishment for Criminal Intimidation",
                 "description": "Punishment for criminal intimidation.",
                 "punishment": "Imprisonment up to 2 years, or fine, or both."},
    "IPC_509":  {"act": "IPC",  "section": "IPC 509",  "title": "Insulting Modesty of Woman",
                 "description": "Word or gesture intended to insult modesty of a woman.",
                 "punishment": "Imprisonment up to 3 years, and fine."},
    "CrPC_154": {"act": "CrPC", "section": "CrPC 154", "title": "FIR Registration",
                 "description": "Information in cognizable cases must be recorded by police.",
                 "punishment": "Procedural — compels police to register FIR."},
    "CrPC_156": {"act": "CrPC", "section": "CrPC 156", "title": "Police Power to Investigate",
                 "description": "Officer in charge may investigate any cognizable case.",
                 "punishment": "Procedural."},
    "CrPC_200": {"act": "CrPC", "section": "CrPC 200", "title": "Magistrate Complaint",
                 "description": "Magistrate taking cognizance shall examine the complainant.",
                 "punishment": "Procedural."},
    "CrPC_482": {"act": "CrPC", "section": "CrPC 482", "title": "Inherent Powers of High Court",
                 "description": "High Court may make orders to prevent abuse of court process.",
                 "punishment": "Procedural — relief from High Court."},
}

LABEL_LIST = list(IPC_SECTION_META.keys())
NUM_LABELS  = len(LABEL_LIST)
LABEL2IDX   = {label: i for i, label in enumerate(LABEL_LIST)}
IDX2LABEL   = {i: label for label, i in LABEL2IDX.items()}


# ── Model Definitions ─────────────────────────────────────────────────────────

class InLegalBERTClassifier(nn.Module):
    """Best model — pre-trained on 5.4M Indian legal documents."""
    def __init__(self, num_labels: int, dropout: float = 0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained("law-ai/InLegalBERT")
        hidden    = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),   nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


class DistilBertIPCClassifier(nn.Module):
    """Fallback — lighter model for CPU."""
    def __init__(self, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.distilbert     = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
        hidden              = self.distilbert.config.hidden_size
        self.pre_classifier = nn.Linear(hidden, 256)
        self.dropout        = nn.Dropout(dropout)
        self.classifier     = nn.Linear(256, num_labels)
        self.relu           = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        out     = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = out.last_hidden_state[:, 0, :]
        x       = self.relu(self.pre_classifier(cls_out))
        x       = self.dropout(x)
        return self.classifier(x)


# ── Keyword Fallback ──────────────────────────────────────────────────────────

KEYWORD_MAP = {
    "IPC_302":  ["murder", "killed", "kill", "death", "shot dead", "stabbed to death"],
    "IPC_307":  ["attempt murder", "tried to kill", "shot at", "stabbed me", "survived attack"],
    "IPC_323":  ["hit", "beat", "slap", "punch", "assault", "hurt", "injured", "beaten", "attacked"],
    "IPC_324":  ["stick", "rod", "knife", "weapon", "iron rod", "bat", "blade", "chain", "dangerous"],
    "IPC_354":  ["molestation", "modesty", "groped", "eve teasing", "sexual harassment", "touched inappropriately"],
    "IPC_376":  ["rape", "sexual assault", "sexually assaulted"],
    "IPC_379":  ["stolen", "theft", "stole", "pickpocket", "snatched", "missing", "took my"],
    "IPC_380":  ["broke into", "burglary", "house theft", "home broken", "dwelling"],
    "IPC_392":  ["robbery", "robbed", "rob", "looted", "armed"],
    "IPC_395":  ["dacoity", "gang robbery", "armed gang"],
    "IPC_406":  ["breach of trust", "misappropriate", "embezzle", "entrusted money"],
    "IPC_420":  ["cheat", "fraud", "cheated", "deceived", "fake", "scam", "online fraud"],
    "IPC_498A": ["dowry", "husband beats", "in-laws", "cruelty by husband", "domestic violence"],
    "IPC_503":  ["threat", "threaten", "threatened", "blackmail", "death threat", "kill me"],
    "IPC_506":  ["criminal intimidation", "threatening messages", "life threat"],
    "IPC_509":  ["obscene gesture", "vulgar", "insult modesty", "sexual remarks"],
    "CrPC_154": ["fir", "police refuse", "not registering complaint", "refuse to file"],
    "CrPC_156": ["police investigate", "investigation", "cognizable"],
    "CrPC_200": ["magistrate complaint", "private complaint", "court complaint"],
    "CrPC_482": ["high court", "quash", "inherent powers"],
}


# ── Main Classifier Class ─────────────────────────────────────────────────────

class IPCClassifier:
    MODEL_DIR = Path(__file__).parent / "saved" / "ipc_classifier"
    THRESHOLD = 0.20
    MAX_LEN   = 128

    def __init__(self, model, tokenizer, model_type: str = "distilbert"):
        self.model      = model
        self.tokenizer  = tokenizer
        self.model_type = model_type
        self.device     = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"IPCClassifier ready — model: {model_type}, device: {self.device}")

    @classmethod
    def load(cls, model_dir: Optional[str] = None) -> "IPCClassifier":
        """
        Load classifier. Priority:
        1. InLegalBERT (if trained and saved)
        2. DistilBERT (if trained and saved)
        3. Keyword-only fallback (if no model trained yet)
        """
        directory = Path(model_dir) if model_dir else cls.MODEL_DIR
        if not directory.is_absolute():
            directory = (Path(__file__).parent / directory).resolve()
        directory = directory.resolve()
        nested_directory = directory / "backend" / "models" / "saved" / "ipc_classifier"
        if (not (directory / "classifier_config.json").exists()) and nested_directory.exists():
            logger.warning("Using nested classifier artifact path: %s", nested_directory)
            directory = nested_directory
        checkpoint = directory / "pytorch_model.bin"
        config_f = directory / "classifier_config.json"
        has_local_tokenizer = (directory / "tokenizer_config.json").exists()
        if not checkpoint.exists() or not has_local_tokenizer:
            logger.warning(
                "Local classifier artifacts incomplete in %s. Using keyword-only classifier.",
                directory,
            )
            dummy_model = nn.Linear(1, NUM_LABELS)
            return cls(dummy_model, None, model_type="keyword_only")

        model_type = "distilbert"  # default

        if config_f.exists():
            with open(config_f, encoding="utf-8") as f:
                saved_cfg = json.load(f)
            base = saved_cfg.get("base_model", "").lower()
            if "inlegalbert" in base or "law-ai" in base:
                model_type = "inlegalbert"
            elif "distilbert" in base:
                model_type = "distilbert"

        # ── Load InLegalBERT ──────────────────────────────────────────────
        if model_type == "inlegalbert":
            logger.info("Loading InLegalBERT classifier...")
            try:
                # InLegalBERT base weights are not bundled in this repository.
                # Avoid runtime network downloads and use keyword mode instead.
                raise RuntimeError("InLegalBERT local base weights unavailable.")
                tokenizer = AutoTokenizer.from_pretrained(
                    str(directory),
                    local_files_only=True,
                )
                model = InLegalBERTClassifier(NUM_LABELS)
                if checkpoint.exists():
                    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    model.load_state_dict(state)
                    logger.info(f"✅ InLegalBERT loaded from {checkpoint}")
                else:
                    logger.warning("⚠️  No checkpoint found. Using untrained InLegalBERT.")
                return cls(model, tokenizer, model_type="inlegalbert")
            except Exception as e:
                logger.error(f"InLegalBERT load failed: {e}. Falling back to DistilBERT.")
                model_type = "distilbert"

        # ── Load DistilBERT ───────────────────────────────────────────────
        logger.info("Loading DistilBERT classifier...")
        try:
            tokenizer = DistilBertTokenizer.from_pretrained(
                str(directory),
                local_files_only=True,
            )
            model = DistilBertIPCClassifier(NUM_LABELS)
            if checkpoint.exists():
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model.load_state_dict(state)
                logger.info(f"✅ DistilBERT loaded from {checkpoint}")
            else:
                logger.warning("⚠️  No checkpoint. Keyword fallback will be used.")
            return cls(model, tokenizer, model_type="distilbert")
        except Exception as e:
            logger.error(f"DistilBERT load failed: {e}")
            # Return dummy instance — keyword fallback will handle predictions
            dummy_model     = nn.Linear(1, NUM_LABELS)
            dummy_tokenizer = None
            return cls(dummy_model, dummy_tokenizer, model_type="keyword_only")

    def predict(
        self,
        text: str,
        context_chunks: Optional[list] = None,
        threshold: float = THRESHOLD,
    ) -> list[dict]:
        """
        Predict IPC/CrPC sections.
        Uses model predictions + keyword fallback for robustness.
        """
        model_results   = []
        keyword_results = self._keyword_fallback(text)

        # Try model prediction
        if self.model_type != "keyword_only" and self.tokenizer is not None:
            try:
                if context_chunks:
                    context_str = " ".join(c["text"][:150] for c in context_chunks[:2])
                    combined    = f"{text} [SEP] {context_str}"
                else:
                    combined = text

                encoding = self.tokenizer(
                    combined,
                    max_length=self.MAX_LEN,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                with torch.no_grad():
                    logits = self.model(
                        input_ids=encoding["input_ids"].to(self.device),
                        attention_mask=encoding["attention_mask"].to(self.device),
                    )
                    probs = torch.sigmoid(logits).cpu().numpy()[0]

                for idx, prob in enumerate(probs):
                    if prob >= threshold:
                        label_key = IDX2LABEL[idx]
                        meta      = IPC_SECTION_META.get(label_key, {})
                        model_results.append({
                            **meta,
                            "label_key":    label_key,
                            "confidence":   float(prob),
                            "statute_text": meta.get("description", ""),
                        })

            except Exception as e:
                logger.warning(f"Model prediction failed: {e}. Using keyword fallback.")

        # Merge model results with keyword results
        if model_results:
            # Model results take priority — boost with keyword matches
            found_keys = {r["label_key"] for r in model_results}
            for kr in keyword_results:
                if kr["label_key"] not in found_keys:
                    # Add keyword match with lower confidence
                    kr["confidence"] = min(kr["confidence"], 0.45)
                    model_results.append(kr)
            results = model_results
        else:
            # No model results — use keyword fallback
            results = keyword_results

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:5]

    def _keyword_fallback(self, text: str) -> list[dict]:
        """Rule-based keyword matching — always runs as safety net."""
        text_lower = text.lower()
        matched    = []

        for label_key, keywords in KEYWORD_MAP.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                meta = IPC_SECTION_META.get(label_key, {})
                matched.append({
                    **meta,
                    "label_key":    label_key,
                    "confidence":   min(0.40 + score * 0.08, 0.75),
                    "statute_text": meta.get("description", ""),
                })

        matched.sort(key=lambda x: x["confidence"], reverse=True)
        return matched[:5]