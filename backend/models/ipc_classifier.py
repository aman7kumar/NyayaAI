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

def _load_label_map() -> Optional[dict]:
    """Load label map from label_map_v5.json."""
    search_paths = [
        Path(__file__).parent / "saved" / "ipc_classifier" / "label_map_v5.json",
        Path(__file__).parent / "label_map_v5.json",
        Path(__file__).parent.parent.parent / "training" / "data_prep" / "raw" / "label_map_v5.json",
        Path(__file__).parent.parent / "data" / "label_map_v5.json",
        Path("training/data_prep/raw/label_map_v5.json"),
        Path("label_map_v5.json"),
    ]
    for path in search_paths:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            n = data.get("n_labels", len(data.get("labels", [])))
            logger.info("Label map loaded from %s (%s labels)", path, n)
            return data
    logger.warning("label_map_v5.json not found - using default 20 labels")
    return None


def _extract_label_list(label_map: dict) -> list[str]:
    """Extract ordered labels from multiple supported label map formats."""
    if "labels" in label_map and isinstance(label_map["labels"], list):
        return [str(label) for label in label_map["labels"]]
    if "label2id" in label_map and isinstance(label_map["label2id"], dict):
        pairs = sorted(label_map["label2id"].items(), key=lambda item: int(item[1]))
        return [str(label) for label, _ in pairs]
    if all(isinstance(value, dict) for value in label_map.values()):
        return list(label_map.keys())
    return list(label_map.keys())


def _build_section_meta(label_map: dict) -> dict:
    """Convert label map formats to IPC_SECTION_META format."""
    meta = {}
    for label_key in _extract_label_list(label_map):
        info = label_map.get(label_key, {}) if isinstance(label_map.get(label_key), dict) else {}
        act = str(info.get("act", label_key.split("_")[0]))
        section = str(info.get("section", label_key.replace("_", " ")))
        title = str(info.get("title", label_key))
        desc = str(info.get("description", info.get("desc", f"{act} Section {section}")))
        punish = str(info.get("punishment", info.get("punish", "As per applicable law")))
        meta[label_key] = {
            "act": act,
            "section": f"{act} {section}",
            "title": title,
            "description": desc,
            "punishment": punish,
        }
    return meta


_DEFAULT_SECTION_META: dict[str, dict] = {
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

_label_map_raw = _load_label_map()
if _label_map_raw:
    IPC_SECTION_META = _build_section_meta(_label_map_raw)
    LABEL_LIST = _extract_label_list(_label_map_raw)
else:
    IPC_SECTION_META = _DEFAULT_SECTION_META
    LABEL_LIST = list(IPC_SECTION_META.keys())

NUM_LABELS = len(LABEL_LIST)
LABEL2IDX = {label: i for i, label in enumerate(LABEL_LIST)}
IDX2LABEL = {i: label for label, i in LABEL2IDX.items()}

logger.info("Classifier configured with %s labels", NUM_LABELS)


# ── Model Definitions ─────────────────────────────────────────────────────────

class InLegalBERTClassifier(nn.Module):
    """Best model — pre-trained on 5.4M Indian legal documents."""
    def __init__(
        self,
        num_labels: int,
        dropout: float = 0.2,
        model_name_or_path: str = "law-ai/InLegalBERT",
        local_files_only: bool = False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
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


class InLegalBERTClassifierV2(nn.Module):
    """
    Mean-pooled encoder + small MLP head — matches training/scripts/train_classifier_fixed.LegalClassifier.
    Weights: encoder from HF folder + classifier_head.pt
    """
    def __init__(self, encoder: AutoModel, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden // 2, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
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

def _build_keyword_map(label_list: list[str]) -> dict[str, list[str]]:
    """Build keyword map for available labels only."""
    base_map: dict[str, list[str]] = {
        "IPC_299": ["culpable homicide", "homicide not murder"],
        "IPC_302": ["murder", "killed", "killed deliberately", "shot dead", "stabbed to death"],
        "IPC_304": ["culpable homicide", "not amounting to murder"],
        "IPC_304A": ["rash act", "negligent act", "accidental death", "road accident death"],
        "IPC_307": ["attempt to murder", "tried to kill", "shot at me", "stabbed me"],
        "IPC_323": ["hit", "beat", "slap", "punch", "assault", "hurt", "injured", "beaten"],
        "IPC_324": ["stick", "rod", "knife", "weapon", "iron rod", "bat", "blade", "chain"],
        "IPC_325": ["grievous hurt", "serious injury", "broken bone", "permanent injury"],
        "IPC_326": ["acid attack", "acid thrown", "grievous hurt by weapon"],
        "IPC_354": ["molestation", "modesty", "groped", "eve teasing", "touched inappropriately"],
        "IPC_354A": ["sexual harassment", "demand for sexual favours", "unwelcome physical contact"],
        "IPC_354B": ["assault with intent to disrobe", "stripping clothes", "forcibly undressing"],
        "IPC_354C": ["voyeurism", "secretly photographing", "recording without consent"],
        "IPC_354D": ["stalking", "following woman", "monitoring online activity"],
        "IPC_363": ["kidnapping", "kidnapped minor", "abduction of child"],
        "IPC_364": ["kidnapping for ransom", "abduction for murder"],
        "IPC_366": ["abduction of woman", "kidnap for marriage", "elope forcibly"],
        "IPC_376": ["rape", "sexual assault", "sexually assaulted"],
        "IPC_376A": ["rape causing death", "rape in custody"],
        "IPC_376AB": ["rape of minor under 12"],
        "IPC_376B": ["rape by husband during separation"],
        "IPC_376C": ["rape by person in authority"],
        "IPC_376D": ["gang rape", "multiple accused rape"],
        "IPC_376E": ["repeat offender rape", "previous conviction rape"],
        "IPC_379": ["stolen", "theft", "stole", "pickpocket", "snatched", "missing property"],
        "IPC_380": ["broke into house", "house theft", "burglary", "home broken into"],
        "IPC_382": ["theft after preparation to hurt", "theft with violence"],
        "IPC_384": ["extortion", "blackmail money", "forced to pay"],
        "IPC_385": ["putting person in fear to commit extortion"],
        "IPC_390": ["robbery definition", "theft with violence"],
        "IPC_392": ["robbery", "robbed", "looted", "snatched forcefully"],
        "IPC_395": ["dacoity", "gang robbery", "armed gang looted"],
        "IPC_406": ["breach of trust", "misappropriated", "entrusted money kept"],
        "IPC_408": ["breach of trust by clerk", "employee misappropriation"],
        "IPC_409": ["breach of trust by public servant", "government funds misused"],
        "IPC_415": ["cheating definition", "dishonest inducement"],
        "IPC_417": ["cheating simple", "deceived by false promise"],
        "IPC_418": ["cheating with knowledge of harm"],
        "IPC_419": ["cheating by impersonation", "fake identity", "posed as someone", "impersonation"],
        "IPC_420": ["cheat", "fraud", "cheated", "fake seller", "scam", "online fraud"],
        "IPC_425": ["mischief", "damaged property", "broke property"],
        "IPC_426": ["mischief punishment"],
        "IPC_427": ["mischief causing damage 50 rupees"],
        "IPC_436": ["arson", "set fire", "burnt house", "fire to property"],
        "IPC_447": ["criminal trespass", "trespassing", "entered without permission"],
        "IPC_448": ["house trespass", "illegal entry into house"],
        "IPC_452": ["trespass after preparation to hurt"],
        "IPC_498A": ["dowry", "husband beats", "in-laws harassment", "cruelty by husband"],
        "IPC_499": ["defamation", "false statement", "reputation damage"],
        "IPC_500": ["punishment for defamation"],
        "IPC_501": ["printing defamatory matter"],
        "IPC_503": ["threat", "threaten", "death threat", "blackmail"],
        "IPC_504": ["intentional insult", "provoked to breach peace"],
        "IPC_506": ["criminal intimidation", "threatening messages", "life threat"],
        "IPC_507": ["anonymous criminal intimidation", "unsigned threat letter"],
        "IPC_509": ["obscene gesture", "vulgar", "insult modesty", "sexual remarks"],
        "IPC_120B": ["criminal conspiracy", "conspired together", "planned crime"],
        "IPC_147": ["rioting", "unlawful assembly riot"],
        "IPC_148": ["rioting with deadly weapon"],
        "IPC_149": ["common object unlawful assembly"],
        "IPC_153A": ["promoting enmity", "communal hatred", "religious hatred speech"],
        "IPC_159": ["affray", "fighting in public", "public fight"],
        "IPC_160": ["punishment for affray", "public brawl"],
        "IPC_279": ["rash driving", "negligent driving", "jumped signal"],
        "IPC_294": ["obscene acts in public", "public indecency"],
        "IPC_304B": ["dowry death", "wife died within 7 years", "dowry related death"],
        "IPC_305": ["abetment of suicide by minor"],
        "IPC_306": ["abetment of suicide", "instigated to commit suicide"],
        "IPC_308": ["attempt culpable homicide"],
        "IPC_312": ["miscarriage", "abortion without consent"],
        "IPC_313": ["abortion without consent", "forced abortion"],
        "IPC_337": ["rash act endangering life", "hurt by negligent act"],
        "IPC_338": ["grievous hurt by rash act", "serious injury by negligence"],
        "IPC_341": ["wrongful restraint", "blocked path", "prevented from moving"],
        "IPC_342": ["wrongful confinement", "locked up", "confined illegally"],
        "IPC_351": ["assault definition", "attempt to cause apprehension"],
        "IPC_352": ["assault or criminal force"],
        "IPC_353": ["assault on public servant", "attacked police officer"],
        "IPC_359": ["kidnapping definition"],
        "IPC_360": ["kidnapping from india"],
        "IPC_361": ["kidnapping from lawful guardian", "minor taken away"],
        "IPC_362": ["abduction definition"],
        "IPC_371": ["habitual dealing in slaves"],
        "IPC_375": ["rape definition", "sexual intercourse without consent"],
        "IPC_396": ["dacoity with murder"],
        "IPC_411": ["receiving stolen property", "handling stolen goods"],
        "IPC_413": ["habitual receiving of stolen property"],
        "IPC_414": ["assisting in concealing stolen property"],
        "IPC_429": ["killing cattle", "harm to animal"],
        "IPC_441": ["criminal trespass definition"],
        "IPC_442": ["house trespass definition"],
        "IPC_443": ["lurking house trespass"],
        "IPC_444": ["lurking house trespass at night"],
        "IPC_445": ["house breaking definition"],
        "IPC_446": ["house breaking at night"],
        "IPC_453": ["trespass with preparation to commit offence"],
        "IPC_454": ["lurking trespass to commit offence"],
        "IPC_455": ["lurking trespass at night"],
        "IPC_457": ["lurking trespass by night for offence"],
        "IPC_458": ["lurking trespass at night causing hurt"],
        "IPC_459": ["grievous hurt while committing trespass"],
        "IPC_460": ["all persons in house when murder committed"],
        "IPC_461": ["dishonestly breaking open receptacle"],
        "IPC_462": ["wrongfully breaking open receptacle by public servant"],
        "IPC_463": ["forgery definition"],
        "IPC_464": ["making false document"],
        "IPC_465": ["punishment for forgery"],
        "IPC_466": ["forgery of court record"],
        "IPC_467": ["forgery of valuable security"],
        "IPC_468": ["forgery for cheating", "forged document to cheat"],
        "IPC_469": ["forgery for reputation damage"],
        "IPC_470": ["forged document"],
        "IPC_471": ["using forged document as genuine"],
        "IPC_472": ["making counterfeit seal"],
        "IPC_474": ["possessing forged document"],
        "IPC_489A": ["counterfeiting currency", "fake notes", "counterfeit money"],
        "IPC_489B": ["selling counterfeit currency"],
        "IPC_489C": ["possessing counterfeit currency"],
        "CrPC_41": ["arrest without warrant", "police arrest without warrant"],
        "CrPC_41A": ["notice of appearance", "police notice to appear"],
        "CrPC_91": ["summons to produce document"],
        "CrPC_100": ["search of closed place"],
        "CrPC_144": ["prohibitory orders", "section 144", "curfew order"],
        "CrPC_145": ["dispute concerning land"],
        "CrPC_151": ["arrest to prevent cognizable offence"],
        "CrPC_154": ["fir", "police refuse fir", "not registering complaint"],
        "CrPC_156": ["police investigation", "cognizable offence investigation"],
        "CrPC_161": ["police examination of witness"],
        "CrPC_164": ["confession before magistrate", "statement before magistrate"],
        "CrPC_167": ["remand", "police custody remand", "judicial custody"],
        "CrPC_173": ["police report", "charge sheet", "final report"],
        "CrPC_177": ["ordinary place of inquiry and trial"],
        "CrPC_190": ["cognizance of offence by magistrate"],
        "CrPC_193": ["cognizance of offence by court of session"],
        "CrPC_197": ["prosecution of judges and public servants"],
        "CrPC_200": ["magistrate complaint", "private complaint court"],
        "CrPC_204": ["issue of process by magistrate"],
        "CrPC_207": ["supply of copy of police report to accused"],
        "CrPC_227": ["discharge", "discharge before framing of charges"],
        "CrPC_228": ["framing of charge"],
        "CrPC_313": ["examination of accused", "accused statement court"],
        "CrPC_320": ["compounding of offences", "compromise in court"],
        "CrPC_357": ["compensation to victims", "court compensation order"],
        "CrPC_374": ["appeal from conviction", "criminal appeal"],
        "CrPC_397": ["revision", "revision application high court"],
        "CrPC_406": ["transfer of cases", "case transfer high court"],
        "CrPC_437": ["bail bailable offence", "bail from magistrate"],
        "CrPC_438": ["anticipatory bail", "bail before arrest"],
        "CrPC_439": ["bail sessions court", "bail high court"],
        "CrPC_482": ["inherent powers high court", "quash fir", "quash proceedings"],
        "IT_65": ["tampering computer source documents"],
        "IT_66": ["computer related offences", "hacking", "cyber attack"],
        "IT_66A": ["sending offensive message online", "fake message online"],
        "IT_66B": ["receiving stolen computer resource"],
        "IT_66C": ["identity theft online", "password stolen", "account hacked"],
        "IT_66D": ["cheating by personation online", "fake profile fraud"],
        "IT_66E": ["violation of privacy", "morphed photos shared"],
        "IT_66F": ["cyber terrorism"],
        "IT_67": ["publishing obscene material online", "obscene content online"],
        "IT_67A": ["publishing sexually explicit content", "porn shared online"],
        "IT_67B": ["child pornography", "child sexual abuse material online"],
        "IT_72": ["breach of confidentiality", "data breach"],
        "POCSO_4": ["penetrative sexual assault on child"],
        "POCSO_6": ["aggravated penetrative sexual assault child"],
        "POCSO_7": ["sexual assault on child"],
        "POCSO_9": ["aggravated sexual assault child"],
        "POCSO_11": ["sexual harassment of child"],
        "POCSO_13": ["using child for pornography"],
        "SC_ST_3": ["caste abuse", "sc st act", "atrocity", "caste discrimination"],
        "SC_ST_3_1": ["untouchability practice", "denying water caste"],
        "DV_18": ["protection order domestic violence"],
        "DV_19": ["residence order domestic violence"],
        "DV_20": ["monetary relief domestic violence"],
        "NDPS_8": ["possession of drugs", "narcotic substance found"],
        "NDPS_20": ["cannabis offence", "ganja seized"],
        "NDPS_21": ["manufactured drug offence", "heroin seized"],
        "NDPS_22": ["psychotropic substance offence"],
        "NDPS_27": ["consumption of drugs", "drug addict"],
        "MV_184": ["dangerous driving", "rash driving motor vehicle"],
        "MV_185": ["drunk driving", "drunken driving"],
        "MV_279_IPC": ["rash driving on public way"],
        "Arms_25": ["unlicensed weapon", "illegal gun", "arms without licence"],
        "Arms_27": ["using arms to cause hurt"],
        "Dowry_3": ["giving or taking dowry", "dowry demand"],
        "Dowry_4": ["demanding dowry", "harassment for dowry"],
        "Consumer_27": ["complaint against company", "defective product"],
    }
    label_set = set(label_list)
    return {k: v for k, v in base_map.items() if k in label_set}


KEYWORD_MAP = _build_keyword_map(LABEL_LIST)


# ── Main Classifier Class ─────────────────────────────────────────────────────

class IPCClassifier:
    MODEL_DIR = Path(__file__).parent / "saved" / "ipc_classifier"
    THRESHOLD = 0.20
    MAX_LEN   = 128

    def __init__(
        self,
        model,
        tokenizer,
        model_type: str = "distilbert",
        label_thresholds: Optional[list[float]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.device = torch.device("cpu")
        self.label_thresholds = label_thresholds
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
        head_pt = directory / "classifier_head.pt"
        config_f = directory / "classifier_config.json"
        has_local_tokenizer = (directory / "tokenizer_config.json").exists()

        # ── V2: HF encoder folder + classifier_head.pt (train_classifier_fixed.py) ──
        if head_pt.exists():
            if not checkpoint.exists() or not has_local_tokenizer:
                logger.warning(
                    "V2 classifier needs pytorch_model.bin + tokenizer in %s. Using keyword-only.",
                    directory,
                )
                return cls(nn.Linear(1, NUM_LABELS), None, model_type="keyword_only")
            logger.info("Loading InLegalBERT V2 (encoder + classifier_head.pt)…")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(directory),
                    local_files_only=True,
                )
                encoder = AutoModel.from_pretrained(str(directory), local_files_only=True)
                dropout = 0.3
                label_thresholds: Optional[list[float]] = None
                meta_path = directory / "classifier_meta.json"
                if meta_path.exists():
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    dropout = float(meta.get("dropout", 0.3))
                    nj = meta.get("n_labels")
                    if nj is not None and int(nj) != NUM_LABELS:
                        logger.warning(
                            "classifier_meta n_labels=%s != NUM_LABELS=%s from label map",
                            nj, NUM_LABELS,
                        )
                    th = meta.get("thresholds")
                    if isinstance(th, list) and len(th) == NUM_LABELS:
                        label_thresholds = [float(x) for x in th]
                model = InLegalBERTClassifierV2(encoder, NUM_LABELS, dropout)
                state = torch.load(head_pt, map_location="cpu", weights_only=False)
                model.classifier.load_state_dict(state)
                logger.info("✅ InLegalBERT V2 loaded from %s", directory)
                return cls(
                    model, tokenizer, model_type="inlegalbert", label_thresholds=label_thresholds
                )
            except Exception as e:
                logger.error("InLegalBERT V2 load failed: %s", e, exc_info=True)
                return cls(nn.Linear(1, NUM_LABELS), None, model_type="keyword_only")

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

        # ── Load InLegalBERT (single-file full state dict) ─────────────────
        if model_type == "inlegalbert":
            logger.info("Loading InLegalBERT classifier...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(directory),
                    local_files_only=True,
                )
                model = InLegalBERTClassifier(
                    NUM_LABELS,
                    model_name_or_path=str(directory),
                    local_files_only=True,
                )
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
                    thr = threshold
                    if self.label_thresholds is not None and idx < len(self.label_thresholds):
                        thr = self.label_thresholds[idx]
                    if prob >= thr:
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