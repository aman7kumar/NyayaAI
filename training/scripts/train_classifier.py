"""
training/scripts/train_classifier.py
Fine-tunes InLegalBERT for IPC/CrPC multi-label section prediction.

On laptop (CPU):  python train_classifier.py  [testing only — slow]
On college GPU:   python train_classifier.py  [actual training]
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# ── Comment this line out when running on GPU server ──
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Default Config ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "base_model":    "law-ai/InLegalBERT",
    "train_file":    "backend/data/classifier_train.jsonl",
    "label_map":     "training/data_prep/raw/label_map_v5.json",
    "output_dir":    "backend/models/saved/ipc_classifier",
    "max_len":       128,
    "batch_size":    8,        # 8 for CPU/low RAM, 32 for A100 GPU
    "epochs":        8,
    "learning_rate": 2e-5,
    "warmup_ratio":  0.1,
    "weight_decay":  0.01,
    "threshold":     0.25,
    "label_smoothing": 0.1,
    "seed":          42,
    "num_workers":   0,        # 0 for Windows, 4 for Linux GPU server
}


def load_label_map(config: dict) -> tuple[list[str], dict[str, int]]:
    """Load label list and indices from configured label map."""
    label_map_path = config.get("label_map", "")
    search_paths = [
        Path(label_map_path) if label_map_path else None,
        Path("training/data_prep/raw/label_map_v5.json"),
        Path("backend/models/label_map_v5.json"),
    ]
    for path in search_paths:
        if path and path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if "labels" in data and isinstance(data["labels"], list):
                labels = [str(label) for label in data["labels"]]
            elif "label2id" in data and isinstance(data["label2id"], dict):
                pairs = sorted(data["label2id"].items(), key=lambda item: int(item[1]))
                labels = [str(label) for label, _ in pairs]
            else:
                labels = list(data.keys())
            logger.info("Label map loaded: %s (%s labels)", path, len(labels))
            return labels, {label: i for i, label in enumerate(labels)}

    from backend.models.ipc_classifier import LABEL_LIST, LABEL2IDX
    logger.warning("Label map file not found. Falling back to ipc_classifier labels.")
    return LABEL_LIST, LABEL2IDX


# ── Model Definition ───────────────────────────────────────────────────────────

class InLegalBERTClassifier(nn.Module):
    """
    InLegalBERT fine-tuned for IPC/CrPC multi-label classification.
    Pre-trained on 5.4 million Indian legal documents.
    Uses mean pooling over all tokens for better representation.
    """

    def __init__(self, num_labels: int, dropout: float = 0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained("law-ai/InLegalBERT")
        hidden    = self.bert.config.hidden_size  # 768

        # Deep classification head with LayerNorm
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling — averages all token embeddings
        # Better than just [CLS] token for classification
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1)
        pooled = pooled / mask.sum(1).clamp(min=1e-9)

        return self.classifier(pooled)


# ── Dataset ────────────────────────────────────────────────────────────────────

class IPCDataset(Dataset):
    def __init__(self, samples: list, tokenizer, max_len: int = 128):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        enc  = self.tokenizer(
            item["text"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(item["labels"], dtype=torch.float32),
        }


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    preds:     np.ndarray,
    targets:   np.ndarray,
    threshold: float,
) -> dict:
    binary    = (preds >= threshold).astype(int)
    tp        = int((binary & targets.astype(bool)).sum())
    fp        = int((binary & ~targets.astype(bool)).sum())
    fn        = int((~binary.astype(bool) & targets.astype(bool)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    # Per-label F1 (so we know which sections are predicted well)
    per_label = []
    for i in range(preds.shape[1]):
        p    = preds[:, i]
        t    = targets[:, i]
        b    = (p >= threshold).astype(int)
        tp_i = int((b & t.astype(bool)).sum())
        fp_i = int((b & ~t.astype(bool)).sum())
        fn_i = int((~b.astype(bool) & t.astype(bool)).sum())
        p_i  = tp_i / (tp_i + fp_i + 1e-8)
        r_i  = tp_i / (tp_i + fn_i + 1e-8)
        f1_i = 2 * p_i * r_i / (p_i + r_i + 1e-8)
        per_label.append(float(f1_i))

    return {
        "micro_f1":     float(f1),
        "macro_f1":     float(np.mean(per_label)),
        "precision":    float(precision),
        "recall":       float(recall),
        "per_label_f1": per_label,
    }


# ── Training Loop ──────────────────────────────────────────────────────────────

def train(config: dict):
    # ── Device setup ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU:   {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        # Reduce settings for CPU to avoid OOM
        config["batch_size"]  = min(config["batch_size"], 8)
        config["num_workers"] = 0
        logger.info("Running on CPU — batch_size capped at 8")

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # ── Load data ──────────────────────────────────────────────────────────
    train_file = Path(config["train_file"])
    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        logger.error("Run: python training/data_prep/collect_training_data.py")
        logger.error("Then: python training/scripts/prepare_dataset.py")
        sys.exit(1)

    LABEL_LIST, LABEL2IDX = load_label_map(config)
    NUM_LABELS = len(LABEL_LIST)
    logger.info("Training with %s labels", NUM_LABELS)

    samples = []
    with open(train_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "labels" in item:
                labels = item["labels"]
                if len(labels) != NUM_LABELS:
                    if len(labels) < NUM_LABELS:
                        labels = labels + [0] * (NUM_LABELS - len(labels))
                    else:
                        labels = labels[:NUM_LABELS]
                item["labels"] = labels
            elif "ipc_sections" in item:
                labels = [0] * NUM_LABELS
                for sec in str(item["ipc_sections"]).split(","):
                    sec = sec.strip()
                    if sec in LABEL2IDX:
                        labels[LABEL2IDX[sec]] = 1
                item["labels"] = labels
            else:
                continue
            samples.append(item)

    if len(samples) < 100:
        logger.error(f"Too few samples: {len(samples)}. Need at least 100.")
        logger.error("Run collect_training_data.py to get more data.")
        sys.exit(1)

    logger.info(f"Total samples: {len(samples)}")
    np.random.shuffle(samples)

    n = len(samples)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    logger.info(
        "Train: %s | Val: %s | Test: %s",
        len(train_samples),
        len(val_samples),
        len(test_samples),
    )

    # ── Tokenizer ──────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {config['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])

    # ── DataLoaders ────────────────────────────────────────────────────────
    train_loader = DataLoader(
        IPCDataset(train_samples, tokenizer, config["max_len"]),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        IPCDataset(val_samples, tokenizer, config["max_len"]),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ──────────────────────────────────────────────────────────────
    logger.info(f"Loading InLegalBERT: {config['base_model']}")
    model = InLegalBERTClassifier(NUM_LABELS).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    # Mixed precision — only on GPU (A100 supports bfloat16)
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Optimizer ──────────────────────────────────────────────────────────
    # Different learning rates:
    # BERT layers: lower LR (already pre-trained)
    # Classifier head: higher LR (newly initialized)
    bert_params = [p for n, p in model.named_parameters() if "bert" in n]
    head_params = [p for n, p in model.named_parameters() if "bert" not in n]

    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": config["learning_rate"]},
        {"params": head_params, "lr": config["learning_rate"] * 5},
    ], weight_decay=config["weight_decay"])

    total_steps  = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    label_smoothing = float(config.get("label_smoothing", 0.1))
    if label_smoothing > 0:
        class SmoothBCELoss(nn.Module):
            def __init__(self, smoothing: float = 0.1):
                super().__init__()
                self.smoothing = smoothing

            def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                smooth_targets = targets * (1 - self.smoothing) + self.smoothing * 0.5
                return nn.functional.binary_cross_entropy_with_logits(logits, smooth_targets)

        criterion = SmoothBCELoss(smoothing=label_smoothing)
        logger.info("Using label smoothing: %s", label_smoothing)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # ── Output directory ───────────────────────────────────────────────────
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    history = []

    # ── Training epochs ────────────────────────────────────────────────────
    for epoch in range(1, config["epochs"] + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch}/{config['epochs']}")

        # ── Train phase ────────────────────────────────────────────────
        model.train()
        train_loss  = 0.0
        train_steps = 0

        for batch in tqdm(train_loader, desc=f"  Training epoch {epoch}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_amp and scaler:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            train_loss  += loss.item()
            train_steps += 1

        # ── Validation phase ───────────────────────────────────────────
        model.eval()
        all_preds, all_targets, val_loss = [], [], 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Validating epoch {epoch}"):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                logits   = model(input_ids, attention_mask)
                val_loss += criterion(logits, labels).item()

                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        all_preds   = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        metrics     = compute_metrics(all_preds, all_targets, config["threshold"])

        avg_train = train_loss / train_steps
        avg_val   = val_loss   / len(val_loader)

        logger.info(
            f"  Train Loss: {avg_train:.4f} | "
            f"Val Loss:   {avg_val:.4f} | "
            f"Micro F1:   {metrics['micro_f1']:.4f} | "
            f"Macro F1:   {metrics['macro_f1']:.4f} | "
            f"P: {metrics['precision']:.4f} | "
            f"R: {metrics['recall']:.4f}"
        )

        # Show per-label F1 for sections that are being predicted
        logger.info("  Per-label F1 (sections with predictions > 0):")
        for i, f1 in enumerate(metrics["per_label_f1"]):
            if f1 > 0.01:
                logger.info(f"    {LABEL_LIST[i]:20s}: {f1:.3f}")

        history.append({
            "epoch":      epoch,
            "train_loss": avg_train,
            "val_loss":   avg_val,
            **metrics,
        })

        # ── Save best model ────────────────────────────────────────────
        if metrics["micro_f1"] > best_f1:
            best_f1 = metrics["micro_f1"]

            torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
            tokenizer.save_pretrained(str(output_dir))

            label_map_src = Path(config.get("label_map", "training/data_prep/raw/label_map_v5.json"))
            label_map_dst = output_dir / "label_map_v5.json"
            if label_map_src.exists():
                shutil.copy(label_map_src, label_map_dst)

            # Save config so backend knows which model to load
            with open(output_dir / "classifier_config.json", "w") as f:
                json.dump({
                    "base_model":  config["base_model"],
                    "model_class": "InLegalBERTClassifier",
                    "num_labels":  NUM_LABELS,
                    "label_list":  LABEL_LIST,
                    "threshold":   config["threshold"],
                    "max_len":     config["max_len"],
                    "label_map_file": "label_map_v5.json",
                }, f, indent=2)

            logger.info(f"  💾 Best model saved! Micro F1 = {best_f1:.4f}")

    # ── Save training history ──────────────────────────────────────────────
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"✅  Training complete!")
    logger.info(f"   Best Micro F1: {best_f1:.4f}")
    logger.info(f"   Model saved:   {output_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. python run.py")
    logger.info(f"  2. Open http://localhost:3000")
    logger.info(f"  3. Test predictions!")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Fine-tune InLegalBERT for IPC/CrPC classification"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config JSON file")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config.update(json.load(f))
        logger.info(f"Config loaded from {args.config}")
    else:
        logger.info("Using default config")

    logger.info("=" * 60)
    logger.info("⚖️  InLegalBERT Fine-tuning for IPC/CrPC Classification")
    logger.info("=" * 60)
    logger.info(f"Model:        {config['base_model']}")
    logger.info(f"Train file:   {config['train_file']}")
    logger.info(f"Epochs:       {config['epochs']}")
    logger.info(f"Batch size:   {config['batch_size']}")
    logger.info(f"Max length:   {config['max_len']}")
    logger.info(f"Threshold:    {config['threshold']}")

    train(config)


if __name__ == "__main__":
    main()