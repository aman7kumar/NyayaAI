"""
Fixed InLegalBERT fine-tuner for 163-class multi-label IPC classification.

Paths default to the NyayaAI repo layout (run from project root: NyayaAI/).

Place your JSONL files and label_map_v5.json under training/data_prep/raw/, or pass
--train_file, --val_file, --label_map_file explicitly (e.g. from Jupyter).

On each improvement, saves: encoder + tokenizer + classifier_head.pt +
classifier_meta.json + label_map_v5.json + classifier_config.json (for the API loader).
"""

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s\t%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Project root = parent of training/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _abs(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


@dataclass
class Config:
    model_name: str = "law-ai/InLegalBERT"
    train_file: str = "training/data_prep/raw/legal_dataset_train.jsonl"
    val_file: str = "training/data_prep/raw/legal_dataset_val.jsonl"
    label_map_file: str = "training/data_prep/raw/label_map_v5.json"
    output_dir: str = "backend/models/saved/ipc_classifier"

    max_length: int = 128
    batch_size: int = 32
    epochs: int = 10
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    dropout: float = 0.3

    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    label_smoothing: float = 0.05

    default_threshold: float = 0.3
    tune_threshold: bool = True

    patience: int = 3
    monitor: str = "macro_f1"

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser()
    for field in fields(cfg):
        f, v = field.name, getattr(cfg, field.name)
        parser.add_argument(f"--{f}", type=type(v), default=v)
    args = parser.parse_args()
    for field in fields(cfg):
        f = field.name
        setattr(cfg, f, getattr(args, f))
    # Resolve paths relative to project root
    cfg.train_file = _abs(cfg.train_file)
    cfg.val_file = _abs(cfg.val_file)
    cfg.label_map_file = _abs(cfg.label_map_file)
    cfg.output_dir = _abs(cfg.output_dir)
    return cfg


class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + self.smoothing * 0.5

        p = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.where(targets >= 0.5, p, 1 - p)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = torch.where(
            targets >= 0.5,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1 - self.alpha),
        )
        return (alpha_t * focal_weight * bce).mean()


class LegalDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int, n_labels: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_labels = n_labels
        self.items = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.items.append({
                    "text": d["text"].strip(),
                    "labels": torch.tensor(d["labels"], dtype=torch.float32),
                })

        log.info(f"Loaded {len(self.items)} samples from {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        enc = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": item["labels"],
        }


class LegalClassifier(nn.Module):
    def __init__(self, model_name: str, n_labels: int, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden // 2, n_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


def compute_metrics(labels_true, labels_pred_prob, thresholds=None, n_labels=163):
    if thresholds is None:
        thresholds = [0.3] * n_labels

    pred = np.zeros_like(labels_pred_prob)
    for i, thr in enumerate(thresholds):
        pred[:, i] = (labels_pred_prob[:, i] >= thr).astype(int)

    return {
        "micro_f1": f1_score(labels_true, pred, average="micro", zero_division=0),
        "macro_f1": f1_score(labels_true, pred, average="macro", zero_division=0),
        "micro_precision": precision_score(labels_true, pred, average="micro", zero_division=0),
        "micro_recall": recall_score(labels_true, pred, average="micro", zero_division=0),
    }


def tune_thresholds(labels_true: np.ndarray, probs: np.ndarray, n_labels: int):
    thresholds = []
    candidates = np.arange(0.1, 0.9, 0.05)
    for i in range(n_labels):
        best_f1, best_thr = 0.0, 0.3
        for thr in candidates:
            pred = (probs[:, i] >= thr).astype(int)
            f1 = f1_score(labels_true[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        thresholds.append(float(best_thr))
    return thresholds


def evaluate(model, loader, device, thresholds=None, n_labels=163):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    metrics = compute_metrics(all_labels, all_probs, thresholds, n_labels)
    return metrics, all_probs, all_labels


def _write_classifier_config(
    out_dir: Path,
    cfg: Config,
    n_labels: int,
    label_names: List[str],
    best_thresholds: List[float],
    best_score: float,
    best_epoch: int,
) -> None:
    with open(out_dir / "classifier_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "base_model": cfg.model_name,
            "model_class": "InLegalBERTClassifierV2",
            "num_labels": n_labels,
            "label_list": label_names,
            "threshold": cfg.default_threshold,
            "max_len": cfg.max_length,
            "label_map_file": "label_map_v5.json",
            "classifier_head_file": "classifier_head.pt",
            "meta_file": "classifier_meta.json",
            "monitor": cfg.monitor,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "thresholds": best_thresholds,
        }, f, indent=2)


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    with open(cfg.label_map_file, encoding="utf-8") as f:
        lmap = json.load(f)
    n_labels = lmap["n_labels"]
    label_names = lmap["labels"]
    log.info(f"Label map: {n_labels} labels")

    log.info(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    log.info(f"Loading model: {cfg.model_name}")
    model = LegalClassifier(cfg.model_name, n_labels, cfg.dropout)
    model.to(cfg.device)
    log.info(f"Device: {cfg.device}")

    train_ds = LegalDataset(cfg.train_file, tokenizer, cfg.max_length, n_labels)
    val_ds = LegalDataset(cfg.val_file, tokenizer, cfg.max_length, n_labels)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    criterion = FocalBCELoss(
        gamma=cfg.focal_gamma, alpha=cfg.focal_alpha, smoothing=cfg.label_smoothing
    )

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": model.classifier.parameters(), "lr": cfg.lr * 5, "weight_decay": cfg.weight_decay},
    ])

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=cfg.fp16 and cfg.device == "cuda")

    best_score = 0.0
    patience_counter = 0
    best_thresholds = [cfg.default_threshold] * n_labels
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"Training | focal γ={cfg.focal_gamma}, α={cfg.focal_alpha} | smoothing={cfg.label_smoothing}")
    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    log.info("=" * 60)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)

            optimizer.zero_grad()
            with autocast(enabled=cfg.fp16 and cfg.device == "cuda"):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / steps

        metrics, val_probs, val_labels = evaluate(
            model, val_loader, cfg.device, best_thresholds, n_labels
        )
        score = metrics[cfg.monitor]

        log.info(
            f"Epoch {epoch}/{cfg.epochs} | Loss: {avg_loss:.4f} | "
            f"Micro-F1: {metrics['micro_f1']:.4f} | Macro-F1: {metrics['macro_f1']:.4f} | "
            f"P: {metrics['micro_precision']:.4f} | R: {metrics['micro_recall']:.4f}"
        )

        if score > best_score:
            best_score = score
            patience_counter = 0

            if cfg.tune_threshold:
                log.info("Tuning per-label thresholds on validation set…")
                best_thresholds = tune_thresholds(val_labels, val_probs, n_labels)

            model.encoder.save_pretrained(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
            torch.save(model.classifier.state_dict(), out_dir / "classifier_head.pt")

            shutil.copy2(cfg.label_map_file, out_dir / "label_map_v5.json")

            meta = {
                "n_labels": n_labels,
                "labels": label_names,
                "thresholds": best_thresholds,
                "best_epoch": epoch,
                "best_score": best_score,
                "monitor": cfg.monitor,
                "dropout": cfg.dropout,
            }
            with open(out_dir / "classifier_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            _write_classifier_config(
                out_dir, cfg, n_labels, label_names, best_thresholds, best_score, epoch
            )

            log.info(f"Best {cfg.monitor}: {best_score:.4f} — saved to {out_dir}")
        else:
            patience_counter += 1
            log.info(f"No improvement. Patience: {patience_counter}/{cfg.patience}")
            if patience_counter >= cfg.patience:
                log.info("Early stopping triggered.")
                break

    log.info("=" * 60)
    log.info(f"Done. Best {cfg.monitor}: {best_score:.4f} | {out_dir}")

    log.info("\nPer-label F1 on val (worst 20):")
    _, val_probs, val_labels_np = evaluate(model, val_loader, cfg.device, best_thresholds, n_labels)
    per_label_f1 = []
    for i in range(n_labels):
        pred = (val_probs[:, i] >= best_thresholds[i]).astype(int)
        per_label_f1.append((label_names[i], f1_score(val_labels_np[:, i], pred, zero_division=0)))
    per_label_f1.sort(key=lambda x: x[1])
    for name, f1v in per_label_f1[:20]:
        log.info(f"  {name:<15}: {f1v:.3f}")


if __name__ == "__main__":
    train(parse_args())
