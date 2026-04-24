import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# ── FORCE CPU — do not remove these two lines ──
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.models.ipc_classifier import NUM_LABELS, LABEL_LIST

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "train_file":    "backend/data/classifier_train.jsonl",
    "val_split":     0.15,
    "output_dir":    "backend/models/saved/ipc_classifier",
    "max_len":       64,
    "batch_size":    16,
    "epochs":        5,
    "learning_rate": 3e-5,
    "warmup_ratio":  0.1,
    "weight_decay":  0.01,
    "threshold":     0.35,
    "seed":          42,
}


class DistilBertIPCClassifier(nn.Module):
    def __init__(self, num_labels, dropout=0.3):
        super().__init__()
        self.distilbert    = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
        hidden             = self.distilbert.config.hidden_size
        self.pre_classifier = nn.Linear(hidden, 256)
        self.dropout       = nn.Dropout(dropout)
        self.classifier    = nn.Linear(256, num_labels)
        self.relu          = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        out     = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = out.last_hidden_state[:, 0, :]
        x       = self.relu(self.pre_classifier(cls_out))
        x       = self.dropout(x)
        return self.classifier(x)


class IPCDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=64):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
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


def compute_f1(preds, targets, threshold=0.35):
    binary    = (preds >= threshold).astype(int)
    tp        = int((binary & targets.astype(bool)).sum())
    fp        = int((binary & ~targets.astype(bool)).sum())
    fn        = int((~binary.astype(bool) & targets.astype(bool)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args   = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config.update(json.load(f))

    # ── Always CPU ──
    device = torch.device("cpu")
    logger.info(f"Training on: {device}")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # ── Load data ──
    samples = []
    with open(config["train_file"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info(f"Total samples: {len(samples)}")
    np.random.shuffle(samples)
    split         = int(len(samples) * (1 - config["val_split"]))
    train_samples = samples[:split]
    val_samples   = samples[split:]
    logger.info(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    # ── Tokenizer & datasets ──
    tokenizer    = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    train_loader = DataLoader(IPCDataset(train_samples, tokenizer, config["max_len"]),
                              batch_size=config["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(IPCDataset(val_samples,   tokenizer, config["max_len"]),
                              batch_size=config["batch_size"], shuffle=False, num_workers=0)

    # ── Model ──
    model     = DistilBertIPCClassifier(NUM_LABELS).to(device)
    criterion = nn.BCEWithLogitsLoss()

    no_decay  = ["bias", "LayerNorm.weight"]
    params    = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": config["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer    = torch.optim.AdamW(params, lr=config["learning_rate"])
    total_steps  = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1    = 0.0
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    history    = []

    # ── Training loop ──
    for epoch in range(1, config["epochs"] + 1):
        logger.info(f"\n{'='*50}\nEPOCH {epoch}/{config['epochs']}")

        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"  Training epoch {epoch}"):
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss   = criterion(logits, batch["labels"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        all_preds, all_targets, val_loss = [], [], 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Validating epoch {epoch}"):
                logits   = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                val_loss += criterion(logits, batch["labels"].to(device)).item()
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(batch["labels"].numpy())

        metrics = compute_f1(np.vstack(all_preds), np.vstack(all_targets), config["threshold"])
        logger.info(
            f"  Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} | "
            f"F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}"
        )
        history.append({"epoch": epoch, **metrics})

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
            tokenizer.save_pretrained(str(output_dir))
            with open(output_dir / "classifier_config.json", "w") as f:
                json.dump({"base_model": "distilbert-base-multilingual-cased",
                           "num_labels": NUM_LABELS, "label_list": LABEL_LIST,
                           "threshold": config["threshold"], "max_len": config["max_len"]}, f, indent=2)
            logger.info(f"  💾 New best model saved! F1={best_f1:.4f}")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"\n✅  Training complete! Best F1: {best_f1:.4f}")
    logger.info(f"   Model saved to: {output_dir}")


if __name__ == "__main__":
    main()