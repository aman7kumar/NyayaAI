# ⚖️ AI-Powered Legal Document Analyzer — India

> A complete, self-trained, open-source AI system for Indian legal assistance.  
> No external AI APIs used. **Everything is trained and served locally.**



## 📁 Project Structure

```
legal_ai_india/
├── backend/
│   ├── api/                  # FastAPI REST endpoints
│   │   ├── main.py           # App entrypoint
│   │   ├── routes/           # Route handlers
│   ├── models/               # Custom trained model definitions
│   │   ├── ipc_classifier.py # IPC/CrPC section classifier (fine-tuned BERT)
│   │   ├── rag_engine.py     # RAG pipeline (FAISS + LLM)
│   │   ├── ocr_module.py     # Tesseract OCR for handwritten FIRs
│   │   └── roadmap_engine.py # Legal roadmap generator
│   ├── modules/              # Business logic
│   │   ├── query_classifier.py
│   │   ├── entity_extractor.py
│   │   ├── pdf_extractor.py
│   │   ├── explainability.py
│   │   └── multilingual.py
│   ├── data/                 # Processed datasets (JSONL)
│   └── utils/                # Helpers
├── training/
│   ├── scripts/              # Training scripts
│   │   ├── train_classifier.py     # Fine-tune BERT for IPC classification
│   │   ├── train_llm.py            # Fine-tune GPT2/LLaMA for legal QA
│   │   ├── build_vector_db.py      # Build FAISS index from legal corpus
│   │   └── prepare_dataset.py      # Dataset preprocessing pipeline
│   ├── configs/              # Training hyperparameters
│   └── data_prep/            # Raw data → JSONL converters
├── frontend/
│   ├── src/
│   │   ├── components/       # Reusable UI components
│   │   ├── pages/            # Main application pages
│   │   └── styles/           # CSS modules
│   └── public/
└── docs/
    ├── DATASET_GUIDE.md      # How to collect/prepare datasets
    └── TRAINING_GUIDE.md     # Step-by-step training instructions
```



## 🚀 Quick Start

### 1. Prerequisites
```bash
Python >= 3.10
Node.js >= 18
CUDA (optional, for GPU training)
Tesseract OCR installed
```

### 2. Install Backend
```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Prepare & Train Models
```bash
# Step 1: Prepare dataset
cd training/scripts
python prepare_dataset.py --input_dir ../data_prep/raw --output_dir ../../backend/data

# Step 2: Build FAISS vector index
python build_vector_db.py --data_dir ../../backend/data --output_dir ../../backend/models/faiss_index

# Step 3: Train IPC classifier
python train_classifier.py --config ../configs/classifier_config.json

# Step 4: Fine-tune LLM
python train_llm.py --config ../configs/llm_config.json
```

### 4. Start Backend
```bash
cd backend
uvicorn api.main:app --reload --port 8000
```

### 5. Start Frontend
```bash
cd frontend
npm install
npm start
```



## 🗄️ Datasets to Collect (Free & Open)

| Dataset | Source | Use |
|---------|--------|-----|
| IPC Full Text | indiacode.nic.in | Legal corpus for RAG |
| CrPC Full Text | indiacode.nic.in | Legal corpus for RAG |
| Indian Constitution | legislative.gov.in | Legal corpus |
| FIR-IPC Labeled Dataset | Kaggle / GitHub | Classifier training |
| Indian Court Judgments | indiankanoon.org | RAG + fine-tuning |
| IndicNLP Corpus | ai4bharat.org | Multilingual support |

See `docs/DATASET_GUIDE.md` for detailed download instructions.



## 🧠 Model Architecture

- **IPC Classifier**: Fine-tuned `bert-base-multilingual-cased` → Multi-label classification
- **RAG Engine**: FAISS vector DB + Fine-tuned GPT-2 / LLaMA-2-7B (quantized)
- **OCR**: Tesseract 5.x with Hindi language pack
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **NER**: Custom spaCy model for legal entities



## ⚖️ Disclaimer

This system provides **preliminary legal information only** and is **not a substitute for qualified legal counsel**. Always consult a licensed lawyer for legal advice.

old code of train_classifier.py
"""
training/scripts/train_classifier.py

Fine-tunes a multilingual BERT model for multi-label IPC/CrPC section classification.

Model:  bert-base-multilingual-cased
Task:   Multi-label classification (sigmoid + BCEWithLogitsLoss)
Data:   backend/data/classifier_train.jsonl

Training strategy:
  - Freeze BERT layers 1-6 in first epoch (warm-up)
  - Unfreeze all layers from epoch 2 (full fine-tuning)
  - Use AdamW + linear warmup scheduler
  - Save best checkpoint based on F1 score

Usage:
  python training/scripts/train_classifier.py \
    --config training/configs/classifier_config.json

GPU highly recommended. ~2-4 hours on RTX 3080 for 10k samples / 5 epochs.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Force CPU — BERT needs ~6GB VRAM, use CPU instead
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.models.ipc_classifier import (
    BertIPCClassifier,
    NUM_LABELS,
    LABEL_LIST,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── Default config ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "train_file":    "backend/data/classifier_train.jsonl",
    "val_split":     0.15,
    "output_dir":    "backend/models/saved/ipc_classifier",
    "model_name":    "bert-base-multilingual-cased",
    "max_len":       128,
    "batch_size":    8,
    "epochs":        5,
    "learning_rate": 2e-5,
    "warmup_ratio":  0.1,
    "weight_decay":  0.01,
    "threshold":     0.35,
    "seed":          42,
}


# ── Dataset ────────────────────────────────────────────────────────────────────

class IPCDataset(Dataset):
    """PyTorch dataset for IPC multi-label classification."""

    def __init__(
        self,
        samples: list[dict],
        tokenizer: BertTokenizer,
        max_len: int = 256,
    ):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item  = self.samples[idx]
        text  = item["text"]
        labels = torch.tensor(item["labels"], dtype=torch.float32)

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         labels,
        }


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_f1(preds: np.ndarray, targets: np.ndarray, threshold: float = 0.35) -> dict:
    """Compute micro/macro F1 for multi-label classification."""
    binary_preds = (preds >= threshold).astype(int)

    tp = (binary_preds & targets).sum()
    fp = (binary_preds & ~targets.astype(bool)).sum()
    fn = (~binary_preds.astype(bool) & targets.astype(bool)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
    }


# ── Training Loop ──────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device("cpu")
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on: {self.device}")

        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])

    def load_data(self) -> tuple[list, list]:
        """Load JSONL and split into train/val."""
        samples = []
        train_file = Path(self.cfg["train_file"])

        if not train_file.exists():
            logger.error(f"Training file not found: {train_file}")
            logger.error("Run prepare_dataset.py first.")
            sys.exit(1)

        with open(train_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        logger.info(f"Total samples: {len(samples)}")

        # Shuffle and split
        np.random.shuffle(samples)
        split = int(len(samples) * (1 - self.cfg["val_split"]))
        return samples[:split], samples[split:]

    def train(self):
        """Full training loop."""
        config = self.cfg

        # Load data
        train_samples, val_samples = self.load_data()
        logger.info(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

        # Tokenizer + datasets
        tokenizer = BertTokenizer.from_pretrained(config["model_name"])
        train_ds  = IPCDataset(train_samples, tokenizer, config["max_len"])
        val_ds    = IPCDataset(val_samples,   tokenizer, config["max_len"])

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False, num_workers=2)

        # Model
        model = BertIPCClassifier(NUM_LABELS).to(self.device)

        # Free any cached memory before training starts
       # torch.cuda.empty_cache()

        # Enable gradient checkpointing to save GPU memory
       # model.bert.gradient_checkpointing_enable()

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": config["weight_decay"]},
            {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(params, lr=config["learning_rate"])

        # Scheduler
        total_steps   = len(train_loader) * config["epochs"]
        warmup_steps  = int(total_steps * config["warmup_ratio"])
        scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        # Loss
        criterion = nn.BCEWithLogitsLoss()

        # Training
        best_f1    = 0.0
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        history = []

        for epoch in range(1, config["epochs"] + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"EPOCH {epoch}/{config['epochs']}")

            # ── Freeze/unfreeze strategy ──────────────────────────────
            if epoch == 1:
                # Freeze lower BERT layers for stable warm-up
                for name, param in model.bert.named_parameters():
                    if "encoder.layer" in name:
                        layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                        if layer_num < 6:
                            param.requires_grad = False
                logger.info("Froze BERT layers 0-5 for epoch 1 warm-up")
            elif epoch == 2:
                # Unfreeze all layers
                for param in model.bert.parameters():
                    param.requires_grad = True
                logger.info("Unfroze all BERT layers from epoch 2")

            # ── Train phase ───────────────────────────────────────────
            model.train()
            train_loss = 0.0
            train_steps = 0

            for batch in tqdm(train_loader, desc=f"  Training"):
                optimizer.zero_grad()

                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss  += loss.item()
                train_steps += 1

            avg_train_loss = train_loss / train_steps

            # ── Validation phase ───────────────────────────────────────
            model.eval()
            all_preds, all_targets = [], []
            val_loss = 0.0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"  Validating"):
                    input_ids      = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels         = batch["labels"].to(self.device)

                    logits = model(input_ids, attention_mask)
                    loss   = criterion(logits, labels)
                    val_loss += loss.item()

                    probs   = torch.sigmoid(logits).cpu().numpy()
                    targets = labels.cpu().numpy()
                    all_preds.append(probs)
                    all_targets.append(targets)

            all_preds   = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)
            metrics     = compute_f1(all_preds, all_targets, config["threshold"])
            avg_val_loss = val_loss / len(val_loader)

            logger.info(
                f"  Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"P: {metrics['precision']:.4f} | "
                f"R: {metrics['recall']:.4f}"
            )

            history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                **metrics,
            })

            # Save best model
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
                tokenizer.save_pretrained(str(output_dir))
                logger.info(f"  💾 New best model saved! F1={best_f1:.4f}")

        # Save training history
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"\n✅  Training complete! Best F1: {best_f1:.4f}")
        logger.info(f"   Model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train IPC/CrPC section classifier")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                config.update(json.load(f))
            logger.info(f"Loaded config from {config_path}")

    logger.info("=" * 60)
    logger.info("⚖️  Legal AI — IPC Classifier Training")
    logger.info("=" * 60)
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

