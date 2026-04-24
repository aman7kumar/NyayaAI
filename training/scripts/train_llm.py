"""
training/scripts/train_llm.py
Lightweight GPT-2 fine-tuning — CPU friendly, low memory
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = "<|user|>\n{prompt}\n<|assistant|>\n{completion}<|endoftext|>"


class LegalQADataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=256):
        self.encodings = []
        tokenizer.pad_token = tokenizer.eos_token

        for sample in tqdm(samples, desc="Tokenizing"):
            text = PROMPT_TEMPLATE.format(
                prompt=sample["prompt"][:200],
                completion=sample["completion"][:200],
            )
            enc = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            ids = enc["input_ids"].squeeze(0)
            self.encodings.append({
                "input_ids":      ids,
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         ids.clone(),
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default="gpt2")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    # Config
    config = {
        "base_model":    "gpt2",
        "train_file":    "backend/data/llm_finetune.jsonl",
        "output_dir":    "backend/models/saved/legal_llm",
        "max_length":    256,
        "batch_size":    2,
        "epochs":        2,
        "learning_rate": 5e-5,
        "seed":          42,
    }
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config.update(json.load(f))

    torch.manual_seed(config["seed"])
    device = torch.device("cpu")
    logger.info("=" * 60)
    logger.info("⚖️  Legal AI — LLM Fine-Tuning (lightweight)")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")

    # Load samples
    train_file = Path(config["train_file"])
    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        sys.exit(1)

    samples = []
    with open(train_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # Limit to 300 samples max to keep training fast
    samples = samples[:300]
    logger.info(f"Training on {len(samples)} samples (capped at 300 for speed)")

    # Load model and tokenizer
    logger.info("Loading GPT-2 ...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        dtype=torch.float32,
    )
    model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Dataset and dataloader
    dataset    = LegalQADataset(samples, tokenizer, config["max_length"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info("Starting fine-tuning ...")
    for epoch in range(1, config["epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{config['epochs']}")
        total_loss  = 0.0
        total_steps = 0

        for batch in tqdm(dataloader, desc=f"  Epoch {epoch}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss  += loss.item()
            total_steps += 1

        avg_loss = total_loss / total_steps
        logger.info(f"  Average Loss: {avg_loss:.4f}")

    # Save model
    logger.info(f"\nSaving model to {output_dir} ...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info(f"✅  LLM fine-tuning complete! Model saved → {output_dir}")
    logger.info("   Now run: uvicorn backend.api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()