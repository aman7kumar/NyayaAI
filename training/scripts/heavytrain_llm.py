"""
training/scripts/train_llm.py
================================
Fine-tunes a causal language model for legal question answering.

Supports two modes:
  1. GPT-2 (small, ~500MB) — good for low-resource machines / CPU
  2. LLaMA-2-7B with 4-bit quantization + LoRA — best quality, needs ~8GB VRAM

Training format (instruction fine-tuning):
  <|user|> A citizen reports: [FIR TEXT]. What laws apply?
  <|assistant|> [LEGAL EXPLANATION WITH SECTIONS]

Usage:
  # Fine-tune GPT-2 (CPU/low VRAM):
  python training/scripts/train_llm.py --config training/configs/llm_config_gpt2.json

  # Fine-tune LLaMA-2-7B with LoRA (needs 8GB VRAM):
  python training/scripts/train_llm.py --config training/configs/llm_config_llama.json

Output:
  backend/models/saved/legal_llm/  ← saved tokenizer + model weights
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Force CPU — prevents GPU OOM
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── Default Configs ────────────────────────────────────────────────────────────

GPT2_CONFIG = {
    "base_model":      "gpt2",
    "train_file":      "backend/data/llm_finetune.jsonl",
    "output_dir":      "backend/models/saved/legal_llm",
    "max_length":      512,
    "batch_size":      4,
    "grad_accum":      4,       # effective batch = 4*4 = 16
    "epochs":          3,
    "learning_rate":   5e-5,
    "warmup_ratio":    0.05,
    "use_lora":        False,
    "save_steps":      200,
    "logging_steps":   50,
    "seed":            42,
}

LLAMA_CONFIG = {
    "base_model":      "meta-llama/Llama-2-7b-hf",   # requires HF access token
    "train_file":      "backend/data/llm_finetune.jsonl",
    "output_dir":      "backend/models/saved/legal_llm",
    "max_length":      1024,
    "batch_size":      2,
    "grad_accum":      8,
    "epochs":          2,
    "learning_rate":   2e-4,
    "warmup_ratio":    0.05,
    "use_lora":        True,
    "lora_r":          16,
    "lora_alpha":      32,
    "lora_dropout":    0.05,
    "lora_target":     ["q_proj", "v_proj"],
    "load_in_4bit":    True,
    "save_steps":      100,
    "logging_steps":   25,
    "seed":            42,
}


# ── Dataset ────────────────────────────────────────────────────────────────────

class LegalQADataset(Dataset):
    """Dataset for causal LM fine-tuning on legal QA pairs."""

    PROMPT_TEMPLATE = (
        "<|user|>\n{prompt}\n<|assistant|>\n{completion}<|endoftext|>"
    )

    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.encodings  = []

        for sample in tqdm(samples, desc="Tokenizing"):
            text = self.PROMPT_TEMPLATE.format(
                prompt=sample["prompt"],
                completion=sample["completion"],
            )
            enc = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            self.encodings.append({
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict:
        item = self.encodings[idx]
        # For causal LM: labels = input_ids (shift done internally by model)
        return {
            "input_ids":      item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels":         item["input_ids"].clone(),
        }


# ── LoRA Setup ────────────────────────────────────────────────────────────────

def apply_lora(model, config: dict):
    """Apply LoRA adapters for parameter-efficient fine-tuning."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        logger.error("peft not installed. Run: pip install peft")
        sys.exit(1)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("lora_target", ["q_proj", "v_proj"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── Main Trainer ───────────────────────────────────────────────────────────────

class LLMTrainer:
    def __init__(self, config: dict):
        self.cfg    = config
        self.device = torch.device("cpu")
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        torch.manual_seed(config["seed"])

    def load_samples(self) -> list[dict]:
        samples = []
        train_file = Path(self.cfg["train_file"])
        if not train_file.exists():
            logger.error(f"Training file not found: {train_file}")
            sys.exit(1)
        with open(train_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        logger.info(f"Loaded {len(samples)} training samples")
        return samples

    def load_model_and_tokenizer(self):
        config     = self.cfg
        model_name = config["base_model"]

        # 4-bit quantization for LLaMA
        if config.get("load_in_4bit") and self.device.type == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
                logger.info("Loaded model in 4-bit quantization")
            except Exception as e:
                logger.warning(f"4-bit quantization failed: {e}. Loading normally.")
                model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        # Apply LoRA if configured
        if config.get("use_lora"):
            model = apply_lora(model, config)

        if self.device.type == "cpu":
            model = model.to(self.device)

        return model, tokenizer

    def train(self):
        config = self.cfg
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        samples = self.load_samples()
        model, tokenizer = self.load_model_and_tokenizer()

        # Use HuggingFace Trainer for simplicity
        train_dataset = LegalQADataset(samples, tokenizer, config["max_length"])
        """
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["grad_accum"],
            learning_rate=config["learning_rate"],
            warmup_ratio=config["warmup_ratio"],
            weight_decay=0.01,
            fp16=(self.device.type == "cuda"),
            logging_steps=config["logging_steps"],
            save_steps=config["save_steps"],
            save_total_limit=2,
            load_best_model_at_end=False,
            report_to="none",   # disable wandb/mlflow
            seed=config["seed"],
            dataloader_num_workers=2,
        )
        """
        training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=config["learning_rate"],
        warmup_steps=10,
        weight_decay=0.01,
        fp16=False,
        logging_steps=20,
        save_steps=200,
        save_total_limit=1,
        load_best_model_at_end=False,
        report_to="none",
        seed=config["seed"],
        dataloader_num_workers=0,
        use_cpu=True,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        logger.info("Starting LLM fine-tuning …")
        trainer.train()

        # Save final model
        if config.get("use_lora"):
            # Merge LoRA weights into base model before saving
            try:
                from peft import PeftModel
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained(str(output_dir))
                logger.info("LoRA weights merged and saved.")
            except Exception:
                model.save_pretrained(str(output_dir))
        else:
            model.save_pretrained(str(output_dir))

        tokenizer.save_pretrained(str(output_dir))
        logger.info(f"✅  LLM fine-tuning complete! Model saved → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for Legal QA")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--model",  type=str, choices=["gpt2", "llama"], default="gpt2")
    args = parser.parse_args()

    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)
    elif args.model == "llama":
        config = LLAMA_CONFIG.copy()
    else:
        config = GPT2_CONFIG.copy()

    logger.info("=" * 60)
    logger.info("⚖️  Legal AI — LLM Fine-Tuning")
    logger.info("=" * 60)
    logger.info(f"Base model: {config['base_model']}")
    logger.info(f"LoRA:       {config.get('use_lora', False)}")
    logger.info(f"Epochs:     {config['epochs']}")

    trainer = LLMTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
