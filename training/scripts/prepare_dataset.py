"""
training/scripts/prepare_dataset.py
======================================
Dataset preparation pipeline.

Steps:
1. Load raw legal PDFs (IPC, CrPC, Constitution, FIR samples)
2. Extract text using PDFExtractor
3. Create labeled FIR-to-IPC training data (for classifier)
4. Create QA pairs (for LLM fine-tuning)
5. Export as JSONL files

Usage:
  python training/scripts/prepare_dataset.py \
    --input_dir training/data_prep/raw \
    --output_dir backend/data

Dataset folder structure (raw/):
  raw/
    ipc.pdf
    crpc.pdf
    constitution.pdf
    fir_labeled.csv        ← FIR text + IPC section labels
    judgments/             ← Judgment PDFs
    contracts/             ← Contract PDFs
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.modules.pdf_extractor import PDFExtractor
from backend.models.ipc_classifier import LABEL2IDX, LABEL_LIST

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def prepare_rag_corpus(input_dir: Path, output_dir: Path) -> None:
    """
    Extract text from all legal PDFs → JSONL for RAG vector DB.
    Each line: {"text": "...", "section": "IPC 323", "act": "IPC", "source": "ipc.pdf"}
    """
    extractor = PDFExtractor()
    all_chunks = []

    pdf_files = list(input_dir.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files.")

    for pdf_file in tqdm(pdf_files, desc="Extracting PDFs"):
        chunks = extractor.extract_from_file(pdf_file)
        all_chunks.extend(chunks)

    output_file = output_dir / "rag_corpus.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"✅  RAG corpus: {len(all_chunks)} chunks → {output_file}")


def prepare_classifier_dataset(input_dir: Path, output_dir: Path) -> None:
    """
    Prepare multi-label classification dataset from FIR CSV.

    Expected CSV columns:
      fir_text      : Raw FIR / complaint text
      ipc_sections  : Comma-separated IPC sections e.g. "IPC 323, IPC 324"

    Output JSONL:
      {"text": "...", "labels": [0, 0, 1, 0, ...]}  ← binary multi-hot vector
    """
    csv_file = input_dir / "fir_labeled.csv"
    if not csv_file.exists():
        logger.warning(f"⚠️  {csv_file} not found. Skipping classifier dataset prep.")
        logger.info("Create fir_labeled.csv with columns: fir_text, ipc_sections")
        _create_sample_classifier_csv(csv_file)
        return

    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} labeled FIR samples.")

    output_file = output_dir / "classifier_train.jsonl"
    skipped = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifier dataset"):
            text = str(row["fir_text"]).strip()
            sections_str = str(row.get("ipc_sections", "")).strip()

            if not text or not sections_str:
                skipped += 1
                continue

            # Parse section labels → multi-hot vector
            labels = [0] * len(LABEL_LIST)
            for sec in sections_str.split(","):
                sec = sec.strip().replace(" ", "_").upper()
                sec = sec.replace("IPC_", "IPC_").replace("CRPC_", "CrPC_")
                if sec in LABEL2IDX:
                    labels[LABEL2IDX[sec]] = 1

            if sum(labels) == 0:
                skipped += 1
                continue

            f.write(json.dumps({"text": text, "labels": labels}, ensure_ascii=False) + "\n")

    logger.info(f"✅  Classifier dataset: {len(df) - skipped} samples → {output_file}")
    if skipped:
        logger.warning(f"   Skipped {skipped} rows (missing text or unrecognized sections)")


def prepare_llm_finetune_dataset(input_dir: Path, output_dir: Path) -> None:
    """
    Prepare instruction-following dataset for LLM fine-tuning.
    Format: {"prompt": "...", "completion": "..."}

    Sources:
      - judgment_qa.csv (question, answer pairs from judgments)
      - fir_labeled.csv (FIR → explanation pairs)
    """
    output_file = output_dir / "llm_finetune.jsonl"
    samples = []

    # From FIR labeled data
    csv_file = input_dir / "fir_labeled.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            text = str(row.get("fir_text", "")).strip()
            sections = str(row.get("ipc_sections", "")).strip()
            explanation = str(row.get("explanation", "")).strip()
            if text and sections:
                prompt = (
                    f"A citizen reports: {text}\n\n"
                    f"What IPC/CrPC sections apply and what should they do?"
                )
                completion = (
                    explanation if explanation else
                    f"Based on the facts described, the applicable sections are: {sections}. "
                    f"The citizen should immediately file an FIR at the local police station "
                    f"and consult a lawyer for further guidance."
                )
                samples.append({"prompt": prompt, "completion": completion})

    # Add from judgment_qa.csv if available
    qa_file = input_dir / "judgment_qa.csv"
    if qa_file.exists():
        df_qa = pd.read_csv(qa_file)
        for _, row in df_qa.iterrows():
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            if q and a:
                samples.append({"prompt": q, "completion": a})

    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"✅  LLM fine-tune dataset: {len(samples)} samples → {output_file}")


def _create_sample_classifier_csv(output_path: Path):
    """Create a small sample CSV to show the expected format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_data = [
        {
            "fir_text": "My neighbor hit me with a stick causing injuries on my arm.",
            "ipc_sections": "IPC_323, IPC_324",
            "explanation": "IPC 323 covers voluntarily causing hurt. IPC 324 applies when a dangerous weapon is used.",
        },
        {
            "fir_text": "Someone stole my mobile phone from my bag at the bus stand.",
            "ipc_sections": "IPC_379",
            "explanation": "IPC 379 covers theft of movable property.",
        },
        {
            "fir_text": "My husband demands dowry and beats me regularly.",
            "ipc_sections": "IPC_498A, IPC_323",
            "explanation": "IPC 498A covers cruelty by husband or relatives. IPC 323 for physical hurt.",
        },
        {
            "fir_text": "A person threatened to kill me if I don't pay him money.",
            "ipc_sections": "IPC_503, IPC_506",
            "explanation": "IPC 503 and 506 cover criminal intimidation and its punishment.",
        },
        {
            "fir_text": "I was cheated of Rs 50000 by a shopkeeper who sold me a fake product.",
            "ipc_sections": "IPC_420",
            "explanation": "IPC 420 covers cheating and dishonestly inducing delivery of property.",
        },
    ]
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    logger.info(f"✅  Sample fir_labeled.csv created at {output_path}. Add your real data here.")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for Legal AI system")
    parser.add_argument("--input_dir",  type=str, default="training/data_prep/raw")
    parser.add_argument("--output_dir", type=str, default="backend/data")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("⚖️  Legal AI — Dataset Preparation Pipeline")
    logger.info("=" * 60)

    logger.info("\n[1/3] Preparing RAG corpus from PDFs …")
    prepare_rag_corpus(input_dir, output_dir)

    logger.info("\n[2/3] Preparing IPC classifier dataset …")
    prepare_classifier_dataset(input_dir, output_dir)

    logger.info("\n[3/3] Preparing LLM fine-tune dataset …")
    prepare_llm_finetune_dataset(input_dir, output_dir)

    logger.info(f"\n✅  All datasets ready in: {output_dir}")


if __name__ == "__main__":
    main()
