# 🧠 Training Guide — AI-Powered Legal Analyzer

## Step-by-Step Training Instructions

---

## Prerequisites

Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-hin

# macOS
brew install tesseract && brew install tesseract-lang
```

Python setup:
```bash
cd legal_ai_india
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
```

---

## Training Pipeline (run in order)

### Step 1: Generate/prepare data
```bash
python training/data_prep/generate_synthetic_data.py
# Adds real PDFs to training/data_prep/raw/ (see DATASET_GUIDE.md)
python training/scripts/prepare_dataset.py
```

### Step 2: Build FAISS vector index
```bash
python training/scripts/build_vector_db.py
# Output: backend/models/saved/faiss_index/
```

### Step 3: Train IPC classifier (BERT)
```bash
python training/scripts/train_classifier.py --config training/configs/classifier_config.json
# Output: backend/models/saved/ipc_classifier/
# Time: 30-60min GPU / 2-6hr CPU
```

### Step 4: Fine-tune LLM
```bash
# Option A — GPT-2 (CPU friendly, ~500MB):
python training/scripts/train_llm.py --config training/configs/llm_config_gpt2.json

# Option B — LLaMA-2-7B with LoRA (needs 8GB VRAM, best quality):
huggingface-cli login
python training/scripts/train_llm.py --config training/configs/llm_config_llama.json
# Output: backend/models/saved/legal_llm/
```

### Step 5: Start services
```bash
# Backend
cd backend && uvicorn api.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend && npm install && npm start
```

---

## Expected F1 Scores (Classifier)

| Training Samples | Expected F1 |
|------------------|-------------|
| 200 (synthetic)  | 0.45–0.60   |
| 1,000            | 0.65–0.75   |
| 5,000+           | 0.80–0.90   |

---

## GPU Memory Requirements

| Model | Min VRAM |
|-------|----------|
| BERT classifier | 2GB |
| GPT-2 fine-tune | 2GB |
| LLaMA-2-7B (4-bit LoRA) | 6GB |
| Mistral-7B (4-bit LoRA) | 6GB |

---

## Troubleshooting

**"CUDA out of memory"** → Reduce batch_size in config, increase grad_accum

**"No checkpoint found"** → Models not trained yet, run Steps 1-4

**"FAISS index not found"** → Run build_vector_db.py

**Poor F1 (< 0.5)** → Add more training data, ensure CSV format is correct

**OCR poor quality** → Install tesseract-ocr-hin, ensure image is clear and high-resolution
