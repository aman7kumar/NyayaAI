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

