# 📦 Dataset Collection Guide

## Overview

You need **3 types of datasets** to fully train the Legal AI system:

| Dataset | Used For | Size Needed |
|---------|----------|-------------|
| Legal PDF corpus (IPC, CrPC, Constitution) | RAG vector DB | All available text |
| FIR-to-IPC labeled CSV | BERT classifier | 500–10,000+ samples |
| Legal QA pairs | LLM fine-tuning | 1,000–50,000+ samples |

---

## 1. Legal PDF Corpus (for RAG)

These are **free** from Government of India websites.

Download:
- IPC 1860: https://indiacode.nic.in → search "Indian Penal Code"
- CrPC 1973: https://indiacode.nic.in → search "Code of Criminal Procedure"
- Constitution: https://legislative.gov.in
- Consumer Protection Act 2019: https://consumeraffairs.nic.in
- IT Act 2000: https://indiacode.nic.in
- Dowry Prohibition Act: https://indiacode.nic.in

Place all PDFs in: `training/data_prep/raw/`

---

## 2. FIR-to-IPC Labeled CSV

Create `training/data_prep/raw/fir_labeled.csv`:

```
fir_text,ipc_sections,explanation
"Someone attacked me with a stick","IPC_323,IPC_324","323 for hurt 324 for weapon"
"My husband beats me and demands dowry","IPC_498A,IPC_323","498A for cruelty"
"Mobile stolen from bag","IPC_379","379 for theft"
```

Sources for real data:
- GitHub: Legal-NLP-EkStep (Indian legal NLP datasets)
- Kaggle: search "Indian FIR dataset"
- IndianKanoon.org: scrape judgments

Run synthetic data generator to bootstrap:
```bash
python training/data_prep/generate_synthetic_data.py
```

---

## 3. Legal QA Pairs (LLM fine-tuning)

Create `training/data_prep/raw/judgment_qa.csv`:
```
question,answer
"What is IPC 302?","Murder — punishable by death or life imprisonment..."
"What to do if police refuse FIR?","Write to SP under CrPC 154(3) or approach Magistrate..."
```

---

## Minimum to Start Training

| Stage | Minimum |
|-------|---------|
| RAG Vector DB | 1 PDF (IPC) |
| Classifier | 200 labeled samples |
| LLM fine-tuning | 500 QA pairs |

Quick start with synthetic data:
```bash
python training/data_prep/generate_synthetic_data.py
python training/scripts/prepare_dataset.py
python training/scripts/build_vector_db.py
python training/scripts/train_classifier.py
python training/scripts/train_llm.py --model gpt2
```
