"""
backend/models/rag_engine.py
==============================
Retrieval-Augmented Generation (RAG) Engine.

Components:
  1. Retriever  — FAISS vector index of Indian legal corpus
                  Embedding: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  2. Generator  — Fine-tuned GPT-2 (or quantized LLaMA-2-7B) for legal QA
                  Generates plain-language answers grounded in retrieved chunks.

Training:
  Run training/scripts/train_llm.py to fine-tune the generator.
  Run training/scripts/build_vector_db.py to build the FAISS index.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)

FAISS_INDEX_DIR  = Path("backend/models/saved/faiss_index")
LLM_MODEL_DIR    = Path("backend/models/saved/legal_llm")
EMBEDDING_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ── Retriever ─────────────────────────────────────────────────────────────────

class LegalRetriever:
    """
    FAISS-based dense retriever over the Indian legal corpus.
    Chunks include IPC sections, CrPC sections, Constitution articles,
    High Court / Supreme Court judgments.
    """

    def __init__(self, index_dir: Path = FAISS_INDEX_DIR):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.index_dir = index_dir

        index_file = index_dir / "legal_index.faiss"
        meta_file  = index_dir / "chunk_metadata.json"

        if index_file.exists() and meta_file.exists():
            self.index = faiss.read_index(str(index_file))
            import json
            #with open(meta_file) as f:
             #   self.metadata = json.load(f)  # list of {text, source, section, act}
            with open(meta_file, encoding="utf-8") as f:
                self.metadata = json.load(f)  # list of {text, source, section, act}
            logger.info(f"✅  FAISS index loaded — {self.index.ntotal} vectors")
        else:
            logger.warning(
                "⚠️  FAISS index not found. Retriever will return empty results. "
                "Run training/scripts/build_vector_db.py to build the index."
            )
            self.index    = None
            self.metadata = []

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Embed query → search FAISS → return top-k chunks with metadata.
        """
        if self.index is None:
            return []

        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_vec.astype(np.float32), top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                chunk = dict(self.metadata[idx])
                chunk["score"] = float(dist)
                results.append(chunk)

        return results


# ── Generator ─────────────────────────────────────────────────────────────────

class LegalGenerator:
    """
    Fine-tuned causal LM (GPT-2 or LLaMA-2-7B quantized) for legal answer generation.
    Takes a query + retrieved context and generates a plain-language answer.
    """

    MAX_NEW_TOKENS = 300
    TEMPERATURE    = 0.3   # Low temp for factual legal answers
    """
    def __init__(self, model_dir: Path = LLM_MODEL_DIR):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_dir.exists():
            logger.info(f"Loading fine-tuned LLM from {model_dir} …")
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
        else:
            logger.warning(
                "⚠️  Fine-tuned LLM not found. Falling back to base GPT-2. "
                "Run training/scripts/train_llm.py to fine-tune."
            )
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
        )
        logger.info("✅  LegalGenerator ready.")
        """
    def __init__(self, model_dir: Path = LLM_MODEL_DIR):
        # Always use CPU for generation to avoid GPU OOM
        device = "cpu"

        if model_dir.exists():
            logger.info(f"Loading fine-tuned LLM from {model_dir} …")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                local_files_only=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                dtype=torch.float32,
                local_files_only=True,
            )
        else:
            logger.warning("⚠️  Fine-tuned LLM not found. Falling back to base GPT-2.")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model     = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                dtype=torch.float32,
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,          # -1 = CPU always
            framework="pt",
        )
        logger.info("✅  LegalGenerator ready.")

    def generate(
        self,
        query: str,
        context_chunks: list[dict],
        predicted_sections: list[dict],
    ) -> str:
        """
        Build a structured prompt and generate a plain-language legal answer.
        """
        # Build context string from RAG chunks
        context_str = "\n".join(
            f"[{c.get('act','')} {c.get('section','')}] {c.get('text','')[:300]}"
            for c in context_chunks[:3]
        )

        # Build section string from classifier predictions
        section_str = ", ".join(
            f"{s.get('section','')} ({s.get('title','')})"
            for s in predicted_sections[:3]
        )

        prompt = (
            "You are a knowledgeable Indian legal assistant. "
            "Answer the citizen's question in simple, clear language. "
            "Base your answer ONLY on the legal context provided below. "
            "Do NOT make up laws or sections.\n\n"
            f"CITIZEN QUESTION: {query}\n\n"
            f"RELEVANT LAWS:\n{context_str}\n\n"
            f"APPLICABLE SECTIONS: {section_str}\n\n"
            "PLAIN LANGUAGE ANSWER:"
        )
        """
        output = self.pipe(
            prompt,
            max_new_tokens=self.MAX_NEW_TOKENS,
            temperature=self.TEMPERATURE,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        """
        output = self.pipe(
            prompt,
            max_new_tokens=150,
            temperature=self.TEMPERATURE,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            truncation=True,
        )

        full_text: str = output[0]["generated_text"]
        # Extract only the generated answer (after the prompt)
        answer = full_text[len(prompt):].strip()
        return answer


# ── Unified RAG Engine ────────────────────────────────────────────────────────

class RAGEngine:
    """
    Combines retriever + generator into one interface.
    """

    def __init__(self, retriever: LegalRetriever, generator: LegalGenerator):
        self.retriever = retriever
        self.generator = generator

    @classmethod
    def load(cls) -> "RAGEngine":
        return cls(
            retriever=LegalRetriever(),
            generator=LegalGenerator(),
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-k relevant statute chunks for a query."""
        return self.retriever.retrieve(query, top_k=top_k)

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: list[dict],
        predicted_sections: list[dict],
    ) -> str:
        """Generate a plain-language answer grounded in retrieved chunks."""
        return self.generator.generate(query, retrieved_chunks, predicted_sections)
