"""
backend/models/rag_engine.py
RAG Engine with Mistral API (primary) + GPT-2 fallback.
"""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAISS_INDEX_DIR = PROJECT_ROOT / "backend" / "models" / "saved" / "faiss_index"
LLM_MODEL_DIR = PROJECT_ROOT / "backend" / "models" / "saved" / "legal_llm"
EMBEDDING_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ── Retriever ─────────────────────────────────────────────────────────────────

class LegalRetriever:
    def __init__(self, index_dir: Path = FAISS_INDEX_DIR):
        import faiss

        index_file = index_dir / "legal_index.faiss"
        meta_file = index_dir / "chunk_metadata.json"

        if index_file.exists() and meta_file.exists():
            self.index = faiss.read_index(str(index_file))
            with open(meta_file, encoding="utf-8") as f:
                self.metadata = json.load(f)
            try:
                from sentence_transformers import SentenceTransformer

                self.embedder = SentenceTransformer(
                    EMBEDDING_MODEL,
                    local_files_only=True,
                )
                logger.info("FAISS index loaded (%s vectors)", self.index.ntotal)
            except Exception as e:
                logger.warning("Embedding model unavailable locally, RAG disabled: %s", e)
                self.index = None
                self.metadata = []
                self.embedder = None
        else:
            logger.warning("FAISS index not found. RAG disabled.")
            self.index = None
            self.metadata = []
            self.embedder = None

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index is None or self.embedder is None:
            return []
        query_vec          = self.embedder.encode(
            [query], normalize_embeddings=True
        )
        distances, indices = self.index.search(
            query_vec.astype(np.float32), top_k
        )
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                chunk          = dict(self.metadata[idx])
                chunk["score"] = float(dist)
                results.append(chunk)
        return results


# ── Generator ─────────────────────────────────────────────────────────────────

class LegalGenerator:
    """
    Uses Mistral API (primary — best quality, no local storage).
    Falls back to GPT-2 if API not available.
    """

    MISTRAL_MODEL = "mistral-small-latest"   # Free tier model

    def __init__(self, model_dir: Path = LLM_MODEL_DIR):
        self.mistral_available = False
        self.mistral_client    = None

        # Load API key from .env file
        self._load_env()

        # Try Mistral API
        self._check_mistral()

        # Load GPT-2 fallback if Mistral not available
        if not self.mistral_available:
            self._load_gpt2(model_dir)

        logger.info("✅ LegalGenerator ready.")

    def _load_env(self):
        """Load environment variables from .env file."""
        try:
            from dotenv import load_dotenv
            env_candidates = [
                PROJECT_ROOT / ".env",
                PROJECT_ROOT / "backend" / ".env",
            ]
            for env_file in env_candidates:
                if env_file.exists():
                    load_dotenv(env_file, override=False)
                    logger.info(".env file loaded from %s", env_file)
                    break
        except ImportError:
            pass

    def _check_mistral(self):
        """Initialize Mistral client using API key from environment."""
        api_key = os.environ.get("MISTRAL_API_KEY", "").strip()

        if not api_key:
            logger.warning(
                "MISTRAL_API_KEY not found in environment. "
                "Add it to your .env file. Using GPT-2 fallback."
            )
            return

        try:
            from mistralai import Mistral
            self.mistral_client    = Mistral(api_key=api_key)
            self.mistral_available = True
            logger.info(
                f"✅ Mistral API connected — model: {self.MISTRAL_MODEL}"
            )
        except ImportError:
            logger.warning("mistralai not installed. Run: pip install mistralai")
        except Exception as e:
            logger.warning(f"Mistral connection failed: {e}")

    def _load_gpt2(self, model_dir):
        """Load GPT-2 as fallback."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        device = torch.device("cpu")
        try:
            if model_dir and Path(model_dir).exists():
                logger.info(f"Loading fine-tuned GPT-2 from {model_dir}...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(model_dir), local_files_only=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_dir),
                    dtype=torch.float32,
                    local_files_only=True,
                )
            else:
                logger.warning("Fine-tuned LLM not found locally. GPT-2 disabled; using static fallback.")
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
                self.model     = AutoModelForCausalLM.from_pretrained(
                    "gpt2", dtype=torch.float32, local_files_only=True
                )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model.eval()
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,
                framework="pt",
            )
            logger.info("✅ GPT-2 fallback loaded.")
        except Exception as e:
            logger.error(f"GPT-2 load failed: {e}")
            self.pipe = None

    def generate(
        self,
        query:              str,
        context_chunks:     list[dict],
        predicted_sections: list[dict],
    ) -> str:
        """Generate legal answer using Mistral or GPT-2 fallback."""

        context_str = "\n".join(
            f"[{c.get('act','')} {c.get('section','')}] "
            f"{c.get('text','')[:300]}"
            for c in context_chunks[:3]
        )

        section_str = ", ".join(
            f"{s.get('section','')} — {s.get('title','')}"
            for s in predicted_sections[:3]
        ) if predicted_sections else "sections to be determined"

        user_prompt = (
            f"A citizen in India needs urgent legal help.\n\n"
            f"THEIR SITUATION:\n{query}\n\n"
            f"APPLICABLE LAW SECTIONS: {section_str}\n\n"
            f"RELEVANT LAW TEXT:\n{context_str}\n\n"
            f"Please provide comprehensive legal guidance with:\n\n"
            f"1. **What Happened Legally** — Which laws apply and why "
            f"in simple words\n"
            f"2. **Your Rights** — What rights the citizen has under "
            f"Indian law\n"
            f"3. **Immediate Steps** — Step by step what to do right now\n"
            f"4. **Where to Go** — Exact authority to approach "
            f"(police station / court / consumer forum)\n"
            f"5. **Documents to Carry** — What evidence/papers to bring\n"
            f"6. **Important Helplines** — Relevant emergency numbers\n\n"
            f"Write in simple English that any common Indian citizen can "
            f"understand. Be specific, practical, and empathetic."
        )

        if self.mistral_available:
            return self._mistral_generate(user_prompt)

        return self._gpt2_generate(user_prompt)

    def _mistral_generate(self, user_prompt: str) -> str:
        """Generate using Mistral API."""
        try:
            response = self.mistral_client.chat.complete(
                model=self.MISTRAL_MODEL,
                messages=[
                    {
                        "role":    "system",
                        "content": (
                            "You are NyayaAI, an expert Indian legal assistant "
                            "helping common citizens understand their rights.\n\n"
                            "Your role:\n"
                            "- Explain Indian laws (IPC, CrPC, Consumer Protection "
                            "Act, IT Act etc.) in simple language\n"
                            "- Always cite specific section numbers\n"
                            "- Give practical, actionable guidance\n"
                            "- Be empathetic — citizens are often scared and confused\n"
                            "- Use simple Hindi-English that common people understand\n"
                            "- Always recommend consulting a lawyer for serious matters\n\n"
                            "Important: Never provide wrong legal information. "
                            "If unsure about something, clearly say so and recommend "
                            "consulting a qualified lawyer."
                        ),
                    },
                    {
                        "role":    "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.3,
                max_tokens=800,
            )

            answer = response.choices[0].message.content.strip()
            if answer and len(answer) > 30:
                return answer

            logger.warning("Mistral returned empty response")
            return self._gpt2_generate(user_prompt)

        except Exception as e:
            logger.warning(f"Mistral API call failed: {e}")
            return self._gpt2_generate(user_prompt)

    def _gpt2_generate(self, prompt: str) -> str:
        """GPT-2 fallback with structured answer."""
        if hasattr(self, "pipe") and self.pipe is not None:
            try:
                output = self.pipe(
                    prompt[:500],
                    max_new_tokens=200,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    truncation=True,
                )
                full   = output[0]["generated_text"]
                answer = full[len(prompt):].strip()
                if len(answer) > 50:
                    return answer
            except Exception:
                pass

        # Structured fallback — always gives a useful answer
        return (
            "Based on your situation, here is what you should know:\n\n"
            "**Your Rights:**\n"
            "• You have the right to file an FIR at any police station.\n"
            "• Police cannot refuse to register an FIR for a cognizable "
            "offence (CrPC Section 154).\n"
            "• You are entitled to a free copy of the FIR.\n"
            "• Free legal aid is available through District Legal Services "
            "Authority (DLSA).\n\n"
            "**Immediate Steps:**\n"
            "1. Ensure your safety — call 112 if in immediate danger.\n"
            "2. Collect and preserve all evidence (photos, messages, "
            "witness names).\n"
            "3. Visit your nearest police station and file an FIR.\n"
            "4. Get a medical examination if you are injured and collect "
            "the injury certificate.\n"
            "5. Consult a lawyer — free legal aid available at DLSA "
            "(call 15100).\n\n"
            "**Emergency Contacts:**\n"
            "• Police: 100  |  Emergency: 112  |  Women Helpline: 181\n"
            "• Legal Aid (NALSA): 15100  |  Cyber Crime: 1930\n\n"
            "⚠️ This is preliminary information. Please consult a qualified "
            "lawyer for advice specific to your case."
        )


# ── RAG Engine ────────────────────────────────────────────────────────────────

class RAGEngine:
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
        return self.retriever.retrieve(query, top_k=top_k)

    def generate_answer(
        self,
        query:              str,
        retrieved_chunks:   list[dict],
        predicted_sections: list[dict],
    ) -> str:
        return self.generator.generate(
            query, retrieved_chunks, predicted_sections
        )