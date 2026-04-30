"""
backend/api/main.py — Fixed complete version
"""

import logging
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env FIRST
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup."""
    logger.info("🔄  Loading models …")
    app.extra["app_state"] = app_state
    app.state.app_state = app_state

    # IPC Classifier
    try:
        from backend.models.ipc_classifier import IPCClassifier
        app_state["ipc_classifier"] = IPCClassifier.load()
        logger.info("✅  IPC Classifier loaded")
    except Exception as e:
        logger.error(f"❌  IPC Classifier failed: {e}")
        app_state["ipc_classifier"] = None

    # RAG Engine
    try:
        from backend.models.rag_engine import RAGEngine
        app_state["rag_engine"] = RAGEngine.load()
        logger.info("✅  RAG Engine loaded")
    except Exception as e:
        logger.error(f"❌  RAG Engine failed: {e}")
        app_state["rag_engine"] = None

    # OCR Module
    try:
        from backend.models.ocr_module import OCRModule
        app_state["ocr_module"] = OCRModule()
        logger.info("✅  OCR Module loaded")
    except Exception as e:
        logger.error(f"❌  OCR Module failed: {e}")
        app_state["ocr_module"] = None

    # Roadmap Engine
    try:
        from backend.models.roadmap_engine import RoadmapEngine
        app_state["roadmap_engine"] = RoadmapEngine()
        logger.info("✅  Roadmap Engine loaded")
    except Exception as e:
        logger.error(f"❌  Roadmap Engine failed: {e}")
        app_state["roadmap_engine"] = None

    logger.info("✅  All models loaded and ready.")
    yield
    app_state.clear()
    logger.info("🛑  Models unloaded.")


app = FastAPI(
    title="NyayaAI — Indian Legal Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "⚖️ NyayaAI API running", "docs": "/docs"}


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "models": {
            "ipc_classifier": app_state.get("ipc_classifier") is not None,
            "rag_engine":     app_state.get("rag_engine")     is not None,
            "ocr_module":     app_state.get("ocr_module")     is not None,
            "roadmap_engine": app_state.get("roadmap_engine") is not None,
        }
    }


@app.post("/api/v1/analyze")
async def analyze(body: dict):
    """Full legal analysis pipeline."""
    query = body.get("query", "").strip()
    if not query or len(query) < 5:
        raise HTTPException(400, "Query too short. Please describe your situation.")

    ipc_clf     = app_state.get("ipc_classifier")
    rag         = app_state.get("rag_engine")
    roadmap_eng = app_state.get("roadmap_engine")

    if not ipc_clf or not rag:
        raise HTTPException(503, "Models not loaded yet. Please wait and retry.")

    # 1. Language detection
    try:
        from backend.modules.multilingual import MultilingualModule
        ml               = MultilingualModule()
        detected_lang    = ml.detect_language(query)
        translated_query = None
        working_query    = query
        if detected_lang != "en":
            translated_query = ml.translate_to_english(query, src=detected_lang)
            working_query    = translated_query
    except Exception:
        detected_lang    = "en"
        translated_query = None
        working_query    = query

    # 2. Query classification
    try:
        from backend.modules.query_classifier import QueryClassifier
        query_type = QueryClassifier().classify(working_query)
    except Exception:
        query_type = "criminal"

    # 3. Entity extraction
    try:
        from backend.modules.entity_extractor import EntityExtractor
        entities = EntityExtractor().extract(working_query)
    except Exception:
        entities = {}

    # 4. RAG retrieval
    try:
        rag_chunks = rag.retrieve(working_query, top_k=5)
    except Exception:
        rag_chunks = []

    # 5. IPC prediction
    try:
        predictions   = ipc_clf.predict(working_query, context_chunks=rag_chunks)
        ipc_sections  = [s for s in predictions if s.get("act") == "IPC"]
        crpc_sections = [s for s in predictions if s.get("act") == "CrPC"]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        ipc_sections  = []
        crpc_sections = []

    # 6. Explainability
    try:
        from backend.modules.explainability import ExplainabilityModule
        explanation = ExplainabilityModule().generate_explanation(
            query=working_query,
            predicted_sections=ipc_sections + crpc_sections,
            rag_chunks=rag_chunks,
        )
    except Exception:
        explanation = "Explanation not available."

    # 7. Roadmap
    try:
        roadmap = roadmap_eng.generate_roadmap(
            query=working_query,
            query_type=query_type,
            entities=entities,
            ipc_sections=ipc_sections,
        )
    except Exception:
        roadmap = []

    # 8. Summary via Mistral/GPT-2
    try:
        summary = rag.generate_answer(
            query=working_query,
            retrieved_chunks=rag_chunks,
            predicted_sections=ipc_sections + crpc_sections,
        )
    except Exception:
        summary = (
            f"Based on your description, this appears to be a {query_type} matter. "
            "Please file an FIR at your nearest police station and consult a lawyer."
        )

    return {
        "query_type":        query_type,
        "detected_language": detected_lang,
        "translated_query":  translated_query,
        "entities":          entities,
        "ipc_sections":      ipc_sections,
        "crpc_sections":     crpc_sections,
        "rag_context":       [c.get("text", "") for c in rag_chunks],
        "explanation":       explanation,
        "roadmap":           roadmap,
        "summary":           summary,
    }


@app.post("/api/v1/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    """Extract text from image or PDF using PaddleOCR/EasyOCR."""
    ocr = app_state.get("ocr_module")
    if not ocr:
        raise HTTPException(503, "OCR module not loaded.")

    allowed = {"image/jpeg", "image/png", "image/webp", "application/pdf"}
    if file.content_type not in allowed:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    content = await file.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(413, "File too large. Maximum 15MB.")

    try:
        result   = ocr.extract(io.BytesIO(content), mime=file.content_type)
        raw_text = result.get("text", "").strip()
        return {
            "raw_text":          raw_text,
            "confidence":        result.get("confidence", 0.0),
            "detected_language": result.get("language",   "unknown"),
            "word_count":        len(raw_text.split()) if raw_text else 0,
            "engine_used":       result.get("engine",     "unknown"),
            "pages":             result.get("pages",      1),
        }
    except Exception as e:
        logger.exception("OCR failed")
        raise HTTPException(500, f"OCR error: {str(e)}")


@app.post("/api/v1/roadmap")
async def get_roadmap(body: dict):
    """Generate legal action roadmap."""
    roadmap_eng = app_state.get("roadmap_engine")
    if not roadmap_eng:
        raise HTTPException(503, "Roadmap engine not loaded.")

    query      = body.get("query", "")
    query_type = body.get("query_type", "criminal")

    steps = roadmap_eng.generate_roadmap(
        query=query, query_type=query_type,
        entities={}, ipc_sections=[],
    )
    return {
        "roadmap":            steps,
        "total_steps":        len(steps),
        "urgency_level":      roadmap_eng.assess_urgency(query, query_type),
        "legal_aid_contacts": roadmap_eng.get_legal_aid_contacts(),
    }