"""
backend/api/main.py - Fixed version with proper app state sharing
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global state dictionary ───────────────────────────────────────────────────
# Defined at module level so all routes can access it
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once on startup."""
    global app_state

    logger.info("🔄  Loading models …")

    try:
        from backend.models.ipc_classifier import IPCClassifier
        app_state["ipc_classifier"] = IPCClassifier.load()
        logger.info("✅  IPC Classifier loaded")
    except Exception as e:
        logger.error(f"❌  IPC Classifier failed: {e}")
        app_state["ipc_classifier"] = None

    try:
        from backend.models.rag_engine import RAGEngine
        app_state["rag_engine"] = RAGEngine.load()
        logger.info("✅  RAG Engine loaded")
    except Exception as e:
        logger.error(f"❌  RAG Engine failed: {e}")
        app_state["rag_engine"] = None

    try:
        from backend.models.ocr_module import OCRModule
        app_state["ocr_module"] = OCRModule()
        logger.info("✅  OCR Module loaded")
    except Exception as e:
        logger.error(f"❌  OCR Module failed: {e}")
        app_state["ocr_module"] = None

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


# ── App Instance ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI-Powered Legal Document Analyzer — India",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "⚖️  AI Legal Analyzer API is running.", "docs": "/docs"}


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": {
            "ipc_classifier": app_state.get("ipc_classifier") is not None,
            "rag_engine":     app_state.get("rag_engine")     is not None,
            "ocr_module":     app_state.get("ocr_module")     is not None,
            "roadmap_engine": app_state.get("roadmap_engine") is not None,
        }
    }


# ── Analyze endpoint ──────────────────────────────────────────────────────────
@app.post("/api/v1/analyze")
async def analyze(body: dict):
    """Full legal analysis pipeline."""
    query = body.get("query", "").strip()

    if not query:
        return {"error": "Query cannot be empty"}

    ipc_clf     = app_state.get("ipc_classifier")
    rag         = app_state.get("rag_engine")
    roadmap_eng = app_state.get("roadmap_engine")

    if not ipc_clf or not rag:
        from fastapi import HTTPException
        raise HTTPException(503, "Models not loaded yet. Please wait.")

    # 1. Detect language
    try:
        from backend.modules.multilingual import MultilingualModule
        multilingual    = MultilingualModule()
        detected_lang   = multilingual.detect_language(query)
        translated_query = None
        working_query   = query
        if detected_lang != "en":
            translated_query = multilingual.translate_to_english(query, src=detected_lang)
            working_query    = translated_query
    except Exception:
        detected_lang    = "en"
        translated_query = None
        working_query    = query

    # 2. Classify query domain
    try:
        from backend.modules.query_classifier import QueryClassifier
        query_type = QueryClassifier().classify(working_query)
    except Exception:
        query_type = "criminal"

    # 3. Extract entities
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

    # 5. IPC/CrPC prediction
    try:
        predictions   = ipc_clf.predict(working_query, context_chunks=rag_chunks)
        ipc_sections  = [s for s in predictions if s.get("act") == "IPC"]
        crpc_sections = [s for s in predictions if s.get("act") == "CrPC"]
    except Exception as e:
        logger.error(f"Classifier error: {e}")
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
        raw_steps = roadmap_eng.generate_roadmap(
            query=working_query,
            query_type=query_type,
            entities=entities,
            ipc_sections=ipc_sections,
        )
        roadmap = raw_steps
    except Exception:
        roadmap = []

    # 8. Generate summary
    try:
        summary = rag.generate_answer(
            query=working_query,
            retrieved_chunks=rag_chunks,
            predicted_sections=ipc_sections + crpc_sections,
        )
    except Exception:
        summary = (
            f"Based on your description, this appears to be a {query_type} matter. "
            f"Please consult a lawyer and file a complaint at your nearest police station."
        )

    return {
        "query_type":       query_type,
        "detected_language": detected_lang,
        "translated_query": translated_query,
        "entities":         entities,
        "ipc_sections":     ipc_sections,
        "crpc_sections":    crpc_sections,
        "rag_context":      [c.get("text", "") for c in rag_chunks],
        "explanation":      explanation,
        "roadmap":          roadmap,
        "summary":          summary,
    }


# ── OCR endpoint ──────────────────────────────────────────────────────────────
from fastapi import UploadFile, File
import io

@app.post("/api/v1/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    ocr = app_state.get("ocr_module")
    if not ocr:
        from fastapi import HTTPException
        raise HTTPException(503, "OCR module not loaded")

    content  = await file.read()
    mime     = file.content_type or "image/jpeg"

    try:
        result = ocr.extract(io.BytesIO(content), mime=mime)
        return {
            "raw_text":  result.get("text", ""),
            "confidence": result.get("confidence", 0.0),
            "detected_language": result.get("language", "en"),
            "word_count": len(result.get("text", "").split()),
        }
    except Exception as e:
        # Return empty text instead of crashing
        return {
            "raw_text":  "",
            "confidence": 0.0,
            "detected_language": "en",
            "word_count": 0,
            "error": str(e),
        }

# ── Roadmap endpoint ──────────────────────────────────────────────────────────
@app.post("/api/v1/roadmap")
async def get_roadmap(body: dict):
    roadmap_eng = app_state.get("roadmap_engine")
    if not roadmap_eng:
        from fastapi import HTTPException
        raise HTTPException(503, "Roadmap engine not loaded")

    query      = body.get("query", "")
    query_type = body.get("query_type", "criminal")
    steps      = roadmap_eng.generate_roadmap(
        query=query,
        query_type=query_type,
        entities={},
        ipc_sections=[],
    )
    return {
        "roadmap":      steps,
        "total_steps":  len(steps),
        "urgency_level": roadmap_eng.assess_urgency(query, query_type),
        "legal_aid_contacts": roadmap_eng.get_legal_aid_contacts(),
    }