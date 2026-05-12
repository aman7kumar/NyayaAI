"""
backend/api/main.py
"""

import logging
import io
import os
import sys
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env FIRST
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes.chat import router as chat_router
from backend.modules.chat_llm import build_llm_client
from backend.modules.chat_store import ChatStore
from backend.modules.ocr_postprocess import improve_ocr_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
app_state: dict = {}

# Event set when OCR init finishes (success or failure)
_ocr_ready = threading.Event()


def _safe_print(message: str) -> None:
    """Print message without crashing on Windows cp1252 consoles."""
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(message.encode("ascii", "replace").decode("ascii"), flush=True)


def _init_ocr_background():
    """Initialize OCR in a background thread."""
    # ✅ Ensure the project root is on sys.path inside the thread
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("🔄  [OCR thread] Importing OCRModule …")
    _safe_print("🔄  [OCR thread] Importing OCRModule …")

    try:
        from backend.models.ocr_module import OCRModule
        logger.info("🔄  [OCR thread] OCRModule imported, calling OCRModule() …")
        _safe_print("🔄  [OCR thread] Calling OCRModule() …")

        ocr = OCRModule()
        app_state["ocr_module"] = ocr

        if ocr._engine:
            msg = f"✅  [OCR thread] OCR ready — engine: {ocr._engine}"
        else:
            msg = "⚠️  [OCR thread] OCR initialized but no engine available"

        logger.info(msg)
        _safe_print(msg)

    except Exception as e:
        err = f"❌  [OCR thread] OCR Module failed: {e}\n{traceback.format_exc()}"
        logger.error(err)
        _safe_print(err)
        app_state["ocr_module"] = None

    finally:
        _ocr_ready.set()
        _safe_print("🏁  [OCR thread] Init complete (event set)")
        logger.info("🏁  [OCR thread] Init complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🔄  Loading models …")
    app.extra["app_state"] = app_state
    app.state.app_state    = app_state

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

    # OCR — background thread
    _ocr_thread = threading.Thread(
        target=_init_ocr_background,
        name="ocr-init",
        daemon=True,
    )
    _ocr_thread.start()
    logger.info("🔄  OCR initializing in background …")

    # Roadmap Engine
    try:
        from backend.models.roadmap_engine import RoadmapEngine
        app_state["roadmap_engine"] = RoadmapEngine()
        logger.info("✅  Roadmap Engine loaded")
    except Exception as e:
        logger.error(f"❌  Roadmap Engine failed: {e}")
        app_state["roadmap_engine"] = None

    # Chat stack
    try:
        app.extra["chat_store"] = ChatStore()
        app.extra["chat_llm"] = build_llm_client()
        logger.info("✅  Case chat stack initialized")
    except Exception as e:
        logger.error(f"❌  Case chat initialization failed: {e}")
        app.extra["chat_store"] = None
        app.extra["chat_llm"] = None

    logger.info("✅  Core models loaded. OCR still initializing in background.")
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

app.include_router(chat_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "⚖️ NyayaAI API running", "docs": "/docs"}


@app.get("/api/v1/health")
async def health():
    ocr      = app_state.get("ocr_module")
    ocr_done = _ocr_ready.is_set()
    return {
        "status": "ok",
        "models": {
            "ipc_classifier": app_state.get("ipc_classifier") is not None,
            "rag_engine":     app_state.get("rag_engine")     is not None,
            "ocr_module":     ocr is not None,
            "ocr_engine":     (
                ocr._engine if ocr
                else ("initializing…" if not ocr_done else "failed")
            ),
            "roadmap_engine": app_state.get("roadmap_engine") is not None,
        }
    }


@app.post("/api/v1/analyze")
async def analyze(body: dict):
    query = body.get("query", "").strip()
    if not query or len(query) < 5:
        raise HTTPException(400, "Query too short.")

    ipc_clf     = app_state.get("ipc_classifier")
    rag         = app_state.get("rag_engine")
    roadmap_eng = app_state.get("roadmap_engine")

    if not ipc_clf or not rag:
        raise HTTPException(503, "Models not loaded yet. Please wait and retry.")

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

    try:
        from backend.modules.query_classifier import QueryClassifier
        query_type = QueryClassifier().classify(working_query)
    except Exception:
        query_type = "criminal"

    try:
        from backend.modules.entity_extractor import EntityExtractor
        entities = EntityExtractor().extract(working_query)
    except Exception:
        entities = {}

    try:
        rag_chunks = rag.retrieve(working_query, top_k=5)
    except Exception:
        rag_chunks = []

    try:
        predictions   = ipc_clf.predict(working_query, context_chunks=rag_chunks)
        ipc_sections  = [s for s in predictions if s.get("act") == "IPC"]
        crpc_sections = [s for s in predictions if s.get("act") == "CrPC"]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        ipc_sections  = []
        crpc_sections = []

    try:
        from backend.modules.explainability import ExplainabilityModule
        explanation = ExplainabilityModule().generate_explanation(
            query=working_query,
            predicted_sections=ipc_sections + crpc_sections,
            rag_chunks=rag_chunks,
        )
    except Exception:
        explanation = "Explanation not available."

    user_role = "victim"
    roadmap   = []
    try:
        user_role = roadmap_eng.detect_user_role(working_query)
        roadmap   = roadmap_eng.generate_roadmap(
            query=working_query, query_type=query_type,
            entities=entities, ipc_sections=ipc_sections,
            user_role=user_role,
        )
    except Exception as e:
        logger.error(f"Roadmap error: {e}")

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
        "user_role":         user_role,
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
    """Extract text from image or PDF."""

    # Wait up to 180s for OCR to finish initializing
    if not _ocr_ready.is_set():
        logger.info("⏳  OCR request waiting for background init to complete …")
        finished = _ocr_ready.wait(timeout=180)
        if not finished:
            raise HTTPException(
                503,
                "OCR is still initializing. Please try again in a moment."
            )

    ocr = app_state.get("ocr_module")
    if not ocr:
        raise HTTPException(503, "OCR failed to initialize. Check server logs.")
    if ocr._engine is None:
        raise HTTPException(503, "OCR engine unavailable. Check server logs.")

    content_type = (file.content_type or "").lower()
    if not any(t in content_type for t in ("jpeg", "jpg", "png", "webp", "pdf")):
        raise HTTPException(400, f"Unsupported type: {file.content_type}. Use JPEG/PNG/WebP/PDF.")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(413, "File too large. Max 15MB.")

    if   "pdf"  in content_type: mime = "application/pdf"
    elif "png"  in content_type: mime = "image/png"
    elif "webp" in content_type: mime = "image/webp"
    else:                         mime = "image/jpeg"

    logger.info(f"OCR request: {file.filename!r} {len(content)//1024}KB {mime}")

    try:
        result   = ocr.extract(io.BytesIO(content), mime=mime)
        raw_text = result.get("text", "").strip()
        cleaned_text = improve_ocr_text(raw_text)
        logger.info(
            f"OCR result — engine={result.get('engine')} "
            f"words={len(raw_text.split()) if raw_text else 0} "
            f"conf={result.get('confidence', 0):.2f}"
        )
        return {
            "raw_text":          raw_text,
            "cleaned_text":      cleaned_text,
            "confidence":        result.get("confidence", 0.0),
            "detected_language": result.get("language",   "unknown"),
            "word_count":        len(raw_text.split()) if raw_text else 0,
            "engine_used":       result.get("engine",     "unknown"),
            "pages":             result.get("pages",      1),
            "success":           bool(raw_text),
        }
    except Exception as e:
        logger.exception("OCR processing failed")
        raise HTTPException(500, f"OCR error: {str(e)}")


@app.post("/api/v1/roadmap")
async def get_roadmap(body: dict):
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


@app.post("/api/v1/fir/upload")
async def fir_upload(file: UploadFile = File(...)):
    """
    OCR + FIR analysis endpoint for case-chat.
    Accepts image/pdf, returns extracted FIR text + structured FIR analysis.
    """
    if not _ocr_ready.is_set():
        finished = _ocr_ready.wait(timeout=180)
        if not finished:
            raise HTTPException(503, "OCR is still initializing. Please try again.")

    ocr = app_state.get("ocr_module")
    if not ocr or ocr._engine is None:
        raise HTTPException(503, "OCR engine unavailable.")

    content_type = (file.content_type or "").lower()
    if not any(t in content_type for t in ("jpeg", "jpg", "png", "webp", "pdf")):
        raise HTTPException(400, f"Unsupported type: {file.content_type}. Use JPEG/PNG/WebP/PDF.")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(413, "File too large. Max 15MB.")

    if "pdf" in content_type:
        mime = "application/pdf"
    elif "png" in content_type:
        mime = "image/png"
    elif "webp" in content_type:
        mime = "image/webp"
    else:
        mime = "image/jpeg"

    try:
        result = ocr.extract(io.BytesIO(content), mime=mime)
        extracted_text = result.get("text", "").strip()
        cleaned_text = improve_ocr_text(extracted_text)

        from backend.modules.fir_analysis import analyze_fir_text
        fir_analysis = analyze_fir_text(cleaned_text)

        return {
            "extracted_text": cleaned_text,
            "raw_extracted_text": extracted_text,
            "fir_analysis": fir_analysis,
            "confidence": result.get("confidence", 0.0),
            "detected_language": result.get("language", "unknown"),
            "engine_used": result.get("engine", "unknown"),
            "pages": result.get("pages", 1),
            "success": bool(extracted_text),
        }
    except Exception as e:
        logger.exception("FIR OCR upload failed")
        raise HTTPException(500, f"FIR upload processing error: {str(e)}")