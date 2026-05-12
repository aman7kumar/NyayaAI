"""
backend/api/routes/analyze.py
==============================
Core legal analysis endpoint.
Accepts text/query input, runs the full pipeline:
  query classify → entity extract → RAG retrieve → IPC predict → explain
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from backend.modules.query_classifier import QueryClassifier
from backend.modules.entity_extractor import EntityExtractor
from backend.modules.explainability import ExplainabilityModule
from backend.modules.multilingual import MultilingualModule

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request / Response Schemas ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=5, description="User's legal query or FIR text")
    language: Optional[str] = Field("auto", description="'en', 'hi', or 'auto'")
    include_explanation: bool = Field(True, description="Include XAI reasoning")
    include_roadmap: bool = Field(True, description="Include action roadmap")


class IPCSection(BaseModel):
    section: str          # e.g. "IPC 323"
    title: str            # e.g. "Voluntarily causing hurt"
    description: str      # Plain-language description
    punishment: str       # Punishment details
    confidence: float     # Model confidence 0-1
    statute_text: str     # Actual law text (for explainability)


class RoadmapStep(BaseModel):
    step_number: int
    action: str                        # What to do
    whom_to_approach: str              # Police / Court / Consumer Forum etc.
    timeline: str                      # "Immediately" / "Within 24 hrs" etc.
    documents_needed: list[str]
    tips: str
    # ── ADDED: fields present in accused roadmap steps ──────────────────────
    warning: Optional[str] = None      # ⚠️ warning box shown for accused steps
    type: Optional[str] = None         # "victim" | "accused" — drives UI styling


class AnalyzeResponse(BaseModel):
    query_type: str                    # criminal / civil / consumer / family
    user_role: str                     # "victim" | "accused" — ALWAYS returned
    detected_language: str
    translated_query: Optional[str]    # If input was Hindi
    entities: dict                     # Extracted legal entities
    ipc_sections: list[IPCSection]
    crpc_sections: list[IPCSection]
    rag_context: list[str]             # Retrieved statute chunks
    explanation: Optional[str]         # XAI reasoning
    roadmap: Optional[list[RoadmapStep]]
    summary: str                       # Plain-language summary for citizen


# ── Dependency: get app_state models ─────────────────────────────────────────

def get_models(request: Request):
    return request.app.extra.get("app_state") or request.state.__dict__.get("app_state")


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a legal query or FIR text",
)
async def analyze_legal_query(
    body: AnalyzeRequest,
    request: Request,
):
    """
    Full pipeline:
    1. Detect / translate language
    2. Classify query domain
    3. Extract legal entities (NER)
    4. RAG: retrieve relevant IPC/CrPC statute chunks
    5. Run fine-tuned IPC classifier
    6. Generate XAI explanation
    7. Detect user role (victim / accused) — ALWAYS, not just when roadmap enabled
    8. Build citizen roadmap
    9. Return structured JSON
    """
    state = request.app.extra.get("app_state", {})
    ipc_clf: "IPCClassifier" = state.get("ipc_classifier")
    rag: "RAGEngine" = state.get("rag_engine")
    roadmap_eng: "RoadmapEngine" = state.get("roadmap_engine")

    if not ipc_clf or not rag:
        raise HTTPException(503, "Models not loaded yet. Please wait.")

    query = body.query.strip()

    # ── Step 1: Language Detection & Translation ───────────────────────────
    multilingual = MultilingualModule()
    detected_lang = multilingual.detect_language(query)
    translated_query = None
    working_query = query

    if detected_lang != "en":
        translated_query = multilingual.translate_to_english(query, src=detected_lang)
        working_query = translated_query
        logger.info(f"Translated [{detected_lang}→en]: {working_query[:80]}")

    # ── Step 2: Query Domain Classification ───────────────────────────────
    query_clf = QueryClassifier()
    query_type = query_clf.classify(working_query)
    logger.info(f"Query type: {query_type}")

    # ── Step 3: Entity Extraction ──────────────────────────────────────────
    extractor = EntityExtractor()
    entities = extractor.extract(working_query)

    # ── Step 4: RAG — retrieve relevant statute chunks ────────────────────
    rag_chunks = rag.retrieve(working_query, top_k=5)

    # ── Step 5: IPC / CrPC section prediction ─────────────────────────────
    predictions = ipc_clf.predict(working_query, context_chunks=rag_chunks)
    ipc_sections  = [s for s in predictions if s["act"] == "IPC"]
    crpc_sections = [s for s in predictions if s["act"] == "CrPC"]

    def _to_section_obj(s: dict) -> IPCSection:
        return IPCSection(
            section=s["section"],
            title=s["title"],
            description=s["description"],
            punishment=s.get("punishment", ""),
            confidence=round(s["confidence"], 3),
            statute_text=s.get("statute_text", ""),
        )

    ipc_objs  = [_to_section_obj(s) for s in ipc_sections]
    crpc_objs = [_to_section_obj(s) for s in crpc_sections]

    # ── Step 6: Explainability ─────────────────────────────────────────────
    explanation_text = None
    if body.include_explanation:
        xai = ExplainabilityModule()
        explanation_text = xai.generate_explanation(
            query=working_query,
            predicted_sections=ipc_sections + crpc_sections,
            rag_chunks=rag_chunks,
        )

    # ── Step 7: Detect user role — ALWAYS run this, outside the roadmap block
    #    This ensures user_role is ALWAYS in the response even if roadmap is off.
    # ──────────────────────────────────────────────────────────────────────────
    user_role = "victim"  # safe default
    if roadmap_eng:
        user_role = roadmap_eng.detect_user_role(working_query)
    logger.info(f"Detected user_role: {user_role}")

    # ── Step 8: Citizen Roadmap ────────────────────────────────────────────
    roadmap_steps = None
    if body.include_roadmap and roadmap_eng:
        raw_steps = roadmap_eng.generate_roadmap(
            query=working_query,
            query_type=query_type,
            entities=entities,
            ipc_sections=ipc_sections,
            user_role=user_role,           # ← always correct role now
        )
        # Use model_validate / dict unpacking safely — extra fields tolerated
        # because RoadmapStep now declares `warning` and `type` as Optional.
        roadmap_steps = [RoadmapStep(**s) for s in raw_steps]

    # ── Step 9: Plain-language Summary ────────────────────────────────────
    summary = rag.generate_answer(
        query=working_query,
        retrieved_chunks=rag_chunks,
        predicted_sections=ipc_sections + crpc_sections,
    )

    return AnalyzeResponse(
        query_type=query_type,
        user_role=user_role,               # ← always present in response
        detected_language=detected_lang,
        translated_query=translated_query,
        entities=entities,
        ipc_sections=ipc_objs,
        crpc_sections=crpc_objs,
        rag_context=[c["text"] for c in rag_chunks],
        explanation=explanation_text,
        roadmap=roadmap_steps,
        summary=summary,
    )