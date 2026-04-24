"""
backend/api/routes/roadmap.py
==============================
Dedicated roadmap endpoint (can be called independently).
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class RoadmapRequest(BaseModel):
    query: str
    query_type: Optional[str] = None     # auto-detect if not provided
    ipc_sections: Optional[list[str]] = []


class RoadmapResponse(BaseModel):
    roadmap: list[dict]
    total_steps: int
    urgency_level: str   # "immediate" / "within_24h" / "within_week"
    legal_aid_contacts: list[dict]


@router.post(
    "/roadmap",
    response_model=RoadmapResponse,
    summary="Generate a step-by-step legal action roadmap",
)
async def generate_roadmap(body: RoadmapRequest, request: Request):
    """
    Returns a step-by-step action plan:
    - What to do first
    - Whom to approach (Police / Court / Consumer Forum / Labour Court etc.)
    - What documents to carry
    - Timeline for each action
    - Legal aid contact info
    """
    state = request.app.extra.get("app_state", {})
    roadmap_eng = state.get("roadmap_engine")

    if not roadmap_eng:
        raise HTTPException(503, "Roadmap engine not loaded.")

    from backend.modules.query_classifier import QueryClassifier
    query_type = body.query_type
    if not query_type:
        query_type = QueryClassifier().classify(body.query)

    steps = roadmap_eng.generate_roadmap(
        query=body.query,
        query_type=query_type,
        entities={},
        ipc_sections=[{"section": s} for s in body.ipc_sections],
    )

    urgency = roadmap_eng.assess_urgency(body.query, query_type)
    contacts = roadmap_eng.get_legal_aid_contacts()

    return RoadmapResponse(
        roadmap=steps,
        total_steps=len(steps),
        urgency_level=urgency,
        legal_aid_contacts=contacts,
    )
