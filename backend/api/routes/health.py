"""backend/api/routes/health.py — Health check endpoint."""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: dict
    version: str


@router.get("/health", response_model=HealthResponse, summary="API health check")
async def health_check(request: Request):
    state = request.app.extra.get("app_state", {})
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded={
            "ipc_classifier": state.get("ipc_classifier") is not None,
            "rag_engine":     state.get("rag_engine") is not None,
            "ocr_module":     state.get("ocr_module") is not None,
            "roadmap_engine": state.get("roadmap_engine") is not None,
        },
        version="1.0.0",
    )
