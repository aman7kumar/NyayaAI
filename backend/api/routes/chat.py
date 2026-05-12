from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.modules.chat_llm import build_llm_client
from backend.modules.chat_service import (
    apply_response_safety,
    build_case_context,
    build_prompt,
    compute_fir_analysis,
)
from backend.modules.chat_store import ChatStore

router = APIRouter(prefix="/chat", tags=["Case Chat"])


class ChatStartRequest(BaseModel):
    userId: Optional[str] = None
    caseContext: Dict[str, Any] = Field(default_factory=dict)


class ChatMessageRequest(BaseModel):
    sessionId: str
    message: str = Field(..., min_length=2)


class ChatContextUpdateRequest(BaseModel):
    sessionId: str
    fir_text: Optional[str] = None
    fir_analysis: Optional[Dict[str, Any]] = None


def _get_store(request: Request) -> ChatStore:
    store = request.app.extra.get("chat_store")
    if not store:
        raise HTTPException(503, "Chat store unavailable.")
    return store


def _get_llm(request: Request):
    llm = request.app.extra.get("chat_llm")
    if not llm:
        raise HTTPException(503, "Chat model unavailable.")
    return llm


@router.post("/start")
async def start_chat_session(body: ChatStartRequest, request: Request):
    store = _get_store(request)
    context = build_case_context(body.caseContext)
    session = store.create_session(user_id=body.userId, case_context=context)
    return {
        "sessionId": session["sessionId"],
        "userId": session["userId"],
        "caseContext": session["caseContext"],
        "messages": session["messages"],
        "createdAt": session["createdAt"],
    }


@router.post("/message")
async def send_chat_message(body: ChatMessageRequest, request: Request):
    store = _get_store(request)
    llm = _get_llm(request)
    session = store.get_session(body.sessionId)
    if not session:
        raise HTTPException(404, "Session not found.")

    user_message = body.message.strip()
    store.append_message(body.sessionId, "user", user_message)

    history = store.get_messages(body.sessionId)
    fir_analysis = compute_fir_analysis(user_message, session.get("caseContext", {}))
    prompt_payload = build_prompt(
        case_context=session.get("caseContext", {}),
        messages=[{"role": m["role"], "content": m["content"]} for m in history],
        fir_analysis=fir_analysis,
    )

    raw = llm.generate(
        system_prompt=prompt_payload["system_prompt"],
        messages=prompt_payload["messages"],
    )
    safe_response = apply_response_safety(raw, session.get("caseContext", {}))
    store.append_message(body.sessionId, "assistant", safe_response)

    suggestions: List[str] = []
    if fir_analysis.get("missing_details"):
        suggestions.append("Add missing detail")
    if fir_analysis.get("bias_indicators"):
        suggestions.append("Document possible police bias")
    suggestions.append("Legal step")

    return {
        "sessionId": body.sessionId,
        "assistantMessage": safe_response,
        "firAnalysis": fir_analysis,
        "suggestions": suggestions,
    }


@router.post("/context")
async def update_chat_context(body: ChatContextUpdateRequest, request: Request):
    store = _get_store(request)
    session = store.get_session(body.sessionId)
    if not session:
        raise HTTPException(404, "Session not found.")

    context_updates: Dict[str, Any] = {}
    if body.fir_text:
        fir_data = dict(session.get("caseContext", {}).get("fir_data", {}))
        fir_data["uploaded_fir_text"] = body.fir_text
        context_updates["fir_data"] = fir_data
    if body.fir_analysis:
        context_updates["fir_analysis"] = body.fir_analysis

    updated = store.update_case_context(body.sessionId, context_updates)
    return {
        "sessionId": body.sessionId,
        "caseContext": updated or session.get("caseContext", {}),
    }


@router.get("/{session_id}")
async def get_chat_history(session_id: str, request: Request):
    store = _get_store(request)
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found.")

    return {
        "sessionId": session["sessionId"],
        "userId": session.get("userId"),
        "caseContext": session.get("caseContext", {}),
        "messages": session.get("messages", []),
        "updatedAt": session.get("updatedAt"),
    }
