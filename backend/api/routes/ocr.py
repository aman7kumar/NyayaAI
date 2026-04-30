"""
backend/api/routes/ocr.py
OCR endpoint using PaddleOCR (primary) or EasyOCR (backup).
"""

import io
import logging

from fastapi import APIRouter, File, Request, UploadFile, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class OCRResponse(BaseModel):
    raw_text:          str
    confidence:        float
    detected_language: str
    word_count:        int
    engine_used:       str
    pages:             int


@router.post(
    "/ocr",
    response_model=OCRResponse,
    summary="Extract text from FIR image or scanned document",
)
async def extract_text_from_image(
    request: Request,
    file:    UploadFile = File(...),
):
    """
    Accepts a FIR image or scanned document.
    Uses PaddleOCR (primary) or EasyOCR (backup).
    Supports Hindi + English mixed text.
    """
    state = request.app.extra.get("app_state", {})
    ocr   = state.get("ocr_module")

    if not ocr:
        raise HTTPException(503, "OCR module not loaded.")

    # Validate file type
    allowed = {
        "image/jpeg", "image/png",
        "image/webp", "application/pdf",
    }
    if file.content_type not in allowed:
        raise HTTPException(
            400,
            f"Unsupported file type: {file.content_type}. "
            f"Allowed: jpg, png, webp, pdf",
        )

    content = await file.read()
    if len(content) > 15 * 1024 * 1024:  # 15MB limit
        raise HTTPException(413, "File too large. Maximum 15MB.")

    if not content:
        raise HTTPException(400, "Empty file uploaded.")

    try:
        result = ocr.extract(
            io.BytesIO(content),
            mime=file.content_type,
        )
    except Exception as e:
        logger.exception("OCR processing failed")
        raise HTTPException(500, f"OCR error: {str(e)}")

    # Check for error
    if result.get("error") and not result.get("text"):
        raise HTTPException(503, result["error"])

    raw_text = result.get("text", "").strip()

    return OCRResponse(
        raw_text=raw_text,
        confidence=result.get("confidence", 0.0),
        detected_language=result.get("language", "unknown"),
        word_count=len(raw_text.split()) if raw_text else 0,
        engine_used=result.get("engine", "unknown"),
        pages=result.get("pages", 1),
    )