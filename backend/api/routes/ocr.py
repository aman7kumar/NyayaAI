"""
backend/api/routes/ocr.py
OCR endpoint using PaddleOCR (primary) or EasyOCR (backup).

Changes vs original:
  - Passes byte_size to ocr.extract() so small-image upscaling triggers correctly.
  - Adds ocr_status field: "extracted" | "empty" | "error"
  - Never raises 503 for empty text — returns 200 with ocr_status="empty"
    so the frontend can show a helpful "no text found" message instead of
    a hard error that blocks the user from typing manually.
"""

import io
import logging

from fastapi import APIRouter, File, Request, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Literal

logger = logging.getLogger(__name__)
router = APIRouter()


class OCRResponse(BaseModel):
    raw_text:          str
    confidence:        float
    detected_language: str
    word_count:        int
    engine_used:       str
    pages:             int
    # NEW: lets the frontend distinguish "ran fine but found nothing" from
    # "actually failed" without parsing the text or checking word_count.
    ocr_status: Literal["extracted", "empty", "error"] = "extracted"


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

    Returns ocr_status:
      "extracted" — text was found
      "empty"     — image processed OK but no text detected (low quality / blank)
      "error"     — OCR engine failed (not a quality problem)
    """
    state = request.app.extra.get("app_state", {})
    ocr   = state.get("ocr_module")

    if not ocr:
        raise HTTPException(503, "OCR module not loaded.")

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
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(413, "File too large. Maximum 15MB.")

    if not content:
        raise HTTPException(400, "Empty file uploaded.")

    try:
        result = ocr.extract(
            io.BytesIO(content),
            mime=file.content_type,
            byte_size=len(content),      # NEW: enables small-image upscaling
        )
    except Exception as e:
        logger.exception("OCR processing failed")
        raise HTTPException(500, f"OCR error: {str(e)}")

    # Distinguish engine failure from "image processed but no text found"
    has_error = bool(result.get("error"))
    raw_text  = result.get("text", "").strip()

    if has_error and not raw_text:
        # Actual engine failure — raise so the frontend shows an error toast
        raise HTTPException(503, result["error"])

    # Determine status — never raise for empty text, just signal it
    if raw_text:
        ocr_status = "extracted"
    elif has_error:
        ocr_status = "error"    # partial failure but engine returned something
    else:
        ocr_status = "empty"    # engine ran fine, image just has no readable text

    return OCRResponse(
        raw_text=raw_text,
        confidence=result.get("confidence", 0.0),
        detected_language=result.get("language", "unknown"),
        word_count=len(raw_text.split()) if raw_text else 0,
        engine_used=result.get("engine", "unknown"),
        pages=result.get("pages", 1),
        ocr_status=ocr_status,
    )