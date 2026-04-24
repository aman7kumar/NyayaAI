"""
backend/api/routes/ocr.py
==========================
OCR endpoint — accepts uploaded FIR image (JPG/PNG/PDF page)
and returns extracted text using Tesseract.
"""
import pytesseract
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from pydantic import BaseModel
import io

logger = logging.getLogger(__name__)
router = APIRouter()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
class OCRResponse(BaseModel):
    raw_text: str
    confidence: float
    detected_language: str
    word_count: int


@router.post(
    "/ocr",
    response_model=OCRResponse,
    summary="Extract text from FIR image or scanned document",
)
async def extract_text_from_image(
    request: Request,
    file: UploadFile = File(..., description="FIR image (JPG/PNG) or PDF"),
):
    """
    Accepts a FIR image or scanned document.
    Uses Tesseract OCR with Hindi + English language packs.
    Returns extracted text for downstream analysis.
    """
    state = request.app.extra.get("app_state", {})
    ocr: "OCRModule" = state.get("ocr_module")

    if not ocr:
        raise HTTPException(503, "OCR module not loaded.")

    # Validate file type
    allowed = {"image/jpeg", "image/png", "image/webp", "application/pdf"}
    if file.content_type not in allowed:
        raise HTTPException(
            400,
            f"Unsupported file type: {file.content_type}. Allowed: {allowed}",
        )

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(413, "File too large. Maximum 10MB.")

    try:
        result = ocr.extract(io.BytesIO(content), mime=file.content_type)
    except Exception as e:
        logger.exception("OCR failed")
        raise HTTPException(500, f"OCR processing error: {str(e)}")

    return OCRResponse(
        raw_text=result["text"],
        confidence=result["confidence"],
        detected_language=result["language"],
        word_count=len(result["text"].split()),
    )
