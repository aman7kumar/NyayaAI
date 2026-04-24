"""
backend/models/ocr_module.py
==============================
OCR module using Tesseract for extracting text from:
  - Handwritten FIR images
  - Scanned PDF pages
  - Digital images with Hindi/English mixed text

Requirements:
  sudo apt-get install tesseract-ocr tesseract-ocr-hin
  pip install pytesseract Pillow PyMuPDF
"""



from __future__ import annotations

import io
import logging
import pytesseract
from pathlib import Path
from typing import Union

from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCRModule:
    """
    Tesseract OCR wrapper with preprocessing for noisy FIR images.
    Supports: English, Hindi, mixed (eng+hin)
    """

    # Tesseract language string
    LANG_MAP = {
        "en": "eng",
        "hi": "hin",
        "mixed": "eng+hin",
    }

    def __init__(self):
            try:
                import pytesseract

                # Set Tesseract path explicitly for Windows
                import os
                tesseract_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    r"C:\Users\ASUS\AppData\Local\Tesseract-OCR\tesseract.exe",
                ]
                for path in tesseract_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        logger.info(f"Tesseract found at: {path}")
                        break

                # Verify it works
                version = pytesseract.get_tesseract_version()
                self._tess = pytesseract
                logger.info(f"✅  Tesseract OCR initialized. Version: {version}")
            except Exception as e:
                logger.error(f"Tesseract init failed: {e}")
                self._tess = None

    # ── Image Preprocessing ───────────────────────────────────────────────────

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        import numpy as np
        import cv2

        # Convert PIL to numpy
        img_array = np.array(img.convert("RGB"))

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Resize if too small
        h, w = gray.shape
        if w < 1500:
            scale = 1500 / w
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)

        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, h=10)

        # Adaptive thresholding (better than simple binarize)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 2
        )

        # Deskew
        coords = np.column_stack(np.where(thresh < 128))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.5:
                (h2, w2) = thresh.shape
                M = cv2.getRotationMatrix2D((w2 // 2, h2 // 2), angle, 1.0)
                thresh = cv2.warpAffine(thresh, M, (w2, h2),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)

        return Image.fromarray(thresh)

    # ── PDF Handling ──────────────────────────────────────────────────────────

    def _pdf_to_images(self, data: io.BytesIO) -> list[Image.Image]:
        """Convert PDF pages to PIL images using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=data.read(), filetype="pdf")
            images = []
            for page in doc:
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            return images
        except Exception as e:
            logger.exception("PDF to image conversion failed")
            return []

    # ── Main Extraction ───────────────────────────────────────────────────────

    def extract(
        self,
        source: Union[io.BytesIO, Path, str],
        mime: str = "image/jpeg",
        lang: str = "mixed",
    ) -> dict:
        """
        Extract text from image or PDF.

        Returns:
            {
                "text": str,
                "confidence": float,
                "language": str,
                "pages": int,
            }
        """
        if self._tess is None:
            return {
                "text": "[OCR unavailable — Tesseract not installed]",
                "confidence": 0.0,
                "language": "unknown",
                "pages": 0,
            }

        tess_lang = self.LANG_MAP.get(lang, "eng+hin")
        config = f"--oem 3 --psm 6 -l {tess_lang}"

        texts = []
        confidences = []

        if mime == "application/pdf":
            if isinstance(source, (str, Path)):
                with open(source, "rb") as f:
                    source = io.BytesIO(f.read())
            images = self._pdf_to_images(source)
        else:
            if isinstance(source, (str, Path)):
                images = [Image.open(source)]
            else:
                images = [Image.open(source)]

        for img in images:
            processed = self._preprocess_image(img)
            text = self._tess.image_to_string(processed, config=config)

            # Get word-level confidence
            try:
                data = self._tess.image_to_data(
                    processed,
                    config=config,
                    output_type=self._tess.Output.DICT,
                )
                confs = [
                    int(c) for c in data["conf"]
                    if str(c).lstrip("-").isdigit() and int(c) != -1
                ]
                avg_conf = sum(confs) / len(confs) if confs else 0.0
            except Exception:
                avg_conf = 0.0

            texts.append(text)
            confidences.append(avg_conf)

        combined_text = "\n\n--- PAGE BREAK ---\n\n".join(texts).strip()
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Simple language detection on result
        detected_lang = self._detect_script(combined_text)

        return {
            "text": combined_text,
            "confidence": round(avg_confidence / 100, 3),  # normalize 0-1
            "language": detected_lang,
            "pages": len(images),
        }

    def _detect_script(self, text: str) -> str:
        """Detect if text is primarily Hindi (Devanagari) or English."""
        devanagari = sum(
            1 for ch in text
            if "\u0900" <= ch <= "\u097F"
        )
        total_alpha = sum(1 for ch in text if ch.isalpha())
        if total_alpha == 0:
            return "unknown"
        ratio = devanagari / total_alpha
        if ratio > 0.5:
            return "hi"
        elif ratio > 0.1:
            return "mixed"
        return "en"
