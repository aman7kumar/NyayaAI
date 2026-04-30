"""
backend/models/ocr_module.py
================================
OCR module using:
  Primary:  PaddleOCR  — best accuracy for printed + handwritten text
                         Supports Hindi (Devanagari) + English
  Backup:   EasyOCR    — good fallback, supports 80+ languages
  Disabled: Tesseract  — removed due to poor accuracy

Supports:
  - Handwritten FIR images
  - Scanned PDF pages
  - Hindi + English mixed text
  - Digital images

Install:
  pip install paddlepaddle paddleocr easyocr opencv-python Pillow
"""




from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import logging
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


class OCRModule:
    """
    Multi-engine OCR with PaddleOCR primary and EasyOCR backup.
    Automatically picks best available engine.
    """

    def __init__(self):
        self._paddle  = None
        self._paddle_en = None
        self._paddle_hi = None
        self._easy    = None
        self._engine  = None

        # Try PaddleOCR first
        self._init_paddle()

        # Try EasyOCR as backup
        if self._paddle_en is None or self._paddle_hi is None:
            self._init_easyocr()

        if self._engine:
            logger.info(f"✅ OCR Module ready — engine: {self._engine}")
        else:
            logger.warning(
                "⚠️ No OCR engine available. "
                "Run: pip install paddlepaddle paddleocr easyocr"
            )

    # ── Engine Initialization ─────────────────────────────────────────────────

    def _init_paddle(self):
        try:
            from paddleocr import PaddleOCR
            self._paddle_en = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                cpu_threads=4,        # ← add this
                enable_mkldnn=False,  # ← add this — disables MKL which causes the conflict
            )
            self._paddle_hi = PaddleOCR(
                use_angle_cls=True,
                lang="hi",
                cpu_threads=4,        # ← add this
                enable_mkldnn=False,  # ← add this
            )
            self._engine = "paddleocr"
            logger.info("✅ PaddleOCR initialized")
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.warning(f"PaddleOCR init failed: {e}")
    '''
    def _init_easyocr(self):
        """Initialize EasyOCR — backup engine."""
        try:
            import easyocr

            # English + Hindi reader
            self._easy = easyocr.Reader(
                ["en", "hi"],
                gpu=True,        # Set True if CUDA available
                verbose=False,
            )
            self._engine = "easyocr"
            logger.info("✅ EasyOCR initialized (English + Hindi backup)")

        except ImportError:
            logger.warning(
                "EasyOCR not installed. "
                "Run: pip install easyocr"
            )
        except Exception as e:
            logger.warning(f"EasyOCR init failed: {e}")
    '''
    def _init_easyocr(self):
        logger.warning("EasyOCR disabled to save memory")
        return
    # ── Image Preprocessing ───────────────────────────────────────────────────

    def _preprocess_image(self, img: Image.Image) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.
        Returns 3-channel RGB numpy array (required by PaddleOCR).
        """
        # Step 1: Always convert to RGB first
        img = img.convert("RGB")

        # Step 2: Resize if too small
        w, h = img.size
        if w < 1200:
            scale = 1200 / w
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Step 3: Mild enhancements only
        img = ImageEnhance.Contrast(img).enhance(1.3)
        img = ImageEnhance.Brightness(img).enhance(1.1)
        img = ImageEnhance.Sharpness(img).enhance(1.5)
        img = img.filter(ImageFilter.MedianFilter(size=3))

        # Step 4: Convert to numpy — shape is (H, W, 3) here
        img_array = np.array(img)

        try:
            import cv2

            # Ensure it's 3-channel before doing anything
            if img_array.ndim == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

            bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Deskew
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            coords = np.column_stack(np.where(gray > 50))
            if len(coords) > 100:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = 90 + angle
                if abs(angle) > 0.5:
                    rows, cols = gray.shape
                    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1.0)
                    bgr = cv2.warpAffine(
                        bgr, M, (cols, rows),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE,
                    )

            # Adaptive threshold on fresh gray conversion
            gray2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            # Ensure uint8
            if gray2.dtype != np.uint8:
                gray2 = gray2.astype(np.uint8)

            thresh = cv2.adaptiveThreshold(
                gray2, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )

            # ✅ CRITICAL: Convert back to 3-channel RGB for PaddleOCR
            img_array = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

            logger.info(f"Preprocessed image shape: {img_array.shape}, dtype: {img_array.dtype}")

        except Exception as e:
            logger.warning(f"CV2 preprocessing failed: {e}, using PIL array")
            # Fallback: return plain PIL numpy array (still valid for PaddleOCR)
            img_array = np.array(img.convert("RGB"))

        return img_array

    

    # ── PDF Handling ──────────────────────────────────────────────────────────

    def _pdf_to_images(self, data: io.BytesIO) -> list[Image.Image]:
        """Convert PDF pages to PIL images using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            doc    = fitz.open(stream=data.read(), filetype="pdf")
            images = []
            for page in doc:
                # 2x zoom for better quality
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes(
                    "RGB", [pix.width, pix.height], pix.samples
                )
                images.append(img)
            return images
        except ImportError:
            logger.warning("PyMuPDF not installed. Run: pip install PyMuPDF")
            return []
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return []

    # ── PaddleOCR Extraction ──────────────────────────────────────────────────

    def _extract_with_paddle(self, img_array: np.ndarray, language: str = "mixed") -> tuple[str, float]:
        try:
            # Safety check — PaddleOCR needs 3-channel uint8
            if img_array.ndim != 3 or img_array.shape[2] != 3:
                logger.error(f"Bad input shape for PaddleOCR: {img_array.shape}")
                return "", 0.0
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)

            logger.info(f"PaddleOCR input — shape: {img_array.shape}, dtype: {img_array.dtype}")

            if language == "hi":
                results = self._paddle_hi.ocr(img_array, cls=True)
            else:
                results = self._paddle_en.ocr(img_array, cls=True)
                print("RAW OCR RESULT:", results)

            texts = []
            scores = []

            if results and results[0]:
                for line in results[0]:
                    try:
                        text = str(line[1][0]).strip()
                        score = float(line[1][1])
                        if text and score > 0.1:
                            texts.append(text)
                            scores.append(score)
                    except:
                        continue

            combined_text = " ".join(texts)
            avg_conf = sum(scores) / len(scores) if scores else 0.0
            return combined_text, avg_conf

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.warning(f"PaddleOCR extraction failed: {e}")
            return "", 0.0

    # ── EasyOCR Extraction ────────────────────────────────────────────────────

    def _extract_with_easyocr(
        self,
        img_array: np.ndarray,
    ) -> tuple[str, float]:
        """
        Extract text using EasyOCR backup.
        Returns (text, confidence) tuple.
        """
        try:
            results = self._easy.readtext(
                img_array,
                detail=1,           # Get confidence scores
                paragraph=True,     # Group into paragraphs
                width_ths=0.7,
                height_ths=0.7,
            )

            texts      = []
            confidences = []

            for result in results:
                if len(result) >= 3:
                    text       = str(result[1]).strip()
                    confidence = float(result[2])
                    if text and confidence > 0.3:
                        texts.append(text)
                        confidences.append(confidence)

            combined_text  = " ".join(texts)
            avg_confidence = (
                sum(confidences) / len(confidences)
                if confidences else 0.0
            )

            return combined_text, avg_confidence

        except Exception as e:
            logger.warning(f"EasyOCR extraction failed: {e}")
            return "", 0.0

    # ── Language Detection ────────────────────────────────────────────────────

    def _detect_language(self, text: str) -> str:
        """Detect if text is Hindi, English, or mixed."""
        if not text:
            return "unknown"

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

    # ── Main Extraction Method ────────────────────────────────────────────────

    def extract(
        self,
        source: Union[io.BytesIO, Path, str],
        mime:   str = "image/jpeg",
        lang:   str = "mixed",
    ) -> dict:
        """
        Extract text from image or PDF.

        Args:
            source: File path, BytesIO, or string path
            mime:   MIME type of the file
            lang:   Language hint ('en', 'hi', 'mixed')

        Returns:
            {
                "text":      str,    — extracted text
                "confidence": float, — 0.0 to 1.0
                "language":  str,    — detected language
                "pages":     int,    — number of pages processed
                "engine":    str,    — which OCR engine was used
            }
        """

        print("OCR ENGINE:", self._engine)
        # No engine available
        if self._engine is None:
            return {
                "text":       "",
                "confidence": 0.0,
                "language":   "unknown",
                "pages":      0,
                "engine":     "none",
                "error":      (
                    "No OCR engine available. "
                    "Run: pip install paddlepaddle paddleocr easyocr"
                ),
            }

        # Load images
        if mime == "application/pdf":
            if isinstance(source, (str, Path)):
                with open(source, "rb") as f:
                    source = io.BytesIO(f.read())
            images = self._pdf_to_images(source)
        else:
            if isinstance(source, (str, Path)):
                images = [Image.open(source)]
            elif isinstance(source, io.BytesIO):
                images = [Image.open(source)]
            else:
                images = [source]

        if not images:
            return {
                "text":       "",
                "confidence": 0.0,
                "language":   "unknown",
                "pages":      0,
                "engine":     self._engine,
            }

        # Process each page
        all_texts       = []
        all_confidences = []

        for img in images:
            # Preprocess
            img_array = self._preprocess_image(img)

            # Extract with primary or backup engine
            if (
                self._engine == "paddleocr"
                and self._paddle_en is not None
            ):
                text, confidence = self._extract_with_paddle(img_array, lang)
            elif self._engine == "easyocr" and self._easy is not None:
                text, confidence = self._extract_with_easyocr(img_array)
            else:
                text, confidence = "", 0.0

            if text.strip():
                all_texts.append(text)
                all_confidences.append(confidence)

        # Combine results
        combined_text = "\n\n".join(all_texts).strip()
        avg_confidence = (
            sum(all_confidences) / len(all_confidences)
            if all_confidences else 0.0
        )
        detected_lang = self._detect_language(combined_text)

        return {
            "text":       combined_text,
            "confidence": round(avg_confidence, 3),
            "language":   detected_lang,
            "pages":      len(images),
            "engine":     self._engine,
        }