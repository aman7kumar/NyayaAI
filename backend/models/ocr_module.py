"""
backend/models/ocr_module.py
================================
OCR module using:
  Primary:  PaddleOCR  — best accuracy for printed + handwritten text
                         Supports Hindi (Devanagari) + English
  Backup:   EasyOCR    — good fallback, supports 80+ languages

Supports:
  - Handwritten FIR images
  - Scanned PDF pages
  - Hindi + English mixed text
  - Digital images

Install:
  pip install paddlepaddle paddleocr easyocr opencv-python Pillow

BUG FIXES (v3):
  1–6: All prior fixes retained (env vars, contiguous arrays, PIL close,
       low-mem reset, threading lock, Paddle re-init).
  7.  _is_small_image now ALSO triggers on pixel dimensions alone — the
      byte_size=0 default (when the route forgets to pass it) no longer
      silently disables the aggressive-upscale pass.
  8.  alpha-ratio post-filter lowered 0.30 → 0.20 and only applied to
      tokens longer than 5 chars, so short valid OCR tokens aren't dropped.
  9.  _quality_score readability threshold lowered 0.58 → 0.45 so the
      second model is tried less eagerly (saves time) while still falling
      back when needed.
  10. "best of all passes" logic now falls back to the LONGEST non-empty
      result when every pass scores below the quality threshold — guarantees
      at least some text is returned rather than an empty string.
  11. _extract_internal returns the raw best result even when
      best_score == 0.0, as long as text is non-empty. The frontend
      "OCR failed" toast was firing because `text` was "".
  12. Soft-preprocess pass now also runs on the aggressively-upscaled
      array (not only the original), giving Paddle a clean high-res input.
"""

from __future__ import annotations

import os

# Safety-net env vars (effective only if not already loaded by run.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK",   "TRUE")
os.environ.setdefault("OMP_NUM_THREADS",         "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",    "1")
os.environ.setdefault("MKL_NUM_THREADS",         "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS",  "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",     "1")
os.environ.setdefault("FLAGS_use_mkldnn",        "0")

import gc
import io
import logging
import re
import threading
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SIDE = 1536


def _ocr_max_side() -> int:
    try:
        v = int(os.getenv("OCR_MAX_IMAGE_SIDE", str(_DEFAULT_MAX_SIDE)))
        return max(640, min(v, 8192))
    except ValueError:
        return _DEFAULT_MAX_SIDE


def _resize_image_max_side(img: Image.Image, max_side: int | None = None) -> Image.Image:
    """Shrink so longest edge <= max_side (no upscaling)."""
    if max_side is None:
        max_side = _ocr_max_side()
    img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    s = max_side / longest
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    return img.resize((nw, nh), Image.LANCZOS)


def _upscale_small_for_ocr(w: int, h: int, max_side: int) -> tuple[int, int]:
    cap_w = min(1400, max_side)
    if w >= 900 or w >= cap_w:
        return w, h
    s = cap_w / float(w)
    nw, nh = int(round(w * s)), int(round(h * s))
    if max(nw, nh) > max_side:
        s2 = max_side / float(max(nw, nh))
        nw = max(1, int(round(nw * s2)))
        nh = max(1, int(round(nh * s2)))
    return nw, nh


def _looks_like_oom(err: Exception | str) -> bool:
    msg = str(err).lower()
    return (
        "unable to allocate" in msg
        or "insufficient memory" in msg
        or "outofmemoryerror" in msg
        or "bad allocation" in msg
    )


def _looks_like_native_crash(err: Exception) -> bool:
    """Detect errors that suggest Paddle's C++ layer is corrupted."""
    msg = str(err).lower()
    return (
        "segmentation fault" in msg
        or "access violation" in msg
        or "enforce_notok" in msg
        or "paddlepaddle" in msg
        or "enforce failed" in msg
        or "external_error" in msg
        or isinstance(err, (SystemError, RuntimeError))
        and "paddle" in msg
    )


def _silence_paddle_loggers() -> None:
    import logging as _log
    _targets = [
        "ppocr", "ppstructure",
        "paddle", "paddle.fluid", "paddle.fluid.core",
        "paddleocr",
    ]
    for name in _targets:
        lg = _log.getLogger(name)
        lg.setLevel(_log.ERROR)
        for h in lg.handlers[:]:
            lg.removeHandler(h)
        if not any(isinstance(h, _log.NullHandler) for h in lg.handlers):
            lg.addHandler(_log.NullHandler())
        lg.propagate = False


def _safe_array(arr: np.ndarray) -> np.ndarray:
    """
    FIX #2: Force a full memory copy and ensure C-contiguous layout.
    This severs any reference Paddle's C++ predictor might hold to the
    original Python-managed buffer after the call returns.
    """
    return np.ascontiguousarray(arr.copy())


def _is_small_image(img: Image.Image, byte_size: int = 0) -> bool:
    """
    FIX #7: True when the image is likely a compressed/low-res photo that
    needs aggressive upscaling before OCR.

    OLD logic relied on byte_size being passed from the route. When the
    route passes byte_size=0 (default), the byte-size branch was always
    False, silently disabling the upscale pass for files < 300 KB.

    NEW logic: trigger on EITHER condition independently:
      - shorter pixel dimension < 800 px  (was 600, raised for better coverage)
      - file byte size > 0 AND < 400 KB   (raised from 300 KB)
    This means the upscale pass fires correctly even when byte_size is not
    provided by the caller.
    """
    w, h = img.size
    short_side = min(w, h)
    pixel_trigger = short_side < 800
    size_trigger  = 0 < byte_size < 400 * 1024
    return pixel_trigger or size_trigger


def _upscale_for_ocr_aggressive(img: Image.Image, target_short: int = 1200) -> Image.Image:
    """
    FIX #7: Scale up so the shorter side reaches *target_short* pixels.
    Raised default from 900 → 1200 for better text detection on real-world
    compressed FIR scans.  Never upscales more than 4× to avoid blur.
    """
    w, h   = img.size
    short  = min(w, h)
    if short >= target_short:
        return img
    scale = min(target_short / short, 4.0)
    nw    = max(1, int(round(w * scale)))
    nh    = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.LANCZOS)


class OCRModule:
    """
    Multi-engine OCR with PaddleOCR primary and EasyOCR backup.

    Thread-safety: a per-instance Lock serialises all Paddle calls.
    Paddle's internal C++ inference engine is NOT thread-safe — concurrent
    calls corrupt its graph execution state.
    """

    def __init__(self):
        self._paddle_en    = None
        self._paddle_hi    = None
        self._easy         = None
        self._engine       = None
        self._low_mem_mode = False
        # FIX #5: serialise all Paddle calls to prevent concurrent corruption
        self._paddle_lock  = threading.Lock()

        self._init_paddle()

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

    def _init_paddle(self) -> bool:
        """Initialize (or re-initialize) PaddleOCR. Returns True on success."""
        try:
            _silence_paddle_loggers()
            from paddleocr import PaddleOCR
            _silence_paddle_loggers()

            self._paddle_en = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                cpu_threads=1,
                enable_mkldnn=False,
                show_log=False,
            )
            _silence_paddle_loggers()

            self._paddle_hi = PaddleOCR(
                use_angle_cls=True,
                lang="hi",
                cpu_threads=1,
                enable_mkldnn=False,
                show_log=False,
            )
            _silence_paddle_loggers()

            self._engine = "paddleocr"
            logger.info("✅ PaddleOCR initialized")
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.warning(f"PaddleOCR init failed: {e}")
            self._paddle_en = None
            self._paddle_hi = None
            return False

    def _reinit_paddle(self) -> bool:
        """
        FIX #6: Re-initialize Paddle after a native crash.
        Drops old instances first so their C++ destructors run.
        """
        logger.warning("🔄 Re-initializing PaddleOCR after crash...")
        self._paddle_en = None
        self._paddle_hi = None
        gc.collect()
        return self._init_paddle()

    def _init_easyocr(self):
        try:
            enabled = os.getenv("OCR_ENABLE_EASYOCR", "true").lower() in ("1", "true", "yes")
            if not enabled:
                logger.warning("EasyOCR disabled by OCR_ENABLE_EASYOCR")
                return
            import easyocr
            self._easy = easyocr.Reader(["en", "hi"], gpu=False, verbose=False)
            if not self._engine:
                self._engine = "easyocr"
            logger.info("✅ EasyOCR initialized as handwriting fallback")
        except Exception as e:
            logger.warning(f"EasyOCR init failed: {e}")

    # ── Image Preprocessing ───────────────────────────────────────────────────

    def _preprocess_image(self, img: Image.Image) -> np.ndarray:
        try:
            import cv2

            max_side = _ocr_max_side()
            img = _resize_image_max_side(img, max_side)
            img_rgb = np.array(img)

            h, w = img_rgb.shape[:2]
            new_w, new_h = _upscale_small_for_ocr(w, h, max_side)
            if (new_w, new_h) != (w, h) and new_w * new_h * 3 < 48_000_000:
                img_rgb = cv2.resize(
                    img_rgb, (new_w, new_h),
                    interpolation=cv2.INTER_CUBIC,
                )

            lab     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l       = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
            img_rgb = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
            del lab, l, a, b

            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            del img_rgb
            gray = cv2.fastNlMeansDenoising(
                gray, h=10, templateWindowSize=7, searchWindowSize=21,
            )

            try:
                coords = np.column_stack(np.where(gray < 200))
                if len(coords) > 100:
                    angle = cv2.minAreaRect(coords)[-1]
                    if angle < -45:
                        angle = 90 + angle
                    if abs(angle) > 0.5:
                        h2, w2 = gray.shape
                        M    = cv2.getRotationMatrix2D((w2 // 2, h2 // 2), angle, 1.0)
                        gray = cv2.warpAffine(
                            gray, M, (w2, h2),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE,
                        )
            except Exception:
                pass

            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=25, C=10,
            )
            del gray
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.filter2D(
                binary, -1,
                np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            )
            result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            del binary
            return result

        except ImportError:
            return self._preprocess_image_pil(img)
        except Exception as e:
            if _looks_like_oom(e):
                self._low_mem_mode = True
            logger.warning(f"Advanced preprocessing failed: {e}. Using PIL fallback.")
            return self._preprocess_image_pil(img)

    def _preprocess_image_soft(self, img: Image.Image) -> np.ndarray:
        try:
            import cv2
            arr  = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            del arr
            gray = cv2.bilateralFilter(gray, d=7, sigmaColor=45, sigmaSpace=45)
            eq   = cv2.equalizeHist(gray)
            del gray
            rgb  = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
            del eq
            return rgb
        except Exception:
            return np.array(img.convert("RGB"))

    def _preprocess_image_pil(self, img: Image.Image) -> np.ndarray:
        max_side = _ocr_max_side()
        img      = _resize_image_max_side(img, max_side)
        w, h     = img.size
        new_w, new_h = _upscale_small_for_ocr(w, h, max_side)
        if (new_w, new_h) != (w, h) and new_w * new_h * 3 < 30_000_000:
            img = img.resize((new_w, new_h), Image.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = ImageEnhance.Sharpness(img).enhance(2.5)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        return np.array(img)

    # ── PDF Handling ──────────────────────────────────────────────────────────

    def _pdf_to_images(self, data: io.BytesIO) -> list[Image.Image]:
        try:
            import fitz
            doc      = fitz.open(stream=data.read(), filetype="pdf")
            max_side = _ocr_max_side()
            images   = []
            for page in doc:
                w_pt, h_pt = page.rect.width, page.rect.height
                if w_pt < 1 or h_pt < 1:
                    continue
                longest_pt = max(w_pt, h_pt)
                zoom = min(2.0, (max_side * 0.92) / longest_pt)
                zoom = max(zoom, 0.72)
                mat  = fitz.Matrix(zoom, zoom)
                pix  = page.get_pixmap(matrix=mat, alpha=False)
                im   = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(_resize_image_max_side(im, max_side))
            return images
        except ImportError:
            logger.warning("PyMuPDF not installed. Run: pip install PyMuPDF")
            return []
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return []

    # ── PaddleOCR Extraction ──────────────────────────────────────────────────

    def _extract_with_paddle(
        self, img_array: np.ndarray, language: str = "mixed",
    ) -> tuple[str, float]:
        """
        FIX #2 + #5 + #8 + #9:
          - Force-copy the array (FIX #2)
          - Hold the lock for the entire Paddle call (FIX #5)
          - Relaxed alpha-ratio post-filter (FIX #8)
          - Lowered quality threshold for model switching (FIX #9)
        """
        try:
            if not self._paddle_en or not self._paddle_hi:
                return "", 0.0

            safe_arr = _safe_array(img_array)

            def _quality_score(texts: list[str], scores: list[float]) -> float:
                if not texts or not scores:
                    return 0.0
                text = " ".join(texts)
                alpha = sum(c.isalpha() for c in text)
                if alpha == 0:
                    return 0.0
                latin      = sum(c.isascii() and c.isalpha() for c in text)
                devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097F")
                readability = max(latin, devanagari) / alpha
                return (sum(scores) / len(scores)) * readability

            if language == "hi":
                ordered_models = [("Hindi", self._paddle_hi), ("English", self._paddle_en)]
            elif language == "en":
                ordered_models = [("English", self._paddle_en), ("Hindi", self._paddle_hi)]
            else:
                ordered_models = [("English", self._paddle_en), ("Hindi", self._paddle_hi)]

            selected_texts:  list[str]   = []
            selected_scores: list[float] = []
            best_q = 0.0

            with self._paddle_lock:
                for label, model in ordered_models:
                    texts, scores = [], []
                    try:
                        raw = model.ocr(safe_arr, cls=True)
                        lines = []
                        if raw and raw[0]:
                            lines = raw[0]
                        for line in lines:
                            if not line or len(line) < 2 or not line[1] or len(line[1]) < 2:
                                continue
                            t, s = str(line[1][0]).strip(), float(line[1][1])
                            if not t or s < 0.25 or len(t) < 1:
                                continue
                            # FIX #8: relaxed alpha-ratio filter.
                            # Old: any token > 3 chars with alpha < 0.30 was dropped.
                            # New: only reject tokens > 5 chars with alpha < 0.20.
                            # This keeps short mixed tokens (e.g. "i2", "FIR-123").
                            if len(t) > 5:
                                alpha_count = sum(c.isalpha() for c in t)
                                if alpha_count / len(t) < 0.20:
                                    continue
                            texts.append(t)
                            scores.append(s)
                    except Exception as e:
                        if _looks_like_oom(e):
                            self._low_mem_mode = True
                        logger.warning(f"PaddleOCR {label} extraction error: {e}")
                        continue

                    q = _quality_score(texts, scores)
                    if q > best_q:
                        best_q          = q
                        selected_texts  = texts
                        selected_scores = scores
                    # FIX #9: lowered early-exit threshold 0.58 → 0.45
                    if q >= 0.45:
                        break

            del safe_arr

            combined = self._clean_ocr_output("\n".join(selected_texts))
            avg_conf = sum(selected_scores) / len(selected_scores) if selected_scores else 0.0
            return combined, avg_conf

        except Exception as e:
            if _looks_like_oom(e):
                self._low_mem_mode = True
            logger.error(f"PaddleOCR extraction completely failed: {e}")
            return "", 0.0

    # ── EasyOCR Extraction ────────────────────────────────────────────────────

    def _extract_with_easyocr(self, img_array: np.ndarray) -> tuple[str, float]:
        try:
            safe_arr = _safe_array(img_array)
            results  = self._easy.readtext(
                safe_arr, detail=1, paragraph=True,
                width_ths=0.7, height_ths=0.7,
            )
            del safe_arr
            texts, confs = [], []
            for r in results:
                if len(r) >= 3 and str(r[1]).strip() and float(r[2]) > 0.25:
                    texts.append(str(r[1]).strip())
                    confs.append(float(r[2]))
            return (
                self._clean_ocr_output(" ".join(texts)),
                sum(confs) / len(confs) if confs else 0.0,
            )
        except Exception as e:
            if _looks_like_oom(e):
                self._low_mem_mode = True
            logger.warning(f"EasyOCR extraction failed: {e}")
            return "", 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _text_quality_score(self, text: str, confidence: float) -> float:
        if not text:
            return 0.0
        tokens = [t for t in re.split(r"\s+", text.strip()) if t]
        if not tokens:
            return 0.0
        alpha_chars = sum(ch.isalpha() for ch in text)
        total_chars = max(len(text), 1)
        alpha_ratio = alpha_chars / total_chars
        good_tokens = 0
        for t in tokens:
            letters = sum(ch.isalpha() for ch in t)
            if len(t) >= 2 and (letters / max(len(t), 1)) >= 0.55:   # relaxed 0.65→0.55
                good_tokens += 1
        token_ratio   = good_tokens / len(tokens)
        weird         = len(re.findall(r"[^\w\s.,:;()'\"/-]", text))
        weird_penalty = min(0.25, weird / max(len(tokens), 1) * 0.05)
        score = (0.45 * confidence) + (0.30 * alpha_ratio) + (0.25 * token_ratio) - weird_penalty
        return max(score, 0.0)

    def _detect_language(self, text: str) -> str:
        if not text:
            return "unknown"
        dev   = sum(1 for c in text if "\u0900" <= c <= "\u097F")
        alpha = sum(1 for c in text if c.isalpha())
        if alpha == 0:
            return "unknown"
        r = dev / alpha
        return "hi" if r > 0.5 else "mixed" if r > 0.1 else "en"

    def _clean_ocr_output(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"([A-Za-z])\1{3,}", r"\1\1", text)
        text = re.sub(r"[^\S\n]+([.,:;])", r"\1", text)
        text = re.sub(r"\b([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\b", r"\1\2\3", text)
        lines = [ln.strip() for ln in text.split("\n")]
        lines = [ln for ln in lines if len(ln) >= 2]   # relaxed 3→2
        return "\n".join(lines).strip()

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def extract(self, source, mime="image/jpeg", lang="mixed", byte_size: int = 0) -> dict:
        """
        Public entry point. Wraps _extract_internal with:
          - EasyOCR emergency fallback
          - Paddle re-init on native crashes (FIX #6)
        byte_size: original file size in bytes — used for small-image detection.
                   Passing 0 is safe; pixel-dimension check handles it (FIX #7).
        """
        try:
            return self._extract_internal(source, mime, lang, byte_size)
        except Exception as e:
            logger.error(f"Primary OCR failed: {e}")

            if _looks_like_native_crash(e) and self._paddle_en:
                logger.warning("Paddle crash detected — attempting re-init")
                if self._reinit_paddle():
                    try:
                        if isinstance(source, io.BytesIO):
                            source.seek(0)
                        return self._extract_internal(source, mime, lang, byte_size)
                    except Exception as e2:
                        logger.error(f"Retry after Paddle re-init also failed: {e2}")

            try:
                if self._easy and mime != "application/pdf":
                    if isinstance(source, io.BytesIO):
                        source.seek(0)
                    im   = _resize_image_max_side(Image.open(source).convert("RGB"))
                    arr  = _safe_array(np.array(im))
                    im.close()
                    text, conf = self._extract_with_easyocr(arr)
                    del arr
                    return {
                        "text": text, "confidence": conf,
                        "language": self._detect_language(text),
                        "pages": 1, "engine": "easyocr_emergency",
                    }
            except Exception as e2:
                logger.error(f"Emergency EasyOCR fallback also failed: {e2}")

            return {
                "text": "", "confidence": 0.0,
                "language": "unknown", "pages": 0,
                "engine": "failed", "error": str(e),
            }

    def _extract_internal(
        self,
        source:    Union[io.BytesIO, Path, str],
        mime:      str = "image/jpeg",
        lang:      str = "mixed",
        byte_size: int = 0,
    ) -> dict:
        if self._engine is None:
            return {
                "text": "", "confidence": 0.0, "language": "unknown",
                "pages": 0, "engine": "none",
                "error": "No OCR engine available. Run: pip install paddlepaddle paddleocr",
            }

        if mime == "application/pdf":
            if isinstance(source, (str, Path)):
                with open(source, "rb") as f:
                    source = io.BytesIO(f.read())
            images = self._pdf_to_images(source)
        else:
            if isinstance(source, io.BytesIO):
                source.seek(0)
            raw_pil = Image.open(source)
            images  = [raw_pil.convert("RGB")]
            raw_pil.close()

        if not images:
            return {
                "text": "", "confidence": 0.0, "language": "unknown",
                "pages": 0, "engine": self._engine,
            }

        all_texts:  list[str]   = []
        all_confs:  list[float] = []
        any_success = False

        for img in images:
            try:
                runtime_max_side = (
                    min(_ocr_max_side(), 1120)
                    if self._low_mem_mode
                    else _ocr_max_side()
                )
                img = _resize_image_max_side(img.convert("RGB"), runtime_max_side)

                best_text, best_conf, best_score = "", 0.0, 0.0
                # FIX #10: track longest non-empty result as ultimate fallback
                longest_fallback_text, longest_fallback_conf = "", 0.0

                def _take(t: str, c: float) -> None:
                    nonlocal best_text, best_conf, best_score
                    nonlocal longest_fallback_text, longest_fallback_conf
                    if t.strip():
                        # Always update longest-text fallback (FIX #10)
                        if len(t) > len(longest_fallback_text):
                            longest_fallback_text = t
                            longest_fallback_conf = c
                    s = self._text_quality_score(t, c)
                    if s > best_score:
                        best_text, best_conf, best_score = t, c, s

                # ── Pass 1: raw pixels ────────────────────────────────────
                arr_raw = _safe_array(np.array(img))
                if self._paddle_en:
                    _take(*self._extract_with_paddle(arr_raw, lang))
                if self._easy is not None:
                    _take(*self._extract_with_easyocr(arr_raw))
                del arr_raw

                # ── Pass 1b: aggressive upscale (FIX #7) ─────────────────
                # Now triggers on pixel dimensions alone, so byte_size=0
                # no longer silently disables this pass.
                if best_score < 0.40 and _is_small_image(img, byte_size):
                    up_img = _upscale_for_ocr_aggressive(img, target_short=1200)
                    arr_up = _safe_array(np.array(up_img))
                    if self._paddle_en:
                        _take(*self._extract_with_paddle(arr_up, lang))
                    if self._easy is not None:
                        _take(*self._extract_with_easyocr(arr_up))

                    # FIX #12: also soft-preprocess the upscaled image so
                    # Paddle gets a high-res denoised input, not just raw pixels.
                    if best_score < 0.40:
                        arr_up_soft = _safe_array(self._preprocess_image_soft(up_img))
                        if self._paddle_en:
                            _take(*self._extract_with_paddle(arr_up_soft, lang))
                        if self._easy is not None:
                            _take(*self._extract_with_easyocr(arr_up_soft))
                        del arr_up_soft

                    del arr_up, up_img

                # ── Pass 2: soft preprocessing ────────────────────────────
                if (not self._low_mem_mode) and best_score < 0.45:
                    arr_soft = _safe_array(self._preprocess_image_soft(img))
                    if self._paddle_en:
                        _take(*self._extract_with_paddle(arr_soft, lang))
                    if self._easy is not None:
                        _take(*self._extract_with_easyocr(arr_soft))
                    del arr_soft

                # ── Pass 3: full preprocessing ────────────────────────────
                if (not self._low_mem_mode) and best_score < 0.52:
                    arr_pre = _safe_array(self._preprocess_image(img))
                    if self._paddle_en:
                        _take(*self._extract_with_paddle(arr_pre, lang))
                    del arr_pre

                # FIX #11: fall back to longest non-empty result if the
                # quality scorer returned "" (which previously caused the
                # frontend "OCR failed" toast even when text was found).
                if not best_text.strip() and longest_fallback_text.strip():
                    logger.info(
                        "Quality scorer returned empty; using longest-result fallback "
                        f"({len(longest_fallback_text)} chars, conf={longest_fallback_conf:.2f})"
                    )
                    best_text = longest_fallback_text
                    best_conf = longest_fallback_conf

                text, conf = best_text, best_conf
                if text.strip():
                    all_texts.append(text)
                    all_confs.append(conf)
                    any_success = True
                else:
                    logger.warning("All OCR passes returned empty text for this page/image.")

            except Exception as page_err:
                logger.warning(f"OCR failed for one page: {page_err}")
            finally:
                try:
                    img.close()
                except Exception:
                    pass
                gc.collect()

        # FIX #4: reset low-mem mode after a successful run
        if any_success and self._low_mem_mode:
            logger.info("Successful extraction after low-mem mode — resetting flag.")
            self._low_mem_mode = False

        combined = "\n\n".join(all_texts).strip()
        return {
            "text":       combined,
            "confidence": round(sum(all_confs) / len(all_confs), 3) if all_confs else 0.0,
            "language":   self._detect_language(combined),
            "pages":      len(images),
            "engine":     self._engine,
        }