"""
backend/modules/pdf_extractor.py
==================================
Extracts structured text from legal PDF documents using PyMuPDF.
Handles:
  - IPC / CrPC / Constitution PDFs from indiacode.nic.in
  - Court judgment PDFs
  - Contract / Agreement PDFs
  
Output: List of structured chunks with metadata for FAISS indexing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extracts and structures text from Indian legal PDFs.
    """

    # Pattern for IPC/CrPC section headers like "323. Voluntarily causing hurt"
    SECTION_PATTERN = re.compile(
        r"^(\d+[A-Za-z]?)\.\s+([A-Z][^\n]{5,80})",
        re.MULTILINE
    )

    # Pattern for Constitution articles
    ARTICLE_PATTERN = re.compile(
        r"(?:Article|Art\.)\s+(\d+[A-Za-z]?)[.\—]\s*([^\n]{5,100})",
        re.IGNORECASE
    )

    def extract_from_file(self, filepath: str | Path) -> list[dict]:
        """
        Extract all text chunks from a PDF file.
        Returns list of dicts: {text, page, section, act, source}
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("PyMuPDF not installed. Run: pip install PyMuPDF")
            return []

        filepath = Path(filepath)
        act_name = self._detect_act_name(filepath.stem)

        chunks = []
        try:
            doc = fitz.open(str(filepath))
            full_text_by_page = []

            for page_num, page in enumerate(doc, 1):
                text = page.get_text("text")
                text = self._clean_text(text)
                if text.strip():
                    full_text_by_page.append((page_num, text))

            # Combine and chunk by section
            full_text = "\n".join(t for _, t in full_text_by_page)
            chunks = list(self._chunk_by_section(full_text, act_name, filepath.name))

            logger.info(f"Extracted {len(chunks)} chunks from {filepath.name}")
        except Exception as e:
            logger.exception(f"PDF extraction failed for {filepath}: {e}")

        return chunks

    def _chunk_by_section(
        self,
        text: str,
        act: str,
        source: str,
        chunk_size: int = 512,
    ) -> Iterator[dict]:
        """
        Split text into meaningful chunks at section boundaries.
        Falls back to sliding window if no section markers found.
        """
        # Try section-based chunking
        sections = self.SECTION_PATTERN.split(text)
        if len(sections) > 3:
            for i in range(1, len(sections) - 1, 3):
                sec_num   = sections[i].strip()
                sec_title = sections[i + 1].strip() if i + 1 < len(sections) else ""
                sec_text  = sections[i + 2].strip() if i + 2 < len(sections) else ""
                if len(sec_text) > 20:
                    yield {
                        "text": f"Section {sec_num}: {sec_title}. {sec_text[:chunk_size]}",
                        "section": f"{act} {sec_num}",
                        "title": sec_title,
                        "act": act,
                        "source": source,
                    }
        else:
            # Sliding window fallback
            words = text.split()
            window, stride = 150, 75
            for i in range(0, len(words), stride):
                chunk_words = words[i: i + window]
                chunk_text  = " ".join(chunk_words)
                if len(chunk_text) > 100:
                    yield {
                        "text": chunk_text,
                        "section": "",
                        "title": "",
                        "act": act,
                        "source": source,
                    }

    def _detect_act_name(self, filename: str) -> str:
        """Detect act name from filename."""
        filename_lower = filename.lower()
        if "ipc" in filename_lower or "penal" in filename_lower:
            return "IPC"
        if "crpc" in filename_lower or "criminal procedure" in filename_lower:
            return "CrPC"
        if "constitution" in filename_lower:
            return "Constitution"
        if "consumer" in filename_lower:
            return "Consumer Protection Act"
        if "dowry" in filename_lower:
            return "Dowry Prohibition Act"
        if "it act" in filename_lower or "information technology" in filename_lower:
            return "IT Act"
        return "Legal Document"

    def _clean_text(self, text: str) -> str:
        """Remove OCR artifacts and normalize whitespace."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"[^\S\n]+", " ", text)
        return text.strip()
