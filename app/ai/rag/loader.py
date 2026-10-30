"""PDF loading and text normalisation.

WHY (architecture decisions)
----------------------------
1. **Bytes in, validated model out.** The loader accepts raw bytes (what a
   FastAPI upload gives us) or a file path, and returns an immutable Pydantic
   model. Downstream stages (chunking, indexing) never touch PyMuPDF, so the
   PDF library can be swapped without changing them.

2. **Fail early, fail specifically.** User-supplied PDFs are untrusted: they
   may be empty, truncated, encrypted, huge, or scans with no text layer. Each
   case raises its own exception (see ``app/core/exceptions.py``) so the API
   can tell the user exactly what is wrong instead of returning a generic 500.

3. **Arabic needs normalisation *before* embedding.** PDFs frequently store
   Arabic as "presentation forms" (U+FB50-U+FEFF: isolated/initial/medial/final
   glyph shapes) instead of base letters, so the same word can be encoded in
   several ways. Without normalising, search for a word would miss the same
   word written with different code points. NFKC folds these into base
   letters; we also drop tatweel, bidi/invisible control characters, and
   (optionally) diacritics. The *same* function is applied to user queries in
   the retriever, so stored text and queries always live in one text space.

4. **Content-hash document IDs.** ``doc_id`` is derived from the file bytes.
   Re-uploading the same file replaces its chunks instead of duplicating them.

Known limitation: a minority of Arabic PDFs store glyphs in *visual* order
(letters reversed). PyMuPDF returns text as stored, so such files extract
backwards. There is no reliable automatic fix; re-export the source document
or run OCR on it.
"""

import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional, Union

import fitz  # PyMuPDF
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_settings
from app.core.exceptions import (
    CorruptPDFError,
    EmptyDocumentError,
    EncryptedPDFError,
    PDFLoadError,
    PDFTooLargeError,
)

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF-"
# The PDF spec allows a little junk before the header, so search a small window.
_MAGIC_SEARCH_WINDOW = 1024
_DEFAULT_SOURCE_NAME = "document.pdf"

# Zero-width / bidi control characters and soft hyphen: invisible to readers but
# they split words for the tokenizer. Tatweel (U+0640) is purely decorative
# stretching ("العـــربية"), so it is removed too.
_INVISIBLE_RE = re.compile(
    "[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff\u00ad\u0640]"
)
# Arabic harakat: U+0610-061A, U+064B-065F and superscript alef U+0670.
_DIACRITICS_RE = re.compile("[\u0610-\u061a\u064b-\u065f\u0670]")
# English word hyphenated across a line break: "exam-\nple" -> "example".
_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z])-\n([a-z])")
# Horizontal whitespace except newline (NBSP is already a space after NFKC).
_HSPACE_RE = re.compile(r"[^\S\n]+")
_SPACE_AROUND_NEWLINE_RE = re.compile(r" *\n *")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


class PageContent(BaseModel):
    """Cleaned text of one non-empty PDF page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1, description="1-based page number.")
    text: str = Field(min_length=1)


class LoadedDocument(BaseModel):
    """Result of loading a PDF: cleaned pages plus provenance information."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(description="Stable ID derived from the file content (SHA-256).")
    source_name: str
    page_count: int = Field(ge=1)
    pages: list[PageContent]
    pdf_metadata: dict[str, str] = Field(default_factory=dict)
    skipped_pages: list[int] = Field(
        default_factory=list, description="Pages that raised errors while extracting."
    )
    image_only_pages: list[int] = Field(
        default_factory=list, description="Pages with images but no text (likely scans)."
    )

    @property
    def char_count(self) -> int:
        return sum(len(page.text) for page in self.pages)


def normalize_text(text: str, strip_diacritics: bool = True) -> str:
    """Normalise text so Arabic/English content is consistent and embeddable.

    Applied to both documents and user queries (see module docstring, point 3).
    Newlines are kept (only blank-line runs are collapsed) because paragraph
    breaks are the splitter's preferred cut points.

    Note: NFKC also maps ligatures ("ﬁ" -> "fi") and super/subscripts
    ("m²" -> "m2"), which is desirable for search but means the stored text
    is not a byte-exact copy of the PDF.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = _INVISIBLE_RE.sub("", text)
    if strip_diacritics:
        text = _DIACRITICS_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _HSPACE_RE.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE_RE.sub("\n", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _read_source(
    source: Union[str, Path, bytes, bytearray],
    source_name: Optional[str],
    max_size_bytes: int,
) -> tuple[bytes, str]:
    """Return ``(file_bytes, display_name)`` after cheap size/existence checks.

    Checking the size of a path *before* reading avoids loading a multi-GB
    file into memory just to reject it.
    """
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        name = source_name or _DEFAULT_SOURCE_NAME
    else:
        path = Path(source)
        if not path.is_file():
            raise PDFLoadError(f"PDF file not found: {path}")
        try:
            size = path.stat().st_size
            if size > max_size_bytes:
                raise PDFTooLargeError(
                    f"File is {size / 1_048_576:.1f} MB; the limit is "
                    f"{max_size_bytes / 1_048_576:.0f} MB."
                )
            data = path.read_bytes() if size else b""
        except OSError as exc:
            raise PDFLoadError(f"Cannot read PDF file '{path}': {exc}") from exc
        name = source_name or path.name

    if not data:
        raise CorruptPDFError("The file is empty (0 bytes).")
    if len(data) > max_size_bytes:
        raise PDFTooLargeError(
            f"File is {len(data) / 1_048_576:.1f} MB; the limit is "
            f"{max_size_bytes / 1_048_576:.0f} MB."
        )
    # Path(...).name drops any directory part, so a hostile upload name such as
    # "../../etc/passwd" can never leak into metadata as a path.
    return data, Path(name).name or _DEFAULT_SOURCE_NAME


def _open_document(data: bytes, name: str) -> "fitz.Document":
    """Open PDF bytes with PyMuPDF, translating its errors to ours."""
    if data[:_MAGIC_SEARCH_WINDOW].find(_PDF_MAGIC) == -1:
        raise CorruptPDFError(f"'{name}' is not a valid PDF (missing %PDF header).")
    try:
        return fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several unrelated types here
        raise CorruptPDFError(f"'{name}' is corrupted or unreadable: {exc}") from exc


def _extract_pages(
    document: "fitz.Document", strip_diacritics: bool
) -> tuple[list[PageContent], list[int], list[int]]:
    """Extract cleaned text page by page.

    A single broken page must not discard a 300-page report, so per-page
    failures are logged and recorded in ``skipped_pages`` instead of aborting.
    """
    pages: list[PageContent] = []
    skipped: list[int] = []
    image_only: list[int] = []

    for index in range(document.page_count):
        page_number = index + 1
        try:
            page = document.load_page(index)
            # Default reading order (sort=False) follows the content stream,
            # which is correct for RTL text. sort=True would reorder blocks
            # strictly left-to-right and scramble Arabic lines.
            raw = page.get_text("text")
            cleaned = normalize_text(raw, strip_diacritics)
            if not cleaned and page.get_images():
                image_only.append(page_number)
        except Exception as exc:
            logger.warning("Skipping page %d: extraction failed (%s).", page_number, exc)
            skipped.append(page_number)
            continue

        if cleaned:
            pages.append(PageContent(page_number=page_number, text=cleaned))

    return pages, skipped, image_only


def load_pdf(
    source: Union[str, Path, bytes, bytearray],
    *,
    source_name: Optional[str] = None,
    max_size_bytes: Optional[int] = None,
    max_pages: Optional[int] = None,
    strip_diacritics: Optional[bool] = None,
) -> LoadedDocument:
    """Load a PDF and return its cleaned text, page by page.

    Args:
        source: File path or raw bytes (e.g. ``await upload.read()``).
        source_name: Display name for citations; defaults to the file name.
        max_size_bytes / max_pages / strip_diacritics: Override the settings.

    Raises:
        PDFLoadError: Missing file or unreadable path.
        PDFTooLargeError: Over the size or page limit.
        CorruptPDFError: Empty, truncated, or not a PDF.
        EncryptedPDFError: Password protected.
        EmptyDocumentError: No extractable text (typically a scanned PDF).
    """
    settings = get_settings()
    size_limit = max_size_bytes if max_size_bytes is not None else settings.max_upload_size_bytes
    page_limit = max_pages if max_pages is not None else settings.pdf_max_pages
    strip = (
        strip_diacritics if strip_diacritics is not None else settings.strip_arabic_diacritics
    )

    data, name = _read_source(source, source_name, size_limit)
    document = _open_document(data, name)

    try:
        if document.needs_pass and not document.authenticate(""):
            # Many PDFs are "encrypted" with an empty user password (only
            # editing is restricted); authenticate("") opens those. Real
            # password protection ends here.
            raise EncryptedPDFError(f"'{name}' is password protected.")
        if document.page_count == 0:
            raise EmptyDocumentError(f"'{name}' has no pages.")
        if document.page_count > page_limit:
            raise PDFTooLargeError(
                f"'{name}' has {document.page_count} pages; the limit is {page_limit}."
            )

        pages, skipped, image_only = _extract_pages(document, strip)
        page_count = document.page_count
        raw_meta = document.metadata or {}
        pdf_metadata = {
            key: str(raw_meta[key]).strip()
            for key in ("title", "author", "subject", "creationDate")
            if raw_meta.get(key) and str(raw_meta[key]).strip()
        }
    except PDFLoadError:
        raise
    except Exception as exc:
        raise CorruptPDFError(f"'{name}' could not be parsed: {exc}") from exc
    finally:
        document.close()

    if not pages:
        if image_only:
            raise EmptyDocumentError(
                f"'{name}' contains only images (scanned document). "
                "Run OCR on it first, then upload the searchable PDF."
            )
        raise EmptyDocumentError(f"'{name}' contains no extractable text.")

    logger.info(
        "Loaded '%s': %d/%d pages with text, %d skipped, %d image-only.",
        name, len(pages), page_count, len(skipped), len(image_only),
    )
    return LoadedDocument(
        doc_id=hashlib.sha256(data).hexdigest()[:32],
        source_name=name,
        page_count=page_count,
        pages=pages,
        pdf_metadata=pdf_metadata,
        skipped_pages=skipped,
        image_only_pages=image_only,
    )
