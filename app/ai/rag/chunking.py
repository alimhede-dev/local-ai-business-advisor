"""Text chunking tuned for Arabic and English.

WHY (architecture decisions)
----------------------------
1. **Chunks are the unit of retrieval.** Too large and one chunk mixes topics,
   diluting its embedding; too small and the LLM loses context. 800 characters
   with 150 overlap (settings ``RAG_CHUNK_SIZE`` / ``RAG_CHUNK_OVERLAP``) keeps
   roughly one paragraph per chunk, and the overlap stops a sentence that
   straddles a boundary from being lost to both neighbours.

2. **Arabic-aware separators.** LangChain's default separators know only the
   Latin ``?`` and ``,``. Arabic uses ``؟`` (U+061F), ``،`` (U+060C) and ``؛``
   (U+061B). Without them the splitter would fall through to cutting at spaces
   mid-sentence. Separators are tried from the largest unit (paragraph) to the
   smallest (word), and ``keep_separator="end"`` leaves the punctuation attached
   to the sentence it closes.

3. **Split per page, not per document.** This keeps an exact page number on
   every chunk, which is what lets the advisor cite "report.pdf, page 7".
   The trade-off is that no chunk spans a page break; that is acceptable for
   business documents and far better than un-citable answers.

4. **Deterministic chunk IDs.** ``{doc_id}-p{page}-c{index}`` means the same
   input always produces the same IDs, which makes re-indexing idempotent.

5. **Language tag per chunk.** Cheap script-ratio detection (no extra
   dependency) is stored as metadata so later stages can filter or log by
   language.
"""

import logging
import re
from typing import Literal, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ai.rag.loader import LoadedDocument
from app.core.config import get_settings
from app.core.exceptions import ChunkingError

logger = logging.getLogger(__name__)

Language = Literal["ar", "en", "mixed", "unknown"]

# Ordered from the largest semantic unit to the smallest. The splitter uses the
# first separator that appears in the text and only recurses into the next one
# when a piece is still too large.
SEPARATORS: list[str] = [
    "\n\n",
    "\n",
    "۔ ",   # Urdu/Arabic-script full stop (appears in mixed-region documents)
    ". ",
    "؟ ",
    "? ",
    "! ",
    "؛ ",
    "; ",
    "، ",
    ", ",
    " ",
    "",
]

# Arabic, Arabic Supplement and Arabic Extended-A blocks.
_ARABIC_LETTER_RE = re.compile("[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
_LATIN_LETTER_RE = re.compile("[A-Za-z]")


def detect_language(text: str) -> Language:
    """Classify text by script: ``ar``, ``en`` (Latin), ``mixed`` or ``unknown``.

    A 70% threshold tolerates the English acronyms ("ROI", "KPI") and numbers
    that are common inside Arabic business text without calling it "mixed".
    """
    arabic = len(_ARABIC_LETTER_RE.findall(text))
    latin = len(_LATIN_LETTER_RE.findall(text))
    total = arabic + latin
    if total == 0:
        return "unknown"
    ratio = arabic / total
    if ratio >= 0.7:
        return "ar"
    if ratio <= 0.3:
        return "en"
    return "mixed"


def contains_arabic(text: str) -> bool:
    """True if the text has at least one Arabic letter."""
    return _ARABIC_LETTER_RE.search(text) is not None


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Create the configured splitter.

    Raises:
        ChunkingError: If the geometry is impossible (overlap >= size).
    """
    if chunk_size <= 0:
        raise ChunkingError("chunk_size must be positive.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ChunkingError(
            f"chunk_overlap ({chunk_overlap}) must be >= 0 and smaller than "
            f"chunk_size ({chunk_size})."
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Characters (not tokens) so no tokenizer dependency is needed; 800
        # Arabic characters stay far below nomic-embed-text's context window.
        length_function=len,
        separators=SEPARATORS,
        keep_separator="end",
        is_separator_regex=False,
        add_start_index=True,
        strip_whitespace=True,
    )


def split_document(
    document: LoadedDocument,
    *,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    min_chunk_chars: Optional[int] = None,
) -> list[Document]:
    """Split a loaded PDF into metadata-rich LangChain ``Document`` chunks.

    Every chunk carries: ``doc_id``, ``source``, ``page``, ``total_pages``,
    ``chunk_index``, ``chunk_id``, ``language`` and ``start_index`` (all
    primitives, because ChromaDB rejects lists/None in metadata).

    Chunks shorter than ``min_chunk_chars`` (page numbers, stray headers) are
    dropped: their embeddings are noise that can outrank real content.

    Raises:
        ChunkingError: Invalid geometry, or the document yielded no chunks.
    """
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.rag_chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.rag_chunk_overlap
    min_chars = (
        min_chunk_chars if min_chunk_chars is not None else settings.rag_min_chunk_chars
    )
    splitter = build_splitter(size, overlap)

    chunks: list[Document] = []
    for page in document.pages:
        try:
            page_chunks = splitter.create_documents(
                texts=[page.text],
                metadatas=[{"page": page.page_number}],
            )
        except Exception as exc:
            raise ChunkingError(
                f"Failed to split page {page.page_number} of '{document.source_name}': {exc}"
            ) from exc

        for piece in page_chunks:
            text = piece.page_content.strip()
            if len(text) < min_chars:
                continue
            index = len(chunks)
            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_id": document.doc_id,
                        "source": document.source_name,
                        "page": page.page_number,
                        "total_pages": document.page_count,
                        "chunk_index": index,
                        "chunk_id": f"{document.doc_id}-p{page.page_number}-c{index}",
                        "language": detect_language(text),
                        "start_index": int(piece.metadata.get("start_index", 0)),
                    },
                )
            )

    if not chunks:
        raise ChunkingError(
            f"'{document.source_name}' produced no chunks of at least {min_chars} characters."
        )
    logger.info(
        "Split '%s' into %d chunks (size=%d, overlap=%d).",
        document.source_name, len(chunks), size, overlap,
    )
    return chunks
