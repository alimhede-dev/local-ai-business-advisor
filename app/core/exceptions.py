"""Application-specific exceptions.

Why a dedicated hierarchy?
--------------------------
The RAG pipeline touches many fragile things: user-supplied PDFs, a local
Ollama server, an on-disk vector database. Each can fail in different ways.
If every layer leaked its own library exceptions (``fitz.FileDataError``,
``httpx.ConnectError``, ``chromadb.errors...``), the API layer would need to
import all of those libraries just to decide between "400 Bad Request" and
"503 Service Unavailable".

Instead, the RAG modules translate low-level errors into the classes below,
always chaining the original (``raise ... from exc``) so debugging
information is never lost. Later, API routes can map:

* ``PDFLoadError`` family  -> HTTP 400 / 413 / 422 (the *user's file* is the problem)
* ``EmbeddingError``       -> HTTP 503 (Ollama / model is unavailable)
* ``VectorStoreError``     -> HTTP 500 / 503 (our storage layer is the problem)
"""


class RAGError(Exception):
    """Base class for every error raised by the RAG pipeline."""


# --- Document loading (problems with the uploaded file) ----------------------
class PDFLoadError(RAGError):
    """The PDF could not be loaded for a reason described in the message."""


class CorruptPDFError(PDFLoadError):
    """The file is empty, truncated, not a real PDF, or structurally broken."""


class EncryptedPDFError(PDFLoadError):
    """The PDF is password protected and cannot be read without the password."""


class PDFTooLargeError(PDFLoadError):
    """The PDF exceeds the configured size or page-count limit."""


class EmptyDocumentError(PDFLoadError):
    """The PDF is valid but contains no extractable text (e.g. scanned images)."""


# --- Chunking ----------------------------------------------------------------
class ChunkingError(RAGError):
    """Text could not be split into usable chunks."""


# --- Embeddings / vector store / retrieval -----------------------------------
class EmbeddingError(RAGError):
    """The embedding backend (Ollama / HuggingFace) failed or is unreachable."""


class VectorStoreError(RAGError):
    """ChromaDB could not be initialised, written to, or queried."""


class RetrievalError(RAGError):
    """A retrieval request was invalid or could not be completed."""


# =============================================================================
# Conversation memory
# =============================================================================
class ConversationMemoryError(Exception):
    """Base class for every error raised by the memory system."""


class SessionNotFoundError(ConversationMemoryError):
    """The referenced chat session does not exist."""


class InvalidMessageError(ConversationMemoryError):
    """A message was rejected (empty content or a role clients may not write)."""


class MemoryStorageError(ConversationMemoryError):
    """Reading or writing memory in SQLite failed (the transaction was rolled back)."""


class SummarizationError(ConversationMemoryError):
    """The local LLM could not produce a usable summary."""


# =============================================================================
# Prompts
# =============================================================================
class PromptBuildError(Exception):
    """A prompt could not be assembled from the given inputs."""
